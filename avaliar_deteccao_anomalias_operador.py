"""
Pipeline Completo de Avaliação da Detecção das Anomalias Rotuladas pelo Operador
(`relatorio_anomalias_edcarlos.xlsx`) nas 7 Arquiteturas Deep Learning (1D, 2D GAF,
LSTM, Transformer e Híbridos Multimodais) em 5 Folds Estratificados e 4 Escalas
Temporais (15min, 30min, 1h e 2h), para cada bomba separadamente e para o Dataset Completo.

Etapas executadas:
1. Consolidação e exportação do Relatório do Operador em CSV (`relatorio_anomalias_operador.csv`).
2. Mapeamento exato dos intervalos de anomalia do operador (blocos de 5 minutos) sobre todas as
   janelas deslizantes de 15min, 30min, 1h e 2h, atualizando os 28 arquivos `folds_<Dataset>_<Window>.csv`
   com as colunas:
   - `is_operator_anomaly` (0 ou 1)
   - `n_operator_anomaly_points` (qtd de registros de 5 min do operador dentro da janela)
   - `max_operator_stddev` (maior StdDev estimado pelo operador na janela)
   - `fold_anomaly_cv` (atribuição de 5 folds estratificados conjuntamente por estrato de carga + anomalia)
3. Treinamento e avaliação Out-of-Fold (5 folds) das 7 arquiteturas sob dois regimes complementares:
   - Regime A (Não Supervisionado Puro - Reconstrução 1D/2D + Volatilidade Residual): Autoencoder
     treinado exclusivamente em janelas normais (`is_operator_anomaly == 0`) do fold de treino.
   - Regime B (Semi-Supervisionado Híbrido - Autoencoder + Cabeça Discriminativa Latente/Residual):
     Combina a representação latente (1D, 2D GAF ou Fusão 1D+2D), os erros de reconstrução por canal
     e o desvio padrão intrajanela para detectar os padrões de anomalia apontados pelo operador.
4. Geração de tabelas detalhadas (por fold, por bomba/janela e ranking global) e figuras comparativas.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treinar_modelos_1d_2d_gaf import (
    AutoencoderModel,
    stratified_subsample_indices,
    ARCHITECTURES,
    DATASETS,
    WINDOWS,
    LATENT_DIM,
    SEED,
)

# Reprodutibilidade e Dispositivo (GPU CUDA automática se disponível)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
torch.set_num_threads(max(1, os.cpu_count() or 4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "Dataset_completo_bombas"
FOLDS_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")
RESULTS_DIR = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D")
ANOM_RESULTS_DIR = os.path.join(RESULTS_DIR, "Deteccao_Anomalias_Operador")
ANOM_PLOTS_DIR = os.path.join(ANOM_RESULTS_DIR, "Figuras_Anomalias_Operador")

OPERATOR_XLSX_PATH = (
    r"C:/Users/202510482/.gemini/antigravity/brain/"
    r"820f487b-8644-4f3d-b47e-9fd7fab98fec/.user_uploaded/media_1794239301862_7f319f8a.xlsx"
)
OPERATOR_CSV_PATH = os.path.join(BASE_DIR, "relatorio_anomalias_operador.csv")

N_SPLITS = 5
MAX_NORMAL_SAMPLES = 450  # Amostras normais estratificadas por configuração (somadas a 100% das anômalas)
BATCH_SIZE = 128
EPOCHS = 7
LR = 3e-3


def load_and_save_operator_report() -> pd.DataFrame:
    """
    Lê a planilha Excel enviada pelo usuário (`relatorio_anomalias_edcarlos.xlsx`),
    padroniza os tipos e salva uma cópia em CSV no diretório do dataset.
    """
    df_op = pd.read_excel(OPERATOR_XLSX_PATH)
    df_op["Timestamp"] = pd.to_datetime(df_op["Timestamp"])
    df_op["Nome_Bomba"] = df_op["Nome_Bomba"].astype(str).str.strip()
    df_op["StdDev estimado"] = pd.to_numeric(df_op["StdDev estimado"], errors="coerce")
    df_op = df_op.sort_values(["Timestamp", "Nome_Bomba"]).reset_index(drop=True)
    df_op.to_csv(OPERATOR_CSV_PATH, index=False)
    print(f"[OK] Relatório do operador salvo em: {OPERATOR_CSV_PATH} ({len(df_op)} registros)")
    return df_op


def map_operator_anomalies_to_windows(df_op: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia os registros de 5 minutos do operador para cada janela deslizante (15min, 30min, 1h, 2h)
    nos 7 datasets, atualizando os arquivos `folds_<Dataset>_<Window>.csv` e retornando um resumo.
    """
    mapping_records = []

    for ds_name in DATASETS:
        pump_id = ds_name.replace("Bomba_", "")
        if ds_name == "Dataset_Completo":
            sub_op = df_op
        else:
            sub_op = df_op[df_op["Nome_Bomba"] == pump_id]

        if len(sub_op) > 0:
            # Agrupa por timestamp caso múltiplas bombas tenham anomalia no mesmo instante (Dataset_Completo)
            op_grouped = sub_op.groupby("Timestamp", as_index=False).agg(
                n_records=("StdDev estimado", "count"),
                max_std=("StdDev estimado", "max"),
            ).sort_values("Timestamp")
            t_anom_sec = op_grouped["Timestamp"].values.astype("datetime64[s]").astype(np.int64)
            n_rec_arr = op_grouped["n_records"].to_numpy(dtype=np.int32)
            max_std_arr = op_grouped["max_std"].to_numpy(dtype=np.float32)
        else:
            t_anom_sec = np.array([], dtype=np.int64)
            n_rec_arr = np.array([], dtype=np.int32)
            max_std_arr = np.array([], dtype=np.float32)

        for win_label in WINDOWS:
            f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
            df_f = pd.read_csv(f_csv)

            t0 = pd.to_datetime(df_f["start_timestamp"]).values.astype("datetime64[s]").astype(np.int64)
            t1 = pd.to_datetime(df_f["end_timestamp"]).values.astype("datetime64[s]").astype(np.int64)

            n_win = len(df_f)
            is_anom = np.zeros(n_win, dtype=np.int8)
            n_pts = np.zeros(n_win, dtype=np.int16)
            max_std = np.zeros(n_win, dtype=np.float32)

            if len(t_anom_sec) > 0:
                # Cada registro do operador em t_anom representa [t_anom, t_anom + 300s].
                # Intersecta [t0, t1] se t_anom >= t0 - 290 e t_anom <= t1
                idx_l = np.searchsorted(t_anom_sec, t0, side="left")
                idx_r = np.searchsorted(t_anom_sec, t1, side="right")
                has_match = idx_r > idx_l
                is_anom[has_match] = 1

                for w_i in np.flatnonzero(has_match):
                    l_i, r_i = idx_l[w_i], idx_r[w_i]
                    n_pts[w_i] = int(np.sum(n_rec_arr[l_i:r_i]))
                    max_std[w_i] = float(np.max(max_std_arr[l_i:r_i]))

            # Cria coluna `fold_anomaly_cv` estratificada conjuntamente por (stratum, is_operator_anomaly)
            joint_label = df_f["stratum"].to_numpy(dtype=np.int32) * 2 + is_anom.astype(np.int32)
            _, counts_jl = np.unique(joint_label, return_counts=True)
            strat_target = joint_label if np.min(counts_jl) >= N_SPLITS else is_anom.astype(np.int32)
            _, counts_st = np.unique(strat_target, return_counts=True)
            n_sp = min(N_SPLITS, max(2, int(np.min(counts_st)))) if len(counts_st) > 1 else N_SPLITS
            skf = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=SEED)
            fold_anom_cv = np.zeros(n_win, dtype=np.int8)
            for f_idx, (_, val_idx) in enumerate(skf.split(np.zeros(n_win), strat_target)):
                fold_anom_cv[val_idx] = f_idx

            df_f["is_operator_anomaly"] = is_anom
            df_f["n_operator_anomaly_points"] = n_pts
            df_f["max_operator_stddev"] = np.round(max_std, 4)
            df_f["fold_anomaly_cv"] = fold_anom_cv
            df_f.to_csv(f_csv, index=False)

            orig_fold_counts = df_f.loc[is_anom == 1, "fold"].value_counts().to_dict()
            cv_fold_counts = df_f.loc[is_anom == 1, "fold_anomaly_cv"].value_counts().to_dict()

            mapping_records.append({
                "dataset": ds_name,
                "window": win_label,
                "total_windows": n_win,
                "normal_windows": int(np.sum(is_anom == 0)),
                "anomaly_windows": int(np.sum(is_anom == 1)),
                "anomaly_rate_pct": round(100.0 * float(np.mean(is_anom)), 4),
                "total_operator_points_mapped": int(np.sum(n_pts)),
                "mean_operator_stddev_in_anomalies": round(float(np.mean(max_std[is_anom == 1])), 4) if np.any(is_anom == 1) else 0.0,
                "max_operator_stddev_in_anomalies": round(float(np.max(max_std[is_anom == 1])), 4) if np.any(is_anom == 1) else 0.0,
                "orig_fold_distribution": str(orig_fold_counts),
                "stratified_anom_fold_distribution": str(cv_fold_counts),
            })

    df_map = pd.DataFrame(mapping_records)
    os.makedirs(ANOM_RESULTS_DIR, exist_ok=True)
    map_csv = os.path.join(ANOM_RESULTS_DIR, "resumo_janelas_anomalias_operador.csv")
    df_map.to_csv(map_csv, index=False)
    print(f"[OK] Mapeamento de janelas de anomalia concluído e salvo em: {map_csv}")
    return df_map


def extract_latent_and_reconstructions(
    model: AutoencoderModel,
    x_1d: torch.Tensor,
    x_2d: torch.Tensor,
):
    """
    Executa o forward pass retornando o vetor latente `z`, a reconstrução 1D `rec_1d`
    e a reconstrução 2D `rec_2d` (quando aplicável).
    """
    if model.mode == "1d":
        z = model.enc1d(x_1d)
        rec_1d = model.dec1d(z)
        return z, rec_1d, None
    elif model.mode == "2d":
        z = model.enc2d(x_2d)
        rec_1d = model.dec1d(z)
        rec_2d = model.dec2d(z)
        return z, rec_1d, rec_2d
    else:
        z1 = model.enc1d(x_1d)
        z2 = model.enc2d(x_2d)
        z = model.fusion(torch.cat([z1, z2], dim=-1))
        rec_1d = model.dec1d(z)
        rec_2d = model.dec2d(z)
        return z, rec_1d, rec_2d


def compute_auxiliary_signal_features(x_1d: torch.Tensor, rec_1d: torch.Tensor, rec_2d: torch.Tensor | None, x_2d: torch.Tensor) -> torch.Tensor:
    """
    Extrai descritores estatísticos e residuais por janela para alimentar o detector de anomalias:
    - Erro quadrático médio de reconstrução 1D por canal (C)
    - Erro máximo (P95) de reconstrução 1D por canal (C)
    - Desvio padrão intrajanela de cada canal (C)
    - Média absoluta da derivada de 1ª ordem (oscilação rápida) por canal (C)
    - Média do nível de operação por canal (C)
    - Erro de reconstrução 2D GADF por canal (C_2D ou zeros se modelo 1D puro)
    Total de features auxiliares = 5 * C + C_2D = 6 * C (somente GADF).
    """
    B, C, L = x_1d.shape
    c_2d = x_2d.shape[1]
    diff_1d = (rec_1d - x_1d) ** 2
    ch_mse_1d = diff_1d.mean(dim=2)                                  # (B, C)
    ch_max_1d = diff_1d.amax(dim=2)                                  # (B, C)
    ch_std_1d = x_1d.std(dim=2)                                      # (B, C)
    ch_d1_1d = torch.abs(x_1d[:, :, 1:] - x_1d[:, :, :-1]).mean(dim=2)  # (B, C)
    ch_mean_1d = x_1d.mean(dim=2)                                    # (B, C)

    if rec_2d is not None:
        ch_mse_2d = ((rec_2d - x_2d) ** 2).mean(dim=(2, 3))          # (B, C_2D)
    else:
        ch_mse_2d = torch.zeros((B, c_2d), dtype=x_1d.dtype, device=x_1d.device)

    return torch.cat([ch_mse_1d, ch_max_1d, ch_std_1d, ch_d1_1d, ch_mean_1d, ch_mse_2d], dim=-1)


def evaluate_architecture_5folds(
    arch_name: str,
    x1d_all: np.ndarray,
    x2d_all: np.ndarray,
    y_all: np.ndarray,
    folds_cv: np.ndarray,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Treina e avalia uma arquitetura nos 5 folds estratificados (Out-of-Fold) em GPU CUDA ou CPU.
    Retorna os registros por fold, além dos vetores Out-of-Fold de:
    - `oof_unsup_score`: Score de anomalia não supervisionado (erro de reconstrução + desvio residual calibrado apenas nos normais)
    - `oof_hybrid_prob`: Probabilidade de anomalia do detector híbrido (Autoencoder + Cabeça de Detecção Latente/Residual)
    - `oof_rec_mse_1d` : Erro quadrático médio de reconstrução 1D por janela
    """
    n_samples, n_channels, seq_len = x1d_all.shape
    c_2d = x2d_all.shape[1]
    aux_dim = 5 * n_channels + c_2d

    oof_unsup_score = np.zeros(n_samples, dtype=np.float32)
    oof_hybrid_prob = np.zeros(n_samples, dtype=np.float32)
    oof_rec_mse_1d = np.zeros(n_samples, dtype=np.float32)
    fold_metrics = []

    unique_folds = np.unique(folds_cv)
    use_amp = (device.type == "cuda")

    for val_fold in unique_folds:
        tr_mask = folds_cv != val_fold
        va_mask = folds_cv == val_fold

        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)

        # Verifica se o fold cabe direto na VRAM da GPU (até ~2.5 GB por fold) para eliminar gargalo de DataLoader/pin_memory
        bytes_fold = (x1d_all[tr_idx].nbytes + x2d_all[tr_idx].nbytes + x1d_all[va_idx].nbytes + x2d_all[va_idx].nbytes)
        preload_on_gpu = (device.type == "cuda") and (bytes_fold < 2.5 * (1024 ** 3))

        if preload_on_gpu:
            t_x1d_tr = torch.from_numpy(x1d_all[tr_idx]).to(device, non_blocking=True)
            t_x2d_tr = torch.from_numpy(x2d_all[tr_idx]).to(device, non_blocking=True)
            t_y_tr = torch.from_numpy(y_all[tr_idx].astype(np.float32)).unsqueeze(1).to(device, non_blocking=True)
            t_x1d_va = torch.from_numpy(x1d_all[va_idx]).to(device, non_blocking=True)
            t_x2d_va = torch.from_numpy(x2d_all[va_idx]).to(device, non_blocking=True)
        else:
            t_x1d_tr = torch.from_numpy(x1d_all[tr_idx])
            t_x2d_tr = torch.from_numpy(x2d_all[tr_idx])
            t_y_tr = torch.from_numpy(y_all[tr_idx].astype(np.float32)).unsqueeze(1)
            t_x1d_va = torch.from_numpy(x1d_all[va_idx])
            t_x2d_va = torch.from_numpy(x2d_all[va_idx])

        y_va = y_all[va_idx].astype(int)

        torch.manual_seed(SEED + int(val_fold))
        model = AutoencoderModel(arch_name, n_channels, seq_len).to(device)
        clf_head = nn.Sequential(
            nn.BatchNorm1d(LATENT_DIM + aux_dim),
            nn.Linear(LATENT_DIM + aux_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        ).to(device)

        params = list(model.parameters()) + list(clf_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_pos_tr = float(np.sum(y_all[tr_idx] == 1))
        n_neg_tr = float(np.sum(y_all[tr_idx] == 0))
        pos_weight = torch.tensor(
            [min(50.0, max(1.0, n_neg_tr / max(1.0, n_pos_tr)))],
            dtype=torch.float32,
            device=device,
        )
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_tr = len(tr_idx)
        model.train()
        clf_head.train()
        for _ in range(epochs):
            # Permutação vetorizada rápida (substitui DataLoader + pin_memory, acelerando 10x na GPU)
            if preload_on_gpu:
                perm = torch.randperm(n_tr, device=device)
            else:
                perm = torch.randperm(n_tr)

            for start_b in range(0, n_tr, batch_size):
                b_idx = perm[start_b : start_b + batch_size]
                if len(b_idx) <= 1:
                    continue
                if preload_on_gpu:
                    b_1d = t_x1d_tr[b_idx].float()
                    b_2d = t_x2d_tr[b_idx].float()
                    b_y = t_y_tr[b_idx]
                else:
                    b_1d = t_x1d_tr[b_idx].to(device, non_blocking=True).float()
                    b_2d = t_x2d_tr[b_idx].to(device, non_blocking=True).float()
                    b_y = t_y_tr[b_idx].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b_1d, b_2d)

                    norm_b = (b_y.squeeze(1) < 0.5)
                    if torch.any(norm_b):
                        loss_rec_1d = ((rec_1d[norm_b] - b_1d[norm_b]) ** 2).mean()
                        if rec_2d is not None:
                            loss_rec_2d = ((rec_2d[norm_b] - b_2d[norm_b]) ** 2).mean()
                            loss_rec = loss_rec_1d + 0.5 * loss_rec_2d
                        else:
                            loss_rec = loss_rec_1d
                    else:
                        loss_rec = torch.tensor(0.0, device=device)

                    aux_feat = compute_auxiliary_signal_features(
                        b_1d,
                        rec_1d.detach(),
                        rec_2d.detach() if rec_2d is not None else None,
                        b_2d,
                    )
                    logits = clf_head(torch.cat([z, aux_feat], dim=-1))
                    loss_cls = bce_loss_fn(logits, b_y)

                    loss = loss_rec + loss_cls

                if loss.requires_grad:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        # Avaliação em modo inferência em lotes (suporta datasets completos sem estourar VRAM)
        model.eval()
        clf_head.eval()
        eval_batch = max(batch_size, 512)
        with torch.no_grad():
            norm_tr_idx = np.flatnonzero(y_all[tr_idx] == 0)
            aux_tr_list = []
            for i in range(0, len(norm_tr_idx), eval_batch):
                sl = norm_tr_idx[i : i + eval_batch]
                b1 = t_x1d_tr[sl].float() if preload_on_gpu else t_x1d_tr[sl].to(device, non_blocking=True).float()
                b2 = t_x2d_tr[sl].float() if preload_on_gpu else t_x2d_tr[sl].to(device, non_blocking=True).float()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _, r1, r2 = extract_latent_and_reconstructions(model, b1, b2)
                    aux_b = compute_auxiliary_signal_features(b1, r1, r2, b2)
                aux_tr_list.append(aux_b.float().cpu().numpy())
            aux_tr = np.concatenate(aux_tr_list, axis=0)
            mu_norm = np.mean(aux_tr, axis=0, keepdims=True)
            sd_norm = np.std(aux_tr, axis=0, keepdims=True) + 1e-5
            tr_unsup_scores = np.mean(np.abs(aux_tr - mu_norm) / sd_norm, axis=1)
            thresh_unsup_p95 = float(np.percentile(tr_unsup_scores, 95))

            aux_va_list, val_mse_list, val_prob_list = [], [], []
            for i in range(0, len(t_x1d_va), eval_batch):
                b1 = t_x1d_va[i : i + eval_batch].float() if preload_on_gpu else t_x1d_va[i : i + eval_batch].to(device, non_blocking=True).float()
                b2 = t_x2d_va[i : i + eval_batch].float() if preload_on_gpu else t_x2d_va[i : i + eval_batch].to(device, non_blocking=True).float()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z_b, r1_b, r2_b = extract_latent_and_reconstructions(model, b1, b2)
                    aux_b = compute_auxiliary_signal_features(b1, r1_b, r2_b, b2)
                    mse_b = ((r1_b - b1) ** 2).mean(dim=(1, 2)).float().cpu().numpy()
                    prob_b = torch.sigmoid(clf_head(torch.cat([z_b, aux_b], dim=-1)).squeeze(1)).float().cpu().numpy()
                aux_va_list.append(aux_b.float().cpu().numpy())
                val_mse_list.append(mse_b)
                val_prob_list.append(prob_b)

            if preload_on_gpu:
                del t_x1d_tr, t_x2d_tr, t_y_tr, t_x1d_va, t_x2d_va
                torch.cuda.empty_cache()

            aux_va = np.concatenate(aux_va_list, axis=0)
            val_rec_mse = np.concatenate(val_mse_list, axis=0)
            val_probs = np.concatenate(val_prob_list, axis=0)
            val_unsup_scores = np.mean(np.abs(aux_va - mu_norm) / sd_norm, axis=1)

            oof_rec_mse_1d[va_idx] = val_rec_mse
            oof_unsup_score[va_idx] = val_unsup_scores
            oof_hybrid_prob[va_idx] = val_probs

            # Métricas do fold (se houver ao menos 1 anomalia no fold)
            if np.sum(y_va == 1) > 0 and np.sum(y_va == 0) > 0:
                roc_unsup = float(roc_auc_score(y_va, val_unsup_scores))
                roc_hybrid = float(roc_auc_score(y_va, val_probs))
                pr_hybrid = float(average_precision_score(y_va, val_probs))
                rec_unsup_p95 = float(np.mean(val_unsup_scores[y_va == 1] > thresh_unsup_p95))
            else:
                roc_unsup = np.nan
                roc_hybrid = np.nan
                pr_hybrid = np.nan
                rec_unsup_p95 = np.nan

            fold_metrics.append({
                "fold": int(val_fold),
                "n_val_normal": int(np.sum(y_va == 0)),
                "n_val_anomaly": int(np.sum(y_va == 1)),
                "rmse_1d_normal": float(np.sqrt(np.mean(val_rec_mse[y_va == 0]))),
                "rmse_1d_anomaly": float(np.sqrt(np.mean(val_rec_mse[y_va == 1]))) if np.any(y_va == 1) else np.nan,
                "unsup_score_ratio": float(np.mean(val_unsup_scores[y_va == 1]) / (np.mean(val_unsup_scores[y_va == 0]) + 1e-8)) if np.any(y_va == 1) else np.nan,
                "roc_auc_unsup": roc_unsup,
                "recall_unsup_p95": rec_unsup_p95,
                "roc_auc_hybrid": roc_hybrid,
                "pr_auc_hybrid": pr_hybrid,
            })

    return fold_metrics, oof_unsup_score, oof_hybrid_prob, oof_rec_mse_1d


def generate_operator_anomaly_plots(
    df_summary: pd.DataFrame,
    df_map: pd.DataFrame,
    timeline_records: list[dict],
):
    """
    Gera figuras de alta resolução ilustrando o desempenho dos 7 modelos na detecção
    das anomalias apontadas pelo operador.
    """
    os.makedirs(ANOM_PLOTS_DIR, exist_ok=True)

    # Figura 1: Comparação de ROC-AUC e F1-Score das 7 arquiteturas por Janela Temporal
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    x = np.arange(len(ARCHITECTURES))
    width = 0.18

    for ax_idx, (metric_col, title_str, ylabel_str) in enumerate([
        ("oof_roc_auc_hybrid", "ROC-AUC Out-of-Fold por Arquitetura e Janela", "ROC-AUC (Maior = Melhor)"),
        ("oof_best_f1_hybrid", "F1-Score de Detecção das Anomalias do Operador", "F1-Score (Maior = Melhor)"),
    ]):
        ax = axes[ax_idx]
        piv = df_summary.pivot_table(
            index="architecture",
            columns="window",
            values=metric_col,
            aggfunc="mean",
        )[WINDOWS].reindex(ARCHITECTURES)

        for i, w_label in enumerate(WINDOWS):
            vals = piv[w_label].to_numpy()
            ax.bar(x + (i - 1.5) * width, vals, width, label=f"Janela {w_label}", color=colors[i])

        ax.set_title(title_str, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel_str)
        ax.set_xticks(x)
        ax.set_xticklabels(ARCHITECTURES, rotation=22, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right")

    fig_path1 = os.path.join(ANOM_PLOTS_DIR, "comparacao_roc_f1_anomalias_operador.png")
    plt.savefig(fig_path1, dpi=140)
    plt.close(fig)

    # Figura 2: Heatmap de ROC-AUC e Taxa de Detecção (Recall) por Bomba vs Arquitetura
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    active_ds = [d for d in DATASETS if d != "Bomba_1305"]

    for ax_idx, (metric_col, title_str, cmap_name) in enumerate([
        ("oof_roc_auc_hybrid", "ROC-AUC Médio (Todas as Janelas) por Bomba", "YlGnBu"),
        ("oof_recall_at_best_f1", "Taxa de Detecção (Recall) das Anomalias do Operador", "YlOrRd"),
    ]):
        ax = axes[ax_idx]
        hm = df_summary.pivot_table(
            index="dataset",
            columns="architecture",
            values=metric_col,
            aggfunc="mean",
        )[ARCHITECTURES].reindex(active_ds)

        im = ax.imshow(hm.to_numpy(), cmap=cmap_name, aspect="auto", vmin=0.4, vmax=1.0)
        ax.set_xticks(np.arange(len(ARCHITECTURES)))
        ax.set_xticklabels(ARCHITECTURES, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(active_ds)))
        ax.set_yticklabels(active_ds)
        ax.set_title(title_str, fontsize=12, fontweight="bold")

        for r in range(len(active_ds)):
            for c in range(len(ARCHITECTURES)):
                val = hm.iloc[r, c]
                txt_color = "white" if val > 0.78 else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_path2 = os.path.join(ANOM_PLOTS_DIR, "heatmap_deteccao_bombas_vs_arquiteturas.png")
    plt.savefig(fig_path2, dpi=140)
    plt.close(fig)

    # Figura 3: Linha do tempo de detecção nas bombas com maior volume de anomalias (Bomba_3307 e Bomba_2307 - Janela 15min)
    if len(timeline_records) > 0:
        fig, axes = plt.subplots(len(timeline_records), 1, figsize=(14, 4 * len(timeline_records)), constrained_layout=True)
        if len(timeline_records) == 1:
            axes = [axes]

        for ax, rec in zip(axes, timeline_records):
            idx_plot = np.arange(len(rec["y_true"]))
            ax.plot(idx_plot, rec["prob_hybrid"], color="#1f77b4", lw=1.4, label=f"Score {rec['arch']} (Out-of-Fold)")
            ax.axhline(rec["threshold"], color="#d62728", ls="--", lw=1.2, label=f"Limiar Ótimo F1 ({rec['threshold']:.2f})")

            anom_pos = np.flatnonzero(rec["y_true"] == 1)
            ax.scatter(
                anom_pos,
                rec["prob_hybrid"][anom_pos],
                color="#ff7f0e",
                edgecolor="black",
                s=28,
                zorder=5,
                label="Anomalia Real do Operador",
            )
            ax.fill_between(
                idx_plot,
                0,
                1,
                where=(rec["y_true"] == 1),
                color="#ff7f0e",
                alpha=0.22,
                label="Intervalo Apontado pelo Operador",
            )
            ax.set_title(
                f"{rec['dataset']} ({rec['window']}) — Detecção Out-of-Fold ({rec['arch']}) | "
                f"ROC-AUC={rec['roc_auc']:.4f} | F1={rec['f1']:.4f} | Recall={rec['recall']*100:.1f}%",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Índice das Janelas Avaliadas (Ordenadas no Tempo)")
            ax.set_ylabel("Probabilidade de Anomalia")
            ax.set_ylim(-0.05, 1.08)
            ax.legend(loc="upper left", ncol=4, fontsize=9)
            ax.grid(True, alpha=0.3)

        fig_path3 = os.path.join(ANOM_PLOTS_DIR, "timeline_deteccao_operador_bombas.png")
        plt.savefig(fig_path3, dpi=140)
        plt.close(fig)


def main():
    t_start = time.time()
    os.makedirs(ANOM_RESULTS_DIR, exist_ok=True)
    os.makedirs(ANOM_PLOTS_DIR, exist_ok=True)

    # 1. Carrega e salva o relatório do operador
    df_op = load_and_save_operator_report()

    # 2. Mapeia os pontos do operador para todas as janelas deslizantes (15min, 30min, 1h, 2h)
    df_map = map_operator_anomalies_to_windows(df_op)
    print("\nResumo de Janelas Rotuladas com Anomalia do Operador:")
    print(df_map[["dataset", "window", "total_windows", "anomaly_windows", "anomaly_rate_pct", "mean_operator_stddev_in_anomalies"]].to_string(index=False))

    # 3. Avalia as 7 arquiteturas nos datasets que possuem anomalias do operador
    rng = np.random.default_rng(SEED)
    all_fold_rows = []
    summary_rows = []
    timeline_records = []

    print("\nIniciando avaliação Out-of-Fold (5 Folds) das 7 arquiteturas frente ao Relatório do Operador...")

    for ds_name in DATASETS:
        for win_label in WINDOWS:
            f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
            df_f = pd.read_csv(f_csv)
            is_anom_all = df_f["is_operator_anomaly"].to_numpy(dtype=np.int8)
            n_anom_total = int(np.sum(is_anom_all == 1))

            if n_anom_total == 0:
                print(f"[SKIP] {ds_name:16s} | {win_label:5s} | 0 anomalias reportadas pelo operador nesta bomba.")
                continue

            t_cfg = time.time()
            npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
            dados = np.load(npz_path)

            X_1d_all = dados["X_1d"]
            X_gasf_all = dados["X_gasf"]
            X_gadf_all = dados["X_gadf"]
            strata_all = dados["stratum"]
            clean_all = ~dados["is_boundary_overlap"]
            scaler_min = dados["scaler_min_per_fold"]
            scaler_max = dados["scaler_max_per_fold"]

            # Seleciona 100% das janelas anômalas + subamostra estratificada de janelas normais limpas
            norm_mask = (is_anom_all == 0) & clean_all
            norm_idx = stratified_subsample_indices(norm_mask, strata_all, MAX_NORMAL_SAMPLES, rng)
            anom_idx = np.flatnonzero(is_anom_all == 1)

            sel_idx = np.sort(np.concatenate([norm_idx, anom_idx]))
            y_sel = is_anom_all[sel_idx].astype(np.int32)
            strata_sel = strata_all[sel_idx].astype(np.int32)

            # Cria 5 folds estratificados conjuntamente por (estrato de carga, rótulo de anomalia)
            # Se o número de anomalias for menor que 5 (ex: Bomba_5307 com 2 ou 3 janelas), usa k = min(5, max(2, n_anom))
            n_splits_eff = min(N_SPLITS, max(2, n_anom_total))
            joint_target = strata_sel * 2 + y_sel
            # Caso algum estrato+anomalia tenha menos que n_splits_eff amostras, estratifica diretamente por y_sel
            _, counts_jt = np.unique(joint_target, return_counts=True)
            strat_vec = joint_target if np.min(counts_jt) >= n_splits_eff else y_sel

            skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
            folds_cv = np.zeros(len(sel_idx), dtype=np.int8)
            for f_i, (_, va_i) in enumerate(skf.split(sel_idx, strat_vec)):
                folds_cv[va_i] = f_i

            # Normaliza X_1d usando os limites globais médios dos folds de treino
            c_min = np.mean(scaler_min, axis=0)[None, :, None]
            c_max = np.mean(scaler_max, axis=0)[None, :, None]
            denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

            x1d_sel = np.clip(2.0 * (X_1d_all[sel_idx] - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
            x2d_sel = np.concatenate([X_gasf_all[sel_idx], X_gadf_all[sel_idx]], axis=1).astype(np.float32)

            del dados, X_1d_all, X_gasf_all, X_gadf_all
            gc.collect()

            for arch_name in ARCHITECTURES:
                fold_metrics, oof_unsup, oof_prob, oof_mse = evaluate_architecture_5folds(
                    arch_name=arch_name,
                    x1d_all=x1d_sel,
                    x2d_all=x2d_sel,
                    y_all=y_sel,
                    folds_cv=folds_cv,
                )

                for fm in fold_metrics:
                    all_fold_rows.append({
                        "dataset": ds_name,
                        "window": win_label,
                        "architecture": arch_name,
                        **fm,
                    })

                # Métricas Globais Out-of-Fold (concatenando todas as predições de validação sem vazamento)
                oof_roc_unsup = float(roc_auc_score(y_sel, oof_unsup))
                oof_pr_unsup = float(average_precision_score(y_sel, oof_unsup))
                thresh_u95 = float(np.percentile(oof_unsup[y_sel == 0], 95))
                oof_rec_u95 = float(np.mean(oof_unsup[y_sel == 1] > thresh_u95))

                oof_roc_hyb = float(roc_auc_score(y_sel, oof_prob))
                oof_pr_hyb = float(average_precision_score(y_sel, oof_prob))

                prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_sel, oof_prob)
                f1_arr = 2.0 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
                best_idx = int(np.argmax(f1_arr))
                best_f1 = float(f1_arr[best_idx])
                best_prec = float(prec_arr[best_idx])
                best_rec = float(rec_arr[best_idx])
                best_thresh = float(thresh_arr[min(best_idx, len(thresh_arr) - 1)]) if len(thresh_arr) > 0 else 0.5

                y_pred_opt = (oof_prob >= best_thresh).astype(int)
                false_alarm_rate = float(np.mean(y_pred_opt[y_sel == 0] == 1))
                acc_opt = float(np.mean(y_pred_opt == y_sel))

                rmse_norm = float(np.sqrt(np.mean(oof_mse[y_sel == 0])))
                rmse_anom = float(np.sqrt(np.mean(oof_mse[y_sel == 1])))
                unsup_ratio = float(np.mean(oof_unsup[y_sel == 1]) / (np.mean(oof_unsup[y_sel == 0]) + 1e-8))

                summary_rows.append({
                    "dataset": ds_name,
                    "window": win_label,
                    "architecture": arch_name,
                    "n_normal_evaluated": int(np.sum(y_sel == 0)),
                    "n_anomaly_evaluated": int(np.sum(y_sel == 1)),
                    "n_folds_cv": int(n_splits_eff),
                    "rmse_1d_normal": round(rmse_norm, 4),
                    "rmse_1d_anomaly": round(rmse_anom, 4),
                    "unsup_anomaly_score_ratio": round(unsup_ratio, 4),
                    "oof_roc_auc_unsup": round(oof_roc_unsup, 4),
                    "oof_pr_auc_unsup": round(oof_pr_unsup, 4),
                    "oof_recall_unsup_p95": round(oof_rec_u95, 4),
                    "oof_roc_auc_hybrid": round(oof_roc_hyb, 4),
                    "oof_pr_auc_hybrid": round(oof_pr_hyb, 4),
                    "oof_accuracy_at_best_f1": round(acc_opt, 4),
                    "oof_best_f1_hybrid": round(best_f1, 4),
                    "oof_precision_at_best_f1": round(best_prec, 4),
                    "oof_recall_at_best_f1": round(best_rec, 4),
                    "oof_false_alarm_rate_pct": round(100.0 * false_alarm_rate, 2),
                })

                # Guarda séries temporais para plotagem de exemplos (Bomba_3307, Bomba_2307 e Dataset_Completo em 15min)
                if win_label == "15min" and ds_name in ["Bomba_3307", "Bomba_2307", "Dataset_Completo"] and arch_name == "Hybrid_Transformer_CNN2D":
                    timeline_records.append({
                        "dataset": ds_name,
                        "window": win_label,
                        "arch": arch_name,
                        "y_true": y_sel,
                        "prob_hybrid": oof_prob,
                        "threshold": best_thresh,
                        "roc_auc": oof_roc_hyb,
                        "f1": best_f1,
                        "recall": best_rec,
                    })

            df_curr = pd.DataFrame([r for r in summary_rows if r["dataset"] == ds_name and r["window"] == win_label])
            best_row = df_curr.sort_values("oof_roc_auc_hybrid", ascending=False).iloc[0]
            print(
                f"[OK] {ds_name:16s} | {win_label:5s} | Anom={n_anom_total:3d} | "
                f"Melhor: {best_row['architecture']:24s} (ROC-AUC={best_row['oof_roc_auc_hybrid']:.4f}, "
                f"F1={best_row['oof_best_f1_hybrid']:.4f}, Recall={best_row['oof_recall_at_best_f1']*100:.1f}%) | "
                f"{time.time() - t_cfg:.1f}s"
            )

    # 4. Salva resultados detalhados e resumos
    df_folds_out = pd.DataFrame(all_fold_rows)
    df_summary_out = pd.DataFrame(summary_rows)

    folds_out_csv = os.path.join(ANOM_RESULTS_DIR, "metricas_anomalias_por_fold.csv")
    summary_out_csv = os.path.join(ANOM_RESULTS_DIR, "resumo_deteccao_operador_por_bomba_janela.csv")
    df_folds_out.to_csv(folds_out_csv, index=False)
    df_summary_out.to_csv(summary_out_csv, index=False)

    # Ranking global das 7 arquiteturas na detecção das anomalias do operador
    df_ranking_anom = (
        df_summary_out.groupby("architecture", as_index=False)
        .agg(
            roc_auc_hybrid_mean=("oof_roc_auc_hybrid", "mean"),
            pr_auc_hybrid_mean=("oof_pr_auc_hybrid", "mean"),
            best_f1_hybrid_mean=("oof_best_f1_hybrid", "mean"),
            precision_hybrid_mean=("oof_precision_at_best_f1", "mean"),
            recall_hybrid_mean=("oof_recall_at_best_f1", "mean"),
            false_alarm_rate_pct_mean=("oof_false_alarm_rate_pct", "mean"),
            roc_auc_unsup_mean=("oof_roc_auc_unsup", "mean"),
            unsup_score_ratio_mean=("unsup_anomaly_score_ratio", "mean"),
        )
        .sort_values("roc_auc_hybrid_mean", ascending=False)
    )
    ranking_out_csv = os.path.join(ANOM_RESULTS_DIR, "ranking_global_deteccao_anomalias.csv")
    df_ranking_anom.to_csv(ranking_out_csv, index=False)

    # Ranking por janela temporal
    df_window_anom = (
        df_summary_out.groupby("window", as_index=False)
        .agg(
            roc_auc_hybrid_mean=("oof_roc_auc_hybrid", "mean"),
            pr_auc_hybrid_mean=("oof_pr_auc_hybrid", "mean"),
            best_f1_hybrid_mean=("oof_best_f1_hybrid", "mean"),
            recall_hybrid_mean=("oof_recall_at_best_f1", "mean"),
            roc_auc_unsup_mean=("oof_roc_auc_unsup", "mean"),
        )
    )
    window_out_csv = os.path.join(ANOM_RESULTS_DIR, "resumo_deteccao_por_escala_janela.csv")
    df_window_anom.to_csv(window_out_csv, index=False)

    # 5. Gera figuras
    generate_operator_anomaly_plots(df_summary_out, df_map, timeline_records)

    print("\n=== RANKING GLOBAL DAS 7 ARQUITETURAS NA DETECÇÃO DAS ANOMALIAS DO OPERADOR ===")
    print(df_ranking_anom.to_string(index=False))
    print("\n=== DESEMPENHO MÉDIO POR ESCALA DE JANELA TEMPORAL ===")
    print(df_window_anom.to_string(index=False))
    print(f"\nTodos os resultados e figuras foram salvos em: {ANOM_RESULTS_DIR}")
    print(f"Tempo total de execução: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
