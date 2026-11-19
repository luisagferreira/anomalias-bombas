"""
Avaliação Comparativa Completa em GPU (CUDA) das Diferentes Formulações de Rótulo (Labeling)
a partir da Marcação Anual do Operador (StdDev 5min >= 2.5 A):

1. Abordagem Atual (Detecção Instantânea Binária 0/1 na janela [t0, t1]):
   - `1_Atual_Binario`

2. Previsão Geral de Horizonte Futuro (Sensores em [t0, t1] -> Anomalia em (t1, t1 + H]):
   - `2_Futuro_Geral_15min` (H = 15 min)
   - `3_Futuro_Geral_30min` (H = 30 min)
   - `4_Futuro_Geral_60min` (H = 60 min)

3. Alerta Precoce Puro / Precursor (Somente janelas ATUALMENTE NORMAIS em [t0, t1] que
   entrarão em anomalia nos próximos H minutos (t1, t1 + H]):
   - `5_Alerta_Precoce_15min` (H = 15 min | Condicionado a y_atual == 0)
   - `6_Alerta_Precoce_30min` (H = 30 min | Condicionado a y_atual == 0)
   - `7_Alerta_Precoce_60min` (H = 60 min | Condicionado a y_atual == 0)

4. Distribuição Contínua de Probabilidade (% de Anomalia na Janela e na Vizinhança Antes/Depois):
   - `8_Pct_IntraJanela_Pura`      : Fração exata de blocos de 5 min anômalos dentro da janela (tau = 0)
   - `9_Dist_Prob_Curta_15m_7m`    : Kernel Gaussiano Assimétrico (tau_antes = 15 min, tau_depois = 7.5 min)
   - `10_Dist_Prob_Media_30m_15m`  : Kernel Gaussiano Assimétrico (tau_antes = 30 min, tau_depois = 15 min)
   - `11_Dist_Prob_Longa_60m_30m`  : Kernel Gaussiano Assimétrico (tau_antes = 60 min, tau_depois = 30 min)

Executa as 7 arquiteturas (1D, 2D GADF e Híbridas) nos 5 Folds estratificados pelo rótulo
com exclusão dinâmica de fronteira (`clean_mask`, zero data leakage).
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treinar_modelos_1d_2d_gaf import (
    AutoencoderModel,
    ARCHITECTURES,
    INDIVIDUAL_PUMPS,
    DATASETS,
    LATENT_DIM,
    SEED,
)
from avaliar_deteccao_anomalias_operador import (
    extract_latent_and_reconstructions,
    compute_auxiliary_signal_features,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

BASE_DIR = "Dataset_completo_bombas"
FOLDS_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")
RESULTS_DIR = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D")
ANOM_RESULTS_DIR = os.path.join(RESULTS_DIR, "Deteccao_Anomalias_Operador")
ANOM_PLOTS_DIR = os.path.join(ANOM_RESULTS_DIR, "Figuras_Anomalias_Operador")
ANNUAL_2_5_CSV = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_acima_2_5.csv")

N_SPLITS = 5
LR = 3e-3

APPROACH_SPECS = [
    ("1_Atual_Binario", "1. Atual Binário (0/1)", "binary_all"),
    ("2_Futuro_Geral_15min", "2. Previsão Geral Futuro (H=15min)", "binary_all"),
    ("3_Futuro_Geral_30min", "3. Previsão Geral Futuro (H=30min)", "binary_all"),
    ("4_Futuro_Geral_60min", "4. Previsão Geral Futuro (H=60min)", "binary_all"),
    ("5_Alerta_Precoce_15min", "5. Alerta Precoce Puro (H=15min | Atual=0)", "early_only"),
    ("6_Alerta_Precoce_30min", "6. Alerta Precoce Puro (H=30min | Atual=0)", "early_only"),
    ("7_Alerta_Precoce_60min", "7. Alerta Precoce Puro (H=60min | Atual=0)", "early_only"),
    ("8_Pct_IntraJanela_Pura", "8. % Duração Intra-Janela (τ=0)", "soft_prob"),
    ("9_Dist_Prob_Curta_15m_7m", "9. Dist. Prob. Curta (τ_ant=15m, τ_dep=7.5m)", "soft_prob"),
    ("10_Dist_Prob_Media_30m_15m", "10. Dist. Prob. Média (τ_ant=30m, τ_dep=15m)", "soft_prob"),
    ("11_Dist_Prob_Longa_60m_30m", "11. Dist. Prob. Longa (τ_ant=60m, τ_dep=30m)", "soft_prob"),
]


def compute_all_label_targets(
    t0_sec: np.ndarray,
    t1_sec: np.ndarray,
    t_anom_sec: np.ndarray,
    win_size_steps: int,
) -> dict[str, np.ndarray]:
    """
    Calcula de forma 100% vetorizada todos os 11 vetores de rótulo para as janelas [t0_sec, t1_sec].
    """
    n_win = len(t0_sec)
    k_blocks_max = max(1.0, (win_size_steps * 10.0) / 300.0)  # qtd de blocos de 5min (300s) na janela

    # 1. Presença e contagem dentro da própria janela [t0, t1]
    idx_l = np.searchsorted(t_anom_sec, t0_sec, side="left")
    idx_r = np.searchsorted(t_anom_sec, t1_sec, side="right")
    n_inside = (idx_r - idx_l).astype(np.float32)
    y_atual = (n_inside > 0).astype(np.float32)

    # 2. Presença no horizonte futuro (t1, t1 + H]
    horizons = {"15min": 15 * 60, "30min": 30 * 60, "60min": 60 * 60}
    y_fut = {}
    for h_name, h_sec in horizons.items():
        idx_fl = np.searchsorted(t_anom_sec, t1_sec + 1, side="left")
        idx_fr = np.searchsorted(t_anom_sec, t1_sec + h_sec, side="right")
        y_fut[h_name] = ((idx_fr - idx_fl) > 0).astype(np.float32)

    # 3. Distância em minutos até a próxima anomalia (antes da falha) e desde a última anomalia (depois da falha)
    idx_next = np.searchsorted(t_anom_sec, t1_sec, side="right")
    dt_before_min = np.full(n_win, 1e6, dtype=np.float32)
    valid_next = idx_next < len(t_anom_sec)
    dt_before_min[valid_next] = np.maximum(0.0, (t_anom_sec[idx_next[valid_next]] - t1_sec[valid_next]) / 60.0)

    idx_prev = np.searchsorted(t_anom_sec, t0_sec, side="left") - 1
    dt_after_min = np.full(n_win, 1e6, dtype=np.float32)
    valid_prev = idx_prev >= 0
    # Considera o término do bloco de 5 min (300s)
    dt_after_min[valid_prev] = np.maximum(0.0, (t0_sec[valid_prev] - (t_anom_sec[idx_prev[valid_prev]] + 300.0)) / 60.0)

    # 4. Porcentagem Intra-Janela pura
    y_pct_intra = np.clip(n_inside / k_blocks_max, 0.0, 1.0).astype(np.float32)

    # 5. Kernels Gaussianos Assimétricos antes/depois
    def build_soft_kernel(tau_before: float, tau_after: float) -> np.ndarray:
        p_bef = np.exp(-0.5 * (dt_before_min / tau_before) ** 2)
        p_aft = np.exp(-0.5 * (dt_after_min / tau_after) ** 2)
        p_neigh = np.maximum(p_bef, p_aft)
        # Se a janela já contém anomalia do operador, combina 0.70 + 0.30 * fração intra-janela (chegando a 1.0)
        p_in = np.clip(0.75 + 0.25 * y_pct_intra, 0.0, 1.0)
        return np.where(y_atual > 0.5, p_in, p_neigh).astype(np.float32)

    targets = {
        "1_Atual_Binario": y_atual,
        "2_Futuro_Geral_15min": y_fut["15min"],
        "3_Futuro_Geral_30min": y_fut["30min"],
        "4_Futuro_Geral_60min": y_fut["60min"],
        "5_Alerta_Precoce_15min": y_fut["15min"],
        "6_Alerta_Precoce_30min": y_fut["30min"],
        "7_Alerta_Precoce_60min": y_fut["60min"],
        "8_Pct_IntraJanela_Pura": y_pct_intra,
        "9_Dist_Prob_Curta_15m_7m": build_soft_kernel(15.0, 7.5),
        "10_Dist_Prob_Media_30m_15m": build_soft_kernel(30.0, 15.0),
        "11_Dist_Prob_Longa_60m_30m": build_soft_kernel(60.0, 30.0),
    }
    return targets


def compute_clean_group_folds(is_anom_all: np.ndarray, block_id_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_win = len(is_anom_all)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds_all = np.zeros(n_win, dtype=np.int8)
    for f_i, (_, va_i) in enumerate(sgkf.split(np.arange(n_win), is_anom_all, groups=block_id_all)):
        folds_all[va_i] = f_i

    diff_prev = np.zeros(n_win, dtype=bool)
    diff_next = np.zeros(n_win, dtype=bool)
    diff_prev[1:] = folds_all[1:] != folds_all[:-1]
    diff_next[:-1] = folds_all[:-1] != folds_all[1:]
    clean_mask = ~(diff_prev | diff_next)
    return folds_all, clean_mask


def evaluate_all_approaches_5folds_gpu(
    arch_name: str,
    x1d_all: np.ndarray,
    x2d_all: np.ndarray,
    targets_dict: dict[str, np.ndarray],
    folds_cv: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """
    Treina em GPU CUDA o Autoencoder (nas janelas estritamente normais) e as 11 cabeças
    de predição correspondentes a cada formulação de rótulo nos 5 Folds (Out-of-Fold).
    """
    n_samples, n_channels, seq_len = x1d_all.shape
    c_2d = x2d_all.shape[1]
    feat_dim = LATENT_DIM + 5 * n_channels + c_2d

    oof_preds = {k: np.zeros(n_samples, dtype=np.float32) for k, _, _ in APPROACH_SPECS}
    y_atual_all = targets_dict["1_Atual_Binario"]
    y_soft30_all = targets_dict["10_Dist_Prob_Media_30m_15m"]

    # Empilha todos os 11 alvos em uma matriz (N, 11) para transferência única à VRAM
    target_keys = [k for k, _, _ in APPROACH_SPECS]
    Y_matrix = np.column_stack([targets_dict[k] for k in target_keys]).astype(np.float32)

    use_amp = (device.type == "cuda")
    unique_folds = np.unique(folds_cv)

    for val_fold in unique_folds:
        tr_idx = np.flatnonzero(folds_cv != val_fold)
        va_idx = np.flatnonzero(folds_cv == val_fold)

        t_x1d_tr = torch.from_numpy(x1d_all[tr_idx]).to(device, non_blocking=True)
        t_x2d_tr = torch.from_numpy(x2d_all[tr_idx]).to(device, non_blocking=True)
        t_Y_tr = torch.from_numpy(Y_matrix[tr_idx]).to(device, non_blocking=True)

        t_x1d_va = torch.from_numpy(x1d_all[va_idx]).to(device, non_blocking=True)
        t_x2d_va = torch.from_numpy(x2d_all[va_idx]).to(device, non_blocking=True)

        torch.manual_seed(SEED + int(val_fold))
        model = AutoencoderModel(arch_name, n_channels, seq_len).to(device)
        shared_bn = nn.BatchNorm1d(feat_dim).to(device)
        heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in target_keys
        ]).to(device)

        # Calcula pos_weight específico de cada abordagem no fold de treino
        pos_weights = []
        for idx_k, (k_name, _, k_mode) in enumerate(APPROACH_SPECS):
            y_col = Y_matrix[tr_idx, idx_k]
            if k_mode == "early_only":
                m_sub = (y_atual_all[tr_idx] < 0.5)
                y_sub = y_col[m_sub]
            else:
                y_sub = y_col
            n_pos = float(np.sum(y_sub >= 0.5))
            n_neg = float(np.sum(y_sub < 0.5))
            pw = min(35.0, max(1.0, n_neg / max(1.0, n_pos)))
            pos_weights.append(torch.tensor([pw], dtype=torch.float32, device=device))

        params = list(model.parameters()) + list(shared_bn.parameters()) + list(heads.parameters())
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_tr = len(tr_idx)
        model.train()
        shared_bn.train()
        heads.train()

        for _ in range(epochs):
            perm = torch.randperm(n_tr, device=device)
            for start_b in range(0, n_tr, batch_size):
                b_idx = perm[start_b : start_b + batch_size]
                if len(b_idx) <= 1:
                    continue

                b_1d = t_x1d_tr[b_idx].float()
                b_2d = t_x2d_tr[b_idx].float()
                b_Y = t_Y_tr[b_idx]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b_1d, b_2d)

                    # Reconstrução apenas nas janelas estritamente normais (sem anomalia atual nem vizinhança próxima)
                    strict_norm = (b_Y[:, 0] < 0.5) & (b_Y[:, 9] < 0.15)
                    if not torch.any(strict_norm):
                        strict_norm = (b_Y[:, 0] < 0.5)

                    if torch.any(strict_norm):
                        loss_rec_1d = ((rec_1d[strict_norm] - b_1d[strict_norm]) ** 2).mean()
                        if rec_2d is not None:
                            loss_rec_2d = ((rec_2d[strict_norm] - b_2d[strict_norm]) ** 2).mean()
                            loss_rec = loss_rec_1d + 0.5 * loss_rec_2d
                        else:
                            loss_rec = loss_rec_1d
                    else:
                        loss_rec = ((rec_1d - b_1d) ** 2).mean() * 0.0

                    aux_f = compute_auxiliary_signal_features(b_1d, rec_1d.detach(), rec_2d.detach() if rec_2d is not None else None, b_2d)
                    full_f = shared_bn(torch.cat([z, aux_f], dim=-1))

                    # Cabeça 0 (Atual) + Cabeça 9 (Soft 30m) guiam o codificador; demais cabeças usam full_f.detach() para não conflitar gradientes
                    loss_heads = 0.0
                    norm_now_mask = (b_Y[:, 0] < 0.5)
                    for idx_k, (_, _, k_mode) in enumerate(APPROACH_SPECS):
                        f_in = full_f if idx_k in (0, 2, 9) else full_f.detach()
                        logits_k = heads[idx_k](f_in).squeeze(1)
                        target_k = b_Y[:, idx_k]

                        if k_mode == "early_only":
                            if torch.any(norm_now_mask):
                                bce_k = nn.functional.binary_cross_entropy_with_logits(
                                    logits_k[norm_now_mask],
                                    target_k[norm_now_mask],
                                    pos_weight=pos_weights[idx_k],
                                )
                                loss_heads = loss_heads + bce_k
                        elif k_mode == "soft_prob":
                            # Combina BCE contínua ponderada + MSE na probabilidade sigmóide
                            probs_k = torch.sigmoid(logits_k)
                            w_soft = 1.0 + (pos_weights[idx_k] - 1.0) * target_k
                            bce_soft = nn.functional.binary_cross_entropy_with_logits(
                                logits_k, target_k, reduction="none"
                            )
                            loss_k = (bce_soft * w_soft).mean() + 2.0 * ((probs_k - target_k) ** 2).mean()
                            loss_heads = loss_heads + loss_k
                        else:
                            bce_k = nn.functional.binary_cross_entropy_with_logits(
                                logits_k, target_k, pos_weight=pos_weights[idx_k]
                            )
                            loss_heads = loss_heads + bce_k

                    loss = loss_rec + 0.35 * loss_heads

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Inferência no Fold de Validação
        model.eval()
        shared_bn.eval()
        heads.eval()
        n_va = len(va_idx)
        with torch.no_grad():
            for start_b in range(0, n_va, batch_size):
                end_b = min(n_va, start_b + batch_size)
                b_1d = t_x1d_va[start_b:end_b].float()
                b_2d = t_x2d_va[start_b:end_b].float()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b_1d, b_2d)
                    aux_f = compute_auxiliary_signal_features(b_1d, rec_1d, rec_2d, b_2d)
                    full_f = shared_bn(torch.cat([z, aux_f], dim=-1))
                    for idx_k, k_name in enumerate(target_keys):
                        probs_va = torch.sigmoid(heads[idx_k](full_f).squeeze(1)).cpu().numpy()
                        oof_preds[k_name][va_idx[start_b:end_b]] = probs_va

        del t_x1d_tr, t_x2d_tr, t_Y_tr, t_x1d_va, t_x2d_va, model, shared_bn, heads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return oof_preds


def compute_metrics_for_approach(
    k_name: str,
    k_label: str,
    k_mode: str,
    oof_prob: np.ndarray,
    targets_dict: dict[str, np.ndarray],
) -> dict:
    y_atual = (targets_dict["1_Atual_Binario"] >= 0.5).astype(np.int32)
    y_early30 = (targets_dict["6_Alerta_Precoce_30min"] >= 0.5).astype(np.int32)
    target_raw = targets_dict[k_name]

    if k_mode == "early_only":
        # Avalia estritamente nas janelas onde a bomba ainda está NORMAL no presente (y_atual == 0)
        eval_mask = (y_atual == 0)
        y_bin = (target_raw[eval_mask] >= 0.5).astype(np.int32)
        p_eval = oof_prob[eval_mask]
    elif k_mode == "soft_prob":
        # Para distribuição contínua, avalia a discriminação tanto de anomalias reais quanto de zona de risco (>= 50%)
        eval_mask = np.ones(len(oof_prob), dtype=bool)
        y_bin = (target_raw >= 0.50).astype(np.int32)
        p_eval = oof_prob
    else:
        eval_mask = np.ones(len(oof_prob), dtype=bool)
        y_bin = (target_raw >= 0.5).astype(np.int32)
        p_eval = oof_prob

    roc_auc = float(roc_auc_score(y_bin, p_eval)) if len(np.unique(y_bin)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_bin, p_eval)) if len(np.unique(y_bin)) > 1 else 0.0

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_bin, p_eval)
    f1_arr = 2.0 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_idx = int(np.argmax(f1_arr))
    best_f1 = float(f1_arr[best_idx])
    best_prec = float(prec_arr[best_idx])
    best_rec = float(rec_arr[best_idx])
    best_th = float(thresh_arr[min(best_idx, len(thresh_arr) - 1)]) if len(thresh_arr) > 0 else 0.5

    y_pred = (p_eval >= best_th).astype(np.int32)
    vn = int(np.sum((y_bin == 0) & (y_pred == 0)))
    fp = int(np.sum((y_bin == 0) & (y_pred == 1)))
    fn = int(np.sum((y_bin == 1) & (y_pred == 0)))
    vp = int(np.sum((y_bin == 1) & (y_pred == 1)))
    acc = float((vn + vp) / max(1, len(y_bin)))
    far_pct = float(100.0 * fp / max(1, vn + fp))

    # Métricas contínuas de aderência à porcentagem (MAE em % e Correlação de Spearman)
    mae_pct = float(100.0 * np.mean(np.abs(p_eval - target_raw[eval_mask])))
    spearman_r = float(spearmanr(p_eval, target_raw[eval_mask]).statistic)
    if np.isnan(spearman_r):
        spearman_r = 0.0

    # Taxa de captura precursora (quantos % dos eventos que iniciarão em até 30min já são alertados enquanto y_atual == 0)
    precursor_mask = (y_atual == 0) & (y_early30 == 1)
    if np.any(precursor_mask):
        # Usa limiar calibrado para ~5% de falso alarme nas janelas estritamente normais
        strict_norm_mask = (y_atual == 0) & (y_early30 == 0)
        th_5pct_far = float(np.percentile(oof_prob[strict_norm_mask], 95.0)) if np.any(strict_norm_mask) else best_th
        precursor_recall_at_5far = float(100.0 * np.mean(oof_prob[precursor_mask] >= th_5pct_far))
        roc_on_actual = float(roc_auc_score(y_atual, oof_prob)) if len(np.unique(y_atual)) > 1 else 0.5
    else:
        precursor_recall_at_5far = 0.0
        roc_on_actual = roc_auc

    return {
        "approach_id": k_name,
        "approach_label": k_label,
        "mode": k_mode,
        "n_evaluated": int(len(y_bin)),
        "n_normal": int(np.sum(y_bin == 0)),
        "n_positive": int(np.sum(y_bin == 1)),
        "VN": vn,
        "FP": fp,
        "FN": fn,
        "VP": vp,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "best_f1": round(best_f1, 4),
        "precision": round(best_prec, 4),
        "recall": round(best_rec, 4),
        "false_alarm_rate_pct": round(far_pct, 2),
        "prob_mae_pct": round(mae_pct, 2),
        "spearman_corr": round(spearman_r, 4),
        "roc_auc_on_operator_actual": round(roc_on_actual, 4),
        "early_warning_30m_recall_at_5pct_far": round(precursor_recall_at_5far, 2),
    }


def generate_comparative_plots(df_all: pd.DataFrame):
    os.makedirs(ANOM_PLOTS_DIR, exist_ok=True)

    # Figura 1: Comparação das 11 Abordagens (ROC-AUC, F1-Score e Captura Precursora)
    df_grp = (
        df_all.groupby(["approach_id", "approach_label"], as_index=False, sort=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            best_f1=("best_f1", "mean"),
            accuracy=("accuracy", "mean"),
            early_rec=("early_warning_30m_recall_at_5pct_far", "mean"),
            spearman=("spearman_corr", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=220, constrained_layout=True)

    y_pos = np.arange(len(df_grp))
    labels = df_grp["approach_label"].tolist()

    ax0 = axes[0]
    w = 0.38
    ax0.barh(y_pos - w / 2, df_grp["roc_auc"], height=w, label="ROC-AUC Médio", color="#2563EB")
    ax0.barh(y_pos + w / 2, df_grp["best_f1"], height=w, label="F1-Score Médio", color="#16A34A")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(labels, fontsize=8.8, fontweight="bold")
    ax0.invert_yaxis()
    ax0.set_xlim(0.45, 1.03)
    ax0.set_xlabel("Pontuação (0 a 1)")
    ax0.set_title("Desempenho Global por Abordagem de Rótulo\n(Média das 7 Arquiteturas e 7 Datasets - 5 Folds)", fontsize=11, fontweight="bold")
    ax0.grid(True, axis="x", alpha=0.3)
    ax0.legend(loc="lower right")
    for i, (r_val, f_val) in enumerate(zip(df_grp["roc_auc"], df_grp["best_f1"])):
        ax0.text(r_val + 0.006, i - w / 2, f"{r_val:.3f}", va="center", fontsize=7.8, fontweight="bold", color="#1E3A8A")
        ax0.text(f_val + 0.006, i + w / 2, f"{f_val:.3f}", va="center", fontsize=7.8, fontweight="bold", color="#14532D")

    ax1 = axes[1]
    ax1.barh(y_pos - w / 2, df_grp["early_rec"], height=w, label="Captura Precursora 30min antes (% a 5% FAR)", color="#D97706")
    ax1.barh(y_pos + w / 2, df_grp["spearman"] * 100.0, height=w, label="Correlação de Spearman com o Risco (%)", color="#9333EA")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(["" for _ in labels])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Porcentagem (%)")
    ax1.set_title("Antecedência (Captura 30min Antes da Falha)\ne Correlação com o Gradiente de Risco", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.legend(loc="lower right")
    for i, (e_val, s_val) in enumerate(zip(df_grp["early_rec"], df_grp["spearman"] * 100.0)):
        ax1.text(e_val + 0.8, i - w / 2, f"{e_val:.1f}%", va="center", fontsize=7.8, fontweight="bold", color="#92400E")
        ax1.text(s_val + 0.8, i + w / 2, f"{s_val:.1f}%", va="center", fontsize=7.8, fontweight="bold", color="#581C87")

    out_fig = os.path.join(ANOM_PLOTS_DIR, "comparacao_11_abordagens_rotulos.png")
    plt.savefig(out_fig, dpi=220)
    plt.close(fig)
    print(f"[OK] Figura comparativa salva em: {out_fig}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU das 11 Formulações de Rótulo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window", type=str, default="15min", help="Janela base (15min tem a maior resolução de 57.215 janelas)")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    win_label = args.window
    win_steps_map = {"15min": 90, "30min": 180, "1h": 360, "2h": 720}
    win_steps = win_steps_map[win_label]

    os.makedirs(ANOM_RESULTS_DIR, exist_ok=True)
    os.makedirs(ANOM_PLOTS_DIR, exist_ok=True)

    t0_global = time.time()
    print("=" * 85, flush=True)
    print(f"BENCHMARK COMPARATIVO DAS 11 ABORDAGENS DE RÓTULO EM GPU ({device})", flush=True)
    print(f"Escala: {win_label} (L={win_steps}) | Épocas: {args.epochs} | Batch: {args.batch_size} | 5 Folds Limpos", flush=True)
    print("=" * 85, flush=True)

    df_annual = pd.read_csv(ANNUAL_2_5_CSV)
    df_annual["Timestamp"] = pd.to_datetime(df_annual["Timestamp Estimado"])
    df_annual["Nome_Bomba"] = df_annual["Nome_Bomba"].astype(str).str.strip()
    df_annual["StdDev_Val"] = pd.to_numeric(df_annual["Valor Estimado (StdDev)"], errors="coerce")

    all_rows = []

    for ds_name in DATASETS:
        t_ds = time.time()
        pump_id = ds_name.replace("Bomba_", "")
        sub_op = df_annual if ds_name == "Dataset_Completo" else df_annual[df_annual["Nome_Bomba"] == pump_id]
        op_grouped = (
            sub_op.groupby("Timestamp", as_index=False)
            .agg(n_records=("StdDev_Val", "count"), max_std=("StdDev_Val", "max"))
            .sort_values("Timestamp")
        )
        t_anom_sec = op_grouped["Timestamp"].values.astype("datetime64[s]").astype(np.int64)

        f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
        df_f = pd.read_csv(f_csv)
        t0_sec = pd.to_datetime(df_f["start_timestamp"]).values.astype("datetime64[s]").astype(np.int64)
        t1_sec = pd.to_datetime(df_f["end_timestamp"]).values.astype("datetime64[s]").astype(np.int64)
        block_id_all = df_f["block_id"].to_numpy(dtype=np.int32)

        targets_full = compute_all_label_targets(t0_sec, t1_sec, t_anom_sec, win_steps)
        is_anom_bin = (targets_full["1_Atual_Binario"] >= 0.5).astype(np.int8)

        folds_all, clean_mask = compute_clean_group_folds(is_anom_bin, block_id_all)
        sel_idx = np.flatnonzero(clean_mask)
        folds_sel = folds_all[sel_idx]
        targets_sel = {k: v[sel_idx] for k, v in targets_full.items()}

        # Carrega tensores 1D e 2D (Somente GADF, concatenando as 6 bombas para o Dataset_Completo)
        if ds_name != "Dataset_Completo":
            npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
            dados = np.load(npz_path)
            c_min = np.mean(dados["scaler_min_per_fold"], axis=0)[None, :, None]
            c_max = np.mean(dados["scaler_max_per_fold"], axis=0)[None, :, None]
            denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
            x1d_raw = dados["X_1d"][sel_idx].astype(np.float32)
            x1d_sel = np.clip(2.0 * (x1d_raw - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
            del x1d_raw
            x2d_sel = np.ascontiguousarray(dados["X_gadf"][sel_idx], dtype=np.float16)
            dados.close()
            del dados
            gc.collect()
        else:
            n_sel = len(sel_idx)
            n_pumps = len(INDIVIDUAL_PUMPS)
            x1d_sel = np.empty((n_sel, 4 * n_pumps, win_steps), dtype=np.float32)
            x2d_sel = np.empty((n_sel, 4 * n_pumps, 32, 32), dtype=np.float16)
            for p_idx, p_name in enumerate(INDIVIDUAL_PUMPS):
                p_npz = np.load(os.path.join(FOLDS_DIR, p_name, f"sequencias_{p_name}_{win_label}.npz"))
                c_min = np.mean(p_npz["scaler_min_per_fold"], axis=0)[None, :, None]
                c_max = np.mean(p_npz["scaler_max_per_fold"], axis=0)[None, :, None]
                denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
                x1d_p = p_npz["X_1d"][sel_idx].astype(np.float32)
                x1d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :] = np.clip(2.0 * (x1d_p - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
                del x1d_p
                x2d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :, :] = p_npz["X_gadf"][sel_idx]
                p_npz.close()
                del p_npz
                gc.collect()

        for arch_name in ARCHITECTURES:
            t_arch = time.time()
            oof_preds = evaluate_all_approaches_5folds_gpu(
                arch_name=arch_name,
                x1d_all=x1d_sel,
                x2d_all=x2d_sel,
                targets_dict=targets_sel,
                folds_cv=folds_sel,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

            for k_name, k_label, k_mode in APPROACH_SPECS:
                m_row = compute_metrics_for_approach(
                    k_name=k_name,
                    k_label=k_label,
                    k_mode=k_mode,
                    oof_prob=oof_preds[k_name],
                    targets_dict=targets_sel,
                )
                all_rows.append({
                    "dataset": ds_name,
                    "window": win_label,
                    "architecture": arch_name,
                    **m_row,
                })

            print(
                f"[OK] {ds_name:16s} | {arch_name:24s} | "
                f"Atual AUC={all_rows[-11]['roc_auc']:.4f} | "
                f"Fut30m AUC={all_rows[-9]['roc_auc']:.4f} | "
                f"Precoce30m AUC={all_rows[-6]['roc_auc']:.4f} | "
                f"Prob30m AUC={all_rows[-2]['roc_auc']:.4f} (EarlyRec={all_rows[-2]['early_warning_30m_recall_at_5pct_far']:.1f}%) | "
                f"{time.time() - t_arch:.1f}s",
                flush=True,
            )

        del x1d_sel, x2d_sel
        gc.collect()
        print(f"--- Concluído {ds_name} em {time.time() - t_ds:.1f}s ---", flush=True)

    df_all = pd.DataFrame(all_rows)
    out_full_csv = os.path.join(ANOM_RESULTS_DIR, "comparacao_detalhada_11_abordagens_rotulos.csv")
    df_all.to_csv(out_full_csv, index=False)

    # Resumo global por abordagem de rótulo
    df_summary_app = (
        df_all.groupby(["approach_id", "approach_label", "mode"], as_index=False, sort=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            best_f1_mean=("best_f1", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            false_alarm_rate_pct_mean=("false_alarm_rate_pct", "mean"),
            prob_mae_pct_mean=("prob_mae_pct", "mean"),
            spearman_corr_mean=("spearman_corr", "mean"),
            roc_auc_on_operator_actual_mean=("roc_auc_on_operator_actual", "mean"),
            early_warning_30m_recall_at_5pct_far_mean=("early_warning_30m_recall_at_5pct_far", "mean"),
        )
    )
    out_app_csv = os.path.join(ANOM_RESULTS_DIR, "resumo_global_11_abordagens_rotulos.csv")
    df_summary_app.to_csv(out_app_csv, index=False)

    # Resumo cruzado Abordagem x Arquitetura
    df_app_arch = (
        df_all.groupby(["approach_id", "approach_label", "architecture"], as_index=False, sort=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            accuracy_mean=("accuracy", "mean"),
            best_f1_mean=("best_f1", "mean"),
            recall_mean=("recall", "mean"),
            early_warning_30m_recall_mean=("early_warning_30m_recall_at_5pct_far", "mean"),
            spearman_corr_mean=("spearman_corr", "mean"),
        )
    )
    out_app_arch_csv = os.path.join(ANOM_RESULTS_DIR, "resumo_abordagens_por_arquitetura.csv")
    df_app_arch.to_csv(out_app_arch_csv, index=False)

    generate_comparative_plots(df_all)

    print("\n=== RESUMO GLOBAL COMPARATIVO DAS 11 ABORDAGENS DE RÓTULO ===", flush=True)
    print(df_summary_app.to_string(index=False), flush=True)
    print(f"\nConcluído com sucesso em {time.time() - t0_global:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
