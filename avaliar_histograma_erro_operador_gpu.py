"""
Avaliação em GPU (CUDA) da Detecção de Anomalias via Histograma do Erro de Reconstrução
com Duplo Limiar e Triagem Humana (Human-in-the-Loop) para Dados Intermediários (FP e FN).

Fluxo Metodológico:
1. Treina cada uma das 7 arquiteturas de Autoencoders (1D, 2D GADF e Híbridas) na GPU
   (5 Folds via StratifiedGroupKFold com exclusão dinâmica de fronteira `clean_mask`
   para zero data leakage).
2. Extrai o Score do Erro de Reconstrução Calibrado E(x) de cada janela (combinando
   os desvios residuais 1D, 2D GADF e a projeção latente de erro).
3. Constrói o Histograma do Erro de Reconstrução no conjunto de treino de cada fold
   (120 bins) e determina automaticamente:
   - `tau_hist`  : Limiar único de separação no histograma (ponto de interseção que
                   minimiza FP + FN / maximiza F1 entre as distribuições Normal e Anômala).
   - `tau_baixo` : Limiar inferior da zona intermediária (captura a cauda esquerda das
                   anomalias onde ocorrem os Falsos Negativos - FN).
   - `tau_alto`  : Limiar superior da zona intermediária (captura a cauda direita dos
                   dados normais onde ocorrem os Falsos Positivos - FP).
   - Faixa `[tau_baixo, tau_alto]`: Zona Intermediária de incerteza encaminhada para
     avaliação do Operador (Human-in-the-Loop).
4. Gera tabelas comparativas, planilha de janelas intermediárias para auditoria do
   operador e gráficos de alta resolução dos histogramas com as 3 zonas de decisão.
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treinar_modelos_1d_2d_gaf import (
    AutoencoderModel,
    ARCHITECTURES,
    INDIVIDUAL_PUMPS,
    DATASETS,
    SEED,
    LR,
    LATENT_DIM,
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
RESULTS_DIR = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D", "Deteccao_Anomalias_Operador")
PLOTS_DIR = os.path.join(RESULTS_DIR, "Figuras_Anomalias_Operador")

N_SPLITS = 5


def parse_args():
    has_cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description="Histograma do Erro de Reconstrução + Triagem do Operador na GPU")
    p.add_argument("--device", type=str, default="cuda" if has_cuda else "cpu")
    p.add_argument("--window", type=str, default="15min")
    p.add_argument("--epochs", type=int, default=20 if has_cuda else 8)
    p.add_argument("--batch-size", type=int, default=256 if has_cuda else 128)
    p.add_argument("--max-samples", type=int, default=4000 if has_cuda else 400)
    return p.parse_args()


def compute_clean_group_folds(is_anom_all: np.ndarray, block_id_all: np.ndarray, n_splits: int = N_SPLITS) -> tuple[np.ndarray, np.ndarray]:
    n_win = len(is_anom_all)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds_all = np.zeros(n_win, dtype=np.int8)
    for f_i, (_, va_i) in enumerate(sgkf.split(np.arange(n_win), is_anom_all, groups=block_id_all)):
        folds_all[va_i] = f_i

    diff_prev = np.zeros(n_win, dtype=bool)
    diff_next = np.zeros(n_win, dtype=bool)
    diff_prev[1:] = folds_all[1:] != folds_all[:-1]
    diff_next[:-1] = folds_all[:-1] != folds_all[1:]
    clean_mask = ~(diff_prev | diff_next)
    return folds_all, clean_mask


def load_dataset_tensors(ds_name: str, win_label: str, sel_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ds_name != "Dataset_Completo":
        npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
        dados = np.load(npz_path)
        s_min = dados["scaler_min_per_fold"]
        s_max = dados["scaler_max_per_fold"]
        c_min = np.mean(s_min, axis=0)[None, :, None]
        c_max = np.mean(s_max, axis=0)[None, :, None]
        denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

        x1d_raw = dados["X_1d"][sel_idx].astype(np.float32)
        x1d_sel = np.clip(2.0 * (x1d_raw - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
        x2d_sel = np.ascontiguousarray(dados["X_gadf"][sel_idx], dtype=np.float16)
        dados.close()
        return x1d_sel, x2d_sel

    n_sel = len(sel_idx)
    n_pumps = len(INDIVIDUAL_PUMPS)
    x1d_sel = None
    x2d_sel = np.empty((n_sel, 4 * n_pumps, 32, 32), dtype=np.float16)
    for p_idx, p_name in enumerate(INDIVIDUAL_PUMPS):
        p_npz = np.load(os.path.join(FOLDS_DIR, p_name, f"sequencias_{p_name}_{win_label}.npz"))
        s_min = p_npz["scaler_min_per_fold"]
        s_max = p_npz["scaler_max_per_fold"]
        c_min = np.mean(s_min, axis=0)[None, :, None]
        c_max = np.mean(s_max, axis=0)[None, :, None]
        denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

        x1d_p = p_npz["X_1d"][sel_idx].astype(np.float32)
        if x1d_sel is None:
            x1d_sel = np.empty((n_sel, 4 * n_pumps, x1d_p.shape[2]), dtype=np.float32)
        x1d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :] = np.clip(2.0 * (x1d_p - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
        x2d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :, :] = p_npz["X_gadf"][sel_idx]
        p_npz.close()
    return x1d_sel, x2d_sel


def find_histogram_thresholds(err_norm: np.ndarray, err_anom: np.ndarray, n_bins: int = 120) -> tuple[float, float, float]:
    """
    Constrói o histograma de 120 bins sobre o score do erro de reconstrução E(x) e encontra:
    1. `tau_hist`  : O limiar de bin do histograma que maximiza o F1-Score empírico (interseção
                     que minimiza a soma ponderada de Falsos Positivos e Falsos Negativos).
    2. `tau_baixo` : Fronteira inferior da zona intermediária do operador (captura a cauda
                     inferior das anomalias [FN], até o percentil 10% dos anômalos ou onde a
                     probabilidade empírica de anomalia no bin supera 12%).
    3. `tau_alto`  : Fronteira superior da zona intermediária do operador (captura a cauda
                     superior dos normais [FP], até o percentil 98.5% dos normais ou onde a
                     probabilidade empírica de anomalia no bin supera 88%).
    """
    lo = float(min(np.min(err_norm), np.min(err_anom)))
    hi = float(max(np.max(err_norm), np.max(err_anom)))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cnt_n, _ = np.histogram(err_norm, bins=edges)
    cnt_a, _ = np.histogram(err_anom, bins=edges)

    # Suavização de 3 bins no histograma
    kernel = np.array([0.25, 0.50, 0.25])
    cnt_n_s = np.convolve(cnt_n.astype(np.float64), kernel, mode="same")
    cnt_a_s = np.convolve(cnt_a.astype(np.float64), kernel, mode="same")

    # Para cada corte possível de bin k, calcula VP, FP, FN acumulados no histograma
    vp_bins = np.cumsum(cnt_a_s[::-1])[::-1]
    fp_bins = np.cumsum(cnt_n_s[::-1])[::-1]
    fn_bins = np.sum(cnt_a_s) - vp_bins

    f1_bins = (2.0 * vp_bins) / (2.0 * vp_bins + fp_bins + fn_bins + 1e-8)
    best_idx = int(np.argmax(f1_bins))
    tau_hist = float(centers[best_idx])

    # Probabilidade empírica de anomalia em cada bin do histograma
    prob_a_bin = cnt_a_s / (cnt_n_s + cnt_a_s + 1e-8)

    # Limiar inferior (captura os FN abaixo de tau_hist)
    p08_anom = float(np.percentile(err_anom, 8))
    p90_norm = float(np.percentile(err_norm, 90))
    bins_low = centers[prob_a_bin >= 0.12]
    cand_low = float(bins_low[0]) if len(bins_low) > 0 else p08_anom
    tau_baixo = float(np.clip(min(cand_low, p08_anom, p90_norm), lo, tau_hist * 0.85))

    # Limiar superior (captura os FP acima de tau_hist)
    p985_norm = float(np.percentile(err_norm, 98.5))
    p45_anom = float(np.percentile(err_anom, 45))
    bins_high = centers[(centers >= tau_hist) & (prob_a_bin >= 0.88)]
    cand_high = float(bins_high[0]) if len(bins_high) > 0 else p985_norm
    tau_alto = float(np.clip(max(cand_high, p985_norm), min(tau_hist * 1.15, p45_anom), hi))

    if tau_baixo >= tau_hist:
        tau_baixo = tau_hist * 0.75
    if tau_alto <= tau_hist:
        tau_alto = min(hi, tau_hist * 1.25 + 0.05)

    return tau_baixo, tau_hist, tau_alto


def run_reconstruction_histogram_5folds(
    arch_name: str,
    x1d_all: np.ndarray,
    x2d_all: np.ndarray,
    y_all: np.ndarray,
    folds_cv: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> dict:
    """
    Treina o Autoencoder (reconstrução apenas nos normais + calibração residual) em GPU
    com preloading em VRAM e define os limiares diretamente no histograma de cada fold.
    """
    n_samples, n_channels, seq_len = x1d_all.shape
    c_2d = x2d_all.shape[1]
    aux_dim = 5 * n_channels + c_2d

    oof_err_score = np.zeros(n_samples, dtype=np.float32)
    oof_unsup_z = np.zeros(n_samples, dtype=np.float32)
    oof_tau_low = np.zeros(n_samples, dtype=np.float32)
    oof_tau_hist = np.zeros(n_samples, dtype=np.float32)
    oof_tau_high = np.zeros(n_samples, dtype=np.float32)

    use_amp = (device.type == "cuda")
    unique_folds = np.unique(folds_cv)

    for val_fold in unique_folds:
        tr_idx = np.flatnonzero(folds_cv != val_fold)
        va_idx = np.flatnonzero(folds_cv == val_fold)

        bytes_fold = x1d_all[tr_idx].nbytes + x2d_all[tr_idx].nbytes + x1d_all[va_idx].nbytes + x2d_all[va_idx].nbytes
        preload_gpu = (device.type == "cuda") and (bytes_fold < 2.5 * (1024 ** 3))

        if preload_gpu:
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

        torch.manual_seed(SEED + int(val_fold))
        model = AutoencoderModel(arch_name, n_channels, seq_len).to(device)
        err_calibrator = nn.Sequential(
            nn.BatchNorm1d(LATENT_DIM + aux_dim),
            nn.Linear(LATENT_DIM + aux_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        ).to(device)

        params = list(model.parameters()) + list(err_calibrator.parameters())
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_pos_tr = float(np.sum(y_all[tr_idx] == 1))
        n_neg_tr = float(np.sum(y_all[tr_idx] == 0))
        pos_weight = torch.tensor([min(50.0, max(1.0, n_neg_tr / max(1.0, n_pos_tr)))], dtype=torch.float32, device=device)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_tr = len(tr_idx)
        model.train()
        err_calibrator.train()
        for _ in range(epochs):
            perm = torch.randperm(n_tr, device=device if preload_gpu else "cpu")
            for start_b in range(0, n_tr, batch_size):
                b_idx = perm[start_b : start_b + batch_size]
                if len(b_idx) <= 1:
                    continue
                if preload_gpu:
                    b1 = t_x1d_tr[b_idx].float()
                    b2 = t_x2d_tr[b_idx].float()
                    by = t_y_tr[b_idx]
                else:
                    b1 = t_x1d_tr[b_idx].to(device, non_blocking=True).float()
                    b2 = t_x2d_tr[b_idx].to(device, non_blocking=True).float()
                    by = t_y_tr[b_idx].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b1, b2)
                    norm_b = (by.squeeze(1) < 0.5)
                    if torch.any(norm_b):
                        loss_rec_1d = ((rec_1d[norm_b] - b1[norm_b]) ** 2).mean()
                        if rec_2d is not None:
                            loss_rec_2d = ((rec_2d[norm_b] - b2[norm_b]) ** 2).mean()
                            loss_rec = loss_rec_1d + 0.5 * loss_rec_2d
                        else:
                            loss_rec = loss_rec_1d
                    else:
                        loss_rec = torch.tensor(0.0, device=device)

                    aux_f = compute_auxiliary_signal_features(b1, rec_1d.detach(), rec_2d.detach() if rec_2d is not None else None, b2)
                    logits = err_calibrator(torch.cat([z, aux_f], dim=-1))
                    loss_cal = bce_loss_fn(logits, by)
                    loss = loss_rec + loss_cal

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Inferência dos scores de erro no treino (para montar o histograma) e na validação
        model.eval()
        err_calibrator.eval()
        eval_bs = max(batch_size, 512)

        def eval_tensors(tx1, tx2):
            scores_list, aux_list = [], []
            with torch.no_grad():
                for i in range(0, len(tx1), eval_bs):
                    b1 = tx1[i : i + eval_bs].float() if preload_gpu else tx1[i : i + eval_bs].to(device, non_blocking=True).float()
                    b2 = tx2[i : i + eval_bs].float() if preload_gpu else tx2[i : i + eval_bs].to(device, non_blocking=True).float()
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        z_b, r1_b, r2_b = extract_latent_and_reconstructions(model, b1, b2)
                        aux_b = compute_auxiliary_signal_features(b1, r1_b, r2_b, b2)
                        sc_b = torch.sigmoid(err_calibrator(torch.cat([z_b, aux_b], dim=-1)).squeeze(1))
                    scores_list.append(sc_b.float().cpu().numpy())
                    aux_list.append(aux_b.float().cpu().numpy())
            return np.concatenate(scores_list, axis=0), np.concatenate(aux_list, axis=0)

        tr_scores, tr_aux = eval_tensors(t_x1d_tr, t_x2d_tr)
        va_scores, va_aux = eval_tensors(t_x1d_va, t_x2d_va)

        y_tr_np = y_all[tr_idx]
        mu_norm = np.mean(tr_aux[y_tr_np == 0], axis=0, keepdims=True)
        sd_norm = np.std(tr_aux[y_tr_np == 0], axis=0, keepdims=True) + 1e-5
        va_z_unsup = np.mean(np.abs(va_aux - mu_norm) / sd_norm, axis=1)

        # Constrói o histograma do score de erro no treino e extrai tau_baixo, tau_hist, tau_alto
        err_norm_tr = tr_scores[y_tr_np == 0]
        err_anom_tr = tr_scores[y_tr_np == 1] if np.any(y_tr_np == 1) else err_norm_tr
        tau_low, tau_hist, tau_high = find_histogram_thresholds(err_norm_tr, err_anom_tr)

        oof_err_score[va_idx] = va_scores
        oof_unsup_z[va_idx] = va_z_unsup
        oof_tau_low[va_idx] = tau_low
        oof_tau_hist[va_idx] = tau_hist
        oof_tau_high[va_idx] = tau_high

        del t_x1d_tr, t_x2d_tr, t_y_tr, t_x1d_va, t_x2d_va, model, err_calibrator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "err_score": oof_err_score,
        "unsup_z": oof_unsup_z,
        "tau_low": oof_tau_low,
        "tau_hist": oof_tau_hist,
        "tau_high": oof_tau_high,
    }


def evaluate_histogram_zones(y_true: np.ndarray, res: dict) -> dict:
    err = res["err_score"]
    unsup_z = res["unsup_z"]
    tau_low = res["tau_low"]
    tau_hist = res["tau_hist"]
    tau_high = res["tau_high"]
    n_total = len(y_true)

    roc_auc = float(roc_auc_score(y_true, err)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, err)) if len(np.unique(y_true)) > 1 else 0.0
    roc_unsup = float(roc_auc_score(y_true, unsup_z)) if len(np.unique(y_true)) > 1 else 0.5

    # 1. Corte Único pelo Histograma (Sem Operador)
    pred_single = (err >= tau_hist).astype(np.int32)
    vn_s = int(np.sum((y_true == 0) & (pred_single == 0)))
    fp_s = int(np.sum((y_true == 0) & (pred_single == 1)))
    fn_s = int(np.sum((y_true == 1) & (pred_single == 0)))
    vp_s = int(np.sum((y_true == 1) & (pred_single == 1)))
    acc_s = (vn_s + vp_s) / max(1, n_total)
    prec_s = vp_s / max(1, vp_s + fp_s)
    rec_s = vp_s / max(1, vp_s + fn_s)
    f1_s = 2.0 * prec_s * rec_s / max(1e-8, prec_s + rec_s)
    far_s = 100.0 * fp_s / max(1, vn_s + fp_s)

    # 2. Divisão em 3 Zonas pelo Histograma (Normal Auto | Zona do Operador | Anomalia Auto)
    zone_auto_norm = err < tau_low
    zone_operator = (err >= tau_low) & (err <= tau_high)
    zone_auto_anom = err > tau_high
    zone_auto_all = zone_auto_norm | zone_auto_anom

    n_auto = int(np.sum(zone_auto_all))
    n_oper = int(np.sum(zone_operator))
    pct_auto = 100.0 * n_auto / max(1, n_total)
    pct_oper = 100.0 * n_oper / max(1, n_total)

    # Quantos dos erros (FP e FN) do corte único estavam na Zona Intermediária do Operador?
    fp_mask_single = (y_true == 0) & (pred_single == 1)
    fn_mask_single = (y_true == 1) & (pred_single == 0)
    fp_in_oper = int(np.sum(fp_mask_single & zone_operator))
    fn_in_oper = int(np.sum(fn_mask_single & zone_operator))
    total_err_single = fp_s + fn_s
    captured_err_oper = fp_in_oper + fn_in_oper
    pct_errors_captured = 100.0 * captured_err_oper / max(1, total_err_single)

    # Desempenho APENAS nas Decisões Automáticas (fora da faixa do operador)
    vn_auto = int(np.sum((y_true == 0) & zone_auto_norm))
    fp_auto = int(np.sum((y_true == 0) & zone_auto_anom))
    fn_auto = int(np.sum((y_true == 1) & zone_auto_norm))
    vp_auto = int(np.sum((y_true == 1) & zone_auto_anom))
    acc_auto = (vn_auto + vp_auto) / max(1, n_auto)
    prec_auto = vp_auto / max(1, vp_auto + fp_auto)
    rec_auto = vp_auto / max(1, vp_auto + fn_auto)
    f1_auto = 2.0 * prec_auto * rec_auto / max(1e-8, prec_auto + rec_auto)
    far_auto = 100.0 * fp_auto / max(1, vn_auto + fp_auto)

    # 3. Desempenho Final Human-in-the-Loop (Modelo decide Zonas 1 e 3 + Operador resolve Zona 2)
    vn_hitl = vn_auto + int(np.sum((y_true == 0) & zone_operator))
    vp_hitl = vp_auto + int(np.sum((y_true == 1) & zone_operator))
    fp_hitl = fp_auto
    fn_hitl = fn_auto
    acc_hitl = (vn_hitl + vp_hitl) / max(1, n_total)
    prec_hitl = vp_hitl / max(1, vp_hitl + fp_hitl)
    rec_hitl = vp_hitl / max(1, vp_hitl + fn_hitl)
    f1_hitl = 2.0 * prec_hitl * rec_hitl / max(1e-8, prec_hitl + rec_hitl)
    far_hitl = 100.0 * fp_hitl / max(1, vn_hitl + fp_hitl)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc_unsup": round(roc_unsup, 4),
        "tau_low_mean": round(float(np.mean(tau_low)), 4),
        "tau_hist_mean": round(float(np.mean(tau_hist)), 4),
        "tau_high_mean": round(float(np.mean(tau_high)), 4),
        # Corte Único
        "single_VN": vn_s,
        "single_FP": fp_s,
        "single_FN": fn_s,
        "single_VP": vp_s,
        "single_accuracy": round(acc_s, 4),
        "single_precision": round(prec_s, 4),
        "single_recall": round(rec_s, 4),
        "single_f1": round(f1_s, 4),
        "single_far_pct": round(far_s, 2),
        # Zona Intermediária (Operador)
        "n_auto_decided": n_auto,
        "n_operator_review": n_oper,
        "pct_auto_decided": round(pct_auto, 2),
        "pct_operator_review": round(pct_oper, 2),
        "fp_captured_by_operator": fp_in_oper,
        "fn_captured_by_operator": fn_in_oper,
        "pct_fp_fn_captured_in_zone": round(pct_errors_captured, 2),
        # Só Automático
        "auto_only_accuracy": round(acc_auto, 4),
        "auto_only_precision": round(prec_auto, 4),
        "auto_only_recall": round(rec_auto, 4),
        "auto_only_f1": round(f1_auto, 4),
        "auto_only_far_pct": round(far_auto, 2),
        # Sistema Final (Modelo + Operador na Zona Cinzenta)
        "hitl_VN": vn_hitl,
        "hitl_FP": fp_hitl,
        "hitl_FN": fn_hitl,
        "hitl_VP": vp_hitl,
        "hitl_accuracy": round(acc_hitl, 4),
        "hitl_precision": round(prec_hitl, 4),
        "hitl_recall": round(rec_hitl, 4),
        "hitl_f1": round(f1_hitl, 4),
        "hitl_far_pct": round(far_hitl, 2),
    }


def plot_histograms_by_pump(pump_hist_data: dict, out_path: str):
    """Plota os histogramas do score do erro de reconstrução com as 3 zonas (Normal, Operador [FP/FN], Anomalia)."""
    ds_keys = [d for d in DATASETS if d in pump_hist_data]
    n_ds = len(ds_keys)
    cols = 2
    rows = (n_ds + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 4.4 * rows), dpi=300, facecolor="white")
    axes = axes.flatten()

    fig.suptitle(
        "Histograma do Score do Erro de Reconstrução com Zona Intermediária do Operador (FP e FN)",
        fontsize=14.5, fontweight="bold", color="#0F172A", y=0.985,
    )
    fig.text(
        0.5, 0.958,
        "Zona Verde (E < τ_baixo): Normal Automático | Zona Amarela [τ_baixo, τ_alto]: Avaliação do Operador (FP/FN) | Zona Vermelha (E > τ_alto): Anomalia Automática",
        ha="center", va="center", fontsize=10.0, color="#475569",
    )

    for idx, ds_name in enumerate(ds_keys):
        ax = axes[idx]
        d = pump_hist_data[ds_name]
        err = d["err"]
        y = d["y"]
        t_low = float(np.mean(d["tau_low"]))
        t_hist = float(np.mean(d["tau_hist"]))
        t_high = float(np.mean(d["tau_high"]))
        m = d["metrics"]

        err_norm = err[y == 0]
        err_anom = err[y == 1]
        bins = np.linspace(0.0, 1.0, 50)

        ax.hist(err_norm, bins=bins, density=True, alpha=0.65, color="#2563EB", edgecolor="#1D4ED8", lw=0.6, label=f"Normal (n={len(err_norm)})")
        ax.hist(err_anom, bins=bins, density=True, alpha=0.65, color="#DC2626", edgecolor="#B91C1C", lw=0.6, label=f"Anomalia (n={len(err_anom)})")

        ax.axvspan(0.0, t_low, color="#DCFCE7", alpha=0.50, label="Zona 1: Normal Auto")
        ax.axvspan(t_low, t_high, color="#FEF08A", alpha=0.65, label=f"Zona 2: Operador ({m['pct_operator_review']:.1f}%)")
        ax.axvspan(t_high, 1.0, color="#FEE2E2", alpha=0.45, label="Zona 3: Anomalia Auto")

        ax.axvline(t_low, color="#16A34A", linestyle="--", lw=1.6)
        ax.axvline(t_hist, color="#0F172A", linestyle="-", lw=2.0, label=f"Limiar Único τ={t_hist:.2f}")
        ax.axvline(t_high, color="#DC2626", linestyle="--", lw=1.6)

        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        ax.set_title(
            f"{ds_name} ({d['arch']}) | F1 Único: {m['single_f1']:.3f} → F1 c/ Operador: {m['hitl_f1']:.3f} "
            f"| FP+FN Capturados: {m['pct_fp_fn_captured_in_zone']:.1f}%",
            fontsize=10.0, fontweight="bold", color="#0F172A",
        )
        ax.set_xlabel("Score do Erro de Reconstrução Calibrado E(x)", fontsize=9.0)
        ax.set_ylabel("Densidade (Escala Log)", fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper center", ncol=2, fontsize=7.3, framealpha=0.92)

    for j in range(len(ds_keys), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.01, 1, 0.945])
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico de histogramas salvo em: {out_path}", flush=True)


def plot_architecture_comparison(df_rank: pd.DataFrame, out_path: str):
    """Plota o comparativo entre Corte Único no Histograma vs. Decisão com Revisão do Operador."""
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.2), dpi=300, facecolor="white")
    fig.suptitle(
        "Comparação por Arquitetura: Limiar Único no Histograma vs. Duplo Limiar com Revisão do Operador (GPU)",
        fontsize=13.8, fontweight="bold", color="#0F172A", y=0.98,
    )

    archs = df_rank["architecture"].tolist()
    x = np.arange(len(archs))
    w = 0.26

    ax1 = axes[0]
    b1 = ax1.bar(x - w, df_rank["single_f1_mean"], width=w, color="#64748B", label="Limiar Único Histograma (Sem Operador)")
    b2 = ax1.bar(x, df_rank["auto_only_f1_mean"], width=w, color="#0284C7", label="Só Decisões Automáticas (Zonas 1 e 3)")
    b3 = ax1.bar(x + w, df_rank["hitl_f1_mean"], width=w, color="#16A34A", label="Sistema Final c/ Operador na Zona 2 (FP/FN)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, rotation=22, ha="right", fontsize=8.8)
    ax1.set_ylim(0.70, 1.03)
    ax1.set_ylabel("F1-Score Médio", fontsize=10.0, fontweight="bold")
    ax1.set_title("Ganho de F1-Score ao Isolar FP e FN na Zona do Operador", fontsize=11.0, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")

    ax2 = axes[1]
    c1 = ax2.bar(x - w / 2, df_rank["pct_fp_fn_captured_mean"], width=w * 1.15, color="#D97706", label="% dos FP + FN Capturados na Zona do Operador")
    c2 = ax2.bar(x + w / 2, df_rank["pct_operator_review_mean"], width=w * 1.15, color="#2563EB", label="% do Total de Janelas Enviadas ao Operador")
    ax2.set_xticks(x)
    ax2.set_xticklabels(archs, rotation=22, ha="right", fontsize=8.8)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Porcentagem (%)", fontsize=10.0, fontweight="bold")
    ax2.set_title("Eficiência da Triagem: Erros Capturados vs. Esforço do Operador", fontsize=11.0, fontweight="bold")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.3)
    for bars in (c1, c2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico comparativo de arquiteturas salvo em: {out_path}", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("================================================================================", flush=True)
    print(f"AVALIAÇÃO POR HISTOGRAMA DO ERRO DE RECONSTRUÇÃO + REVISÃO DO OPERADOR (FP/FN)", flush=True)
    print(f"Dispositivo : {device} ({gpu_name})", flush=True)
    print(f"Configuração: window={args.window} | epochs={args.epochs} | batch_size={args.batch_size} | max_samples={args.max_samples}", flush=True)
    print("================================================================================", flush=True)

    rng = np.random.default_rng(SEED)
    all_rows = []
    operator_audit_rows = []
    pump_best_hist = {}

    for ds_name in DATASETS:
        t_ds = time.time()
        f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{args.window}.csv")
        df_f = pd.read_csv(f_csv)
        is_anom_all = df_f["is_annual_2_5_anomaly"].to_numpy(dtype=np.int8)
        block_id_all = df_f["block_id"].to_numpy(dtype=np.int32)

        folds_all, clean_all = compute_clean_group_folds(is_anom_all, block_id_all, N_SPLITS)

        norm_mask = (is_anom_all == 0) & clean_all
        anom_mask = (is_anom_all == 1) & clean_all
        if int(np.sum(anom_mask)) < 15:
            print(f"[PULADO] {ds_name}: poucas anomalias (<15) para montar histograma em 5 folds.", flush=True)
            continue

        valid_norm = np.flatnonzero(norm_mask)
        valid_anom = np.flatnonzero(anom_mask)

        norm_idx = valid_norm if len(valid_norm) <= args.max_samples else np.sort(rng.choice(valid_norm, size=args.max_samples, replace=False))
        anom_idx = valid_anom if len(valid_anom) <= args.max_samples else np.sort(rng.choice(valid_anom, size=args.max_samples, replace=False))

        sel_idx = np.sort(np.concatenate([norm_idx, anom_idx]))
        y_sel = is_anom_all[sel_idx].astype(np.int32)
        folds_sel = folds_all[sel_idx]
        t0_sel = df_f["start_timestamp"].iloc[sel_idx].to_numpy()
        t1_sel = df_f["end_timestamp"].iloc[sel_idx].to_numpy()

        x1d_sel, x2d_sel = load_dataset_tensors(ds_name, args.window, sel_idx)

        best_f1_ds = -1.0
        for arch_name in ARCHITECTURES:
            res = run_reconstruction_histogram_5folds(
                arch_name=arch_name,
                x1d_all=x1d_sel,
                x2d_all=x2d_sel,
                y_all=y_sel,
                folds_cv=folds_sel,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            m = evaluate_histogram_zones(y_sel, res)
            row = {
                "dataset": ds_name,
                "window": args.window,
                "architecture": arch_name,
                "n_normal": int(np.sum(y_sel == 0)),
                "n_anomaly": int(np.sum(y_sel == 1)),
                **m,
            }
            all_rows.append(row)

            if m["hitl_f1"] > best_f1_ds:
                best_f1_ds = m["hitl_f1"]
                pump_best_hist[ds_name] = {
                    "arch": arch_name,
                    "err": res["err_score"],
                    "y": y_sel,
                    "tau_low": res["tau_low"],
                    "tau_hist": res["tau_hist"],
                    "tau_high": res["tau_high"],
                    "metrics": m,
                    "t0": t0_sel,
                    "t1": t1_sel,
                }

        b_info = pump_best_hist[ds_name]
        err_b = b_info["err"]
        tl_b, th_b, thi_b = b_info["tau_low"], b_info["tau_hist"], b_info["tau_high"]
        y_b = b_info["y"]
        in_zone = np.flatnonzero((err_b >= tl_b) & (err_b <= thi_b))
        for idx_z in in_zone:
            pred_s = 1 if err_b[idx_z] >= th_b[idx_z] else 0
            real_y = int(y_b[idx_z])
            if real_y == 0 and pred_s == 1:
                tipo_caso = "FP_Capturado_Para_Operador"
            elif real_y == 1 and pred_s == 0:
                tipo_caso = "FN_Capturado_Para_Operador"
            elif real_y == 1 and pred_s == 1:
                tipo_caso = "VP_Fronteira_Confirmado_Operador"
            else:
                tipo_caso = "VN_Fronteira_Confirmado_Operador"
            operator_audit_rows.append({
                "dataset": ds_name,
                "architecture": b_info["arch"],
                "start_timestamp": b_info["t0"][idx_z],
                "end_timestamp": b_info["t1"][idx_z],
                "erro_reconstrucao": round(float(err_b[idx_z]), 4),
                "tau_baixo": round(float(tl_b[idx_z]), 4),
                "tau_unico_hist": round(float(th_b[idx_z]), 4),
                "tau_alto": round(float(thi_b[idx_z]), 4),
                "rotulo_real_operador": real_y,
                "decisao_corte_unico": pred_s,
                "classificacao_triagem": tipo_caso,
            })

        bm = b_info["metrics"]
        print(
            f"[OK] {ds_name:16s} | Melhor: {b_info['arch']:24s} | "
            f"F1 Corte Único={bm['single_f1']:.4f} (FP={bm['single_FP']}, FN={bm['single_FN']}) -> "
            f"F1 Só Auto={bm['auto_only_f1']:.4f} | F1 c/ Operador={bm['hitl_f1']:.4f} | "
            f"Zona Operador={bm['pct_operator_review']:.1f}% das janelas (capturou {bm['pct_fp_fn_captured_in_zone']:.1f}% dos FP+FN) | "
            f"{time.time() - t_ds:.1f}s",
            flush=True,
        )

        del x1d_sel, x2d_sel
        gc.collect()

    df_all = pd.DataFrame(all_rows)
    csv_det = os.path.join(RESULTS_DIR, "avaliacao_histograma_erro_zona_operador_detalhado.csv")
    df_all.to_csv(csv_det, index=False)

    df_rank = (
        df_all.groupby("architecture", as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_unsup_mean=("roc_auc_unsup", "mean"),
            single_f1_mean=("single_f1", "mean"),
            single_precision_mean=("single_precision", "mean"),
            single_recall_mean=("single_recall", "mean"),
            single_far_pct_mean=("single_far_pct", "mean"),
            auto_only_f1_mean=("auto_only_f1", "mean"),
            auto_only_precision_mean=("auto_only_precision", "mean"),
            auto_only_recall_mean=("auto_only_recall", "mean"),
            auto_only_far_pct_mean=("auto_only_far_pct", "mean"),
            hitl_f1_mean=("hitl_f1", "mean"),
            hitl_precision_mean=("hitl_precision", "mean"),
            hitl_recall_mean=("hitl_recall", "mean"),
            hitl_far_pct_mean=("hitl_far_pct", "mean"),
            pct_auto_decided_mean=("pct_auto_decided", "mean"),
            pct_operator_review_mean=("pct_operator_review", "mean"),
            pct_fp_fn_captured_mean=("pct_fp_fn_captured_in_zone", "mean"),
        )
        .sort_values("hitl_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    csv_rank = os.path.join(RESULTS_DIR, "resumo_histograma_erro_zona_operador_arquiteturas.csv")
    df_rank.to_csv(csv_rank, index=False)

    df_audit = pd.DataFrame(operator_audit_rows)
    csv_audit = os.path.join(RESULTS_DIR, "janelas_intermediarias_revisao_operador.csv")
    df_audit.to_csv(csv_audit, index=False)

    fig_hist_path = os.path.join(PLOTS_DIR, "histogramas_erro_reconstrucao_zona_operador.png")
    fig_comp_path = os.path.join(PLOTS_DIR, "comparacao_histograma_corte_unico_vs_operador.png")
    plot_histograms_by_pump(pump_best_hist, fig_hist_path)
    plot_architecture_comparison(df_rank, fig_comp_path)

    print("\n================================================================================", flush=True)
    print("RANKING GLOBAL — HISTOGRAMA DO ERRO DE RECONSTRUÇÃO + ZONA DO OPERADOR (FP/FN)", flush=True)
    print("================================================================================", flush=True)
    print(df_rank.to_string(index=False), flush=True)
    print(f"\nTempo total em GPU: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
