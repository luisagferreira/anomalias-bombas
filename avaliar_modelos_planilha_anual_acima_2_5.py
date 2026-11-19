"""
Avalia as 7 arquiteturas (1D, 2D GAF, LSTM, Transformer e Híbridos) em Validação Cruzada
Estratificada de 5 Folds (Out-of-Fold) usando a Nova Planilha Anual Completa com
Desvio Padrão de 5 em 5 minutos da Corrente >= 2.5 (`relatorio_anomalias_ano_completo_acima_2_5.csv`).
"""

import os
import gc
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treinar_modelos_1d_2d_gaf import (
    stratified_subsample_indices,
    ARCHITECTURES,
    INDIVIDUAL_PUMPS,
    DATASETS,
    WINDOWS,
    SEED,
)
from avaliar_deteccao_anomalias_operador import (
    evaluate_architecture_5folds,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
torch.set_num_threads(max(1, os.cpu_count() or 4))

BASE_DIR = "Dataset_completo_bombas"
FOLDS_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")
RESULTS_DIR = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D")
ANOM_RESULTS_DIR = os.path.join(RESULTS_DIR, "Deteccao_Anomalias_Operador")
ANOM_PLOTS_DIR = os.path.join(ANOM_RESULTS_DIR, "Figuras_Anomalias_Operador")
ART_DIR = r"C:\Users\202510482\.gemini\antigravity\brain\820f487b-8644-4f3d-b47e-9fd7fab98fec"

ANNUAL_2_5_CSV = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_acima_2_5.csv")
SUMMARY_OUT_CSV = os.path.join(ANOM_RESULTS_DIR, "resumo_deteccao_ano_acima_2_5_por_bomba_janela.csv")

N_SPLITS = 5


def parse_args():
    has_cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description="Avaliação 5-Folds (StdDev >= 2.5) com suporte nativo a GPU CUDA")
    p.add_argument("--device", type=str, default="cuda" if has_cuda else "cpu", help="Dispositivo: 'cuda', 'cuda:0' ou 'cpu'")
    p.add_argument("--epochs", type=int, default=20 if has_cuda else 7, help="Número de épocas por fold (padrão: 20 na GPU, 7 na CPU)")
    p.add_argument("--batch-size", type=int, default=256 if has_cuda else 128, help="Tamanho do lote (padrão: 256 na GPU, 128 na CPU)")
    p.add_argument("--max-samples", type=int, default=2000 if has_cuda else 225, help="Máximo de janelas normais e anômalas por configuração")
    p.add_argument("--full-dataset", action="store_true", help="Se ativado, utiliza 100%% de todas as janelas sem subamostragem")
    p.add_argument("--no-resume", action="store_true", help="Se ativado, recomeça do zero em vez de retomar do CSV salvo")
    return p.parse_args()


def compute_anomaly_group_folds(
    is_anom_all: np.ndarray,
    block_id_all: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Gera 5 folds via StratifiedGroupKFold usando SOMENTE o rótulo de anomalia (0 ou 1)
       e agrupando pelos blocos temporais (`block_id`).
    2. Identifica e exclui dinamicamente as janelas de fronteira entre folds diferentes
       (`folds_all[w] != folds_all[w-1]` ou `folds_all[w] != folds_all[w+1]`), garantindo
       zero vazamento temporal (data leakage) devido ao overlap de 50%.
    Retorna: (folds_all, clean_mask)
    """
    n_win = len(is_anom_all)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_all = np.zeros(n_win, dtype=np.int8)
    for f_i, (_, va_i) in enumerate(sgkf.split(np.arange(n_win), is_anom_all, groups=block_id_all)):
        folds_all[va_i] = f_i

    diff_prev = np.zeros(n_win, dtype=bool)
    diff_next = np.zeros(n_win, dtype=bool)
    diff_prev[1:] = folds_all[1:] != folds_all[:-1]
    diff_next[:-1] = folds_all[:-1] != folds_all[1:]
    is_boundary_overlap = diff_prev | diff_next
    clean_mask = ~is_boundary_overlap
    return folds_all, clean_mask


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    max_norm = 10_000_000 if args.full_dataset else args.max_samples
    max_anom = 10_000_000 if args.full_dataset else args.max_samples

    t0_all = time.time()
    os.makedirs(ANOM_RESULTS_DIR, exist_ok=True)
    os.makedirs(ANOM_PLOTS_DIR, exist_ok=True)

    gpu_info = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"======================================================================", flush=True)
    print(f"Dispositivo ativo : {device} ({gpu_info})", flush=True)
    print(f"Configuração      : epochs={args.epochs} | batch_size={args.batch_size} | max_samples={'FULL (100%)' if args.full_dataset else args.max_samples}", flush=True)
    print(f"======================================================================", flush=True)

    df_annual = pd.read_csv(ANNUAL_2_5_CSV)
    df_annual["Timestamp"] = pd.to_datetime(df_annual["Timestamp Estimado"])
    df_annual["Nome_Bomba"] = df_annual["Nome_Bomba"].astype(str).str.strip()
    df_annual["StdDev_Val"] = pd.to_numeric(df_annual["Valor Estimado (StdDev)"], errors="coerce")

    mapping_rows = []
    rng = np.random.default_rng(SEED)
    summary_rows = []

    print(f"Mapeando {len(df_annual)} registros anômalos (StdDev >= 2.5) nas 28 configurações de janelas...", flush=True)

    for ds_name in DATASETS:
        pump_id = ds_name.replace("Bomba_", "")
        sub_op = df_annual if ds_name == "Dataset_Completo" else df_annual[df_annual["Nome_Bomba"] == pump_id]

        op_grouped = (
            sub_op.groupby("Timestamp", as_index=False)
            .agg(n_records=("StdDev_Val", "count"), max_std=("StdDev_Val", "max"))
            .sort_values("Timestamp")
        )
        t_anom_sec = op_grouped["Timestamp"].values.astype("datetime64[s]").astype(np.int64)
        n_rec_arr = op_grouped["n_records"].to_numpy(dtype=np.int32)
        max_std_arr = op_grouped["max_std"].to_numpy(dtype=np.float32)

        for win_label in WINDOWS:
            f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
            df_f = pd.read_csv(f_csv)
            t0 = pd.to_datetime(df_f["start_timestamp"]).values.astype("datetime64[s]").astype(np.int64)
            t1 = pd.to_datetime(df_f["end_timestamp"]).values.astype("datetime64[s]").astype(np.int64)

            n_win = len(df_f)
            idx_l = np.searchsorted(t_anom_sec, t0, side="left")
            idx_r = np.searchsorted(t_anom_sec, t1, side="right")
            has_match = idx_r > idx_l
            is_anom_25 = has_match.astype(np.int8)

            n_pts = np.zeros(n_win, dtype=np.int16)
            max_std = np.zeros(n_win, dtype=np.float32)
            for w_i in np.flatnonzero(has_match):
                l_i, r_i = idx_l[w_i], idx_r[w_i]
                n_pts[w_i] = int(np.sum(n_rec_arr[l_i:r_i]))
                max_std[w_i] = float(np.max(max_std_arr[l_i:r_i]))

            df_f["is_annual_2_5_anomaly"] = is_anom_25
            df_f["n_annual_2_5_points"] = n_pts
            df_f["max_annual_2_5_stddev"] = np.round(max_std, 4)
            df_f.to_csv(f_csv, index=False)

            orig_fold_dist = df_f.loc[is_anom_25 == 1, "fold"].value_counts().sort_index().to_dict()
            mapping_rows.append({
                "dataset": ds_name,
                "window": win_label,
                "total_windows": n_win,
                "normal_windows_(<2.5)": int(np.sum(is_anom_25 == 0)),
                "anomaly_windows_(>=2.5)": int(np.sum(is_anom_25 == 1)),
                "anomaly_rate_(>=2.5)_pct": round(100.0 * float(np.mean(is_anom_25)), 2),
                "orig_5folds_anomaly_distribution": str(orig_fold_dist),
            })

    df_map_25 = pd.DataFrame(mapping_rows)
    df_map_25.to_csv(os.path.join(ANOM_RESULTS_DIR, "resumo_janelas_anomalias_ano_acima_2_5.csv"), index=False)
    print("\n=== DISTRIBUICAO DE JANELAS ANOMALAS (STDDEV >= 2.5) NO ANO COMPLETO ===", flush=True)
    print(df_map_25.to_string(index=False), flush=True)

    print("\nIniciando avaliação Out-of-Fold (5 Folds) das 7 arquiteturas nas 28 configurações (StdDev >= 2.5)...", flush=True)

    done_configs = set()
    if (not args.no_resume) and os.path.isfile(SUMMARY_OUT_CSV):
        try:
            df_prev = pd.read_csv(SUMMARY_OUT_CSV)
            for (d_k, w_k), g_k in df_prev.groupby(["dataset", "window"]):
                # Retoma configurações que já têm as 7 arquiteturas salvas (e se for full-dataset, que já rodaram com >250 amostras)
                if len(g_k) == len(ARCHITECTURES):
                    if (not args.full_dataset) or (int(g_k["n_normal_evaluated"].iloc[0]) > 500):
                        done_configs.add((d_k, w_k))
                        summary_rows.extend(g_k.to_dict("records"))
            if len(done_configs) > 0:
                print(f"[RESUME] Retomando execução: {len(done_configs)} configurações já concluídas foram recuperadas do CSV!", flush=True)
        except Exception as e:
            print(f"[AVISO] Não foi possível ler checkpoint anterior ({e}); iniciando do zero.", flush=True)

    for ds_name in DATASETS:
        for win_label in WINDOWS:
            if (ds_name, win_label) in done_configs:
                df_curr = pd.DataFrame([r for r in summary_rows if r["dataset"] == ds_name and r["window"] == win_label])
                best_row = df_curr.sort_values("oof_roc_auc_hybrid", ascending=False).iloc[0]
                print(
                    f"[PULADO - JÁ CONCLUÍDO] {ds_name:16s} | {win_label:5s} | Norm={best_row['n_normal_evaluated']:3d}, Anom={best_row['n_anomaly_evaluated']:3d} | "
                    f"Melhor: {best_row['architecture']:24s} (ROC-AUC={best_row['oof_roc_auc_hybrid']:.4f}, "
                    f"Acc={best_row['oof_accuracy_at_best_f1']*100:.1f}%, F1={best_row['oof_best_f1_hybrid']:.4f}, "
                    f"Recall={best_row['oof_recall_at_best_f1']*100:.1f}%)",
                    flush=True,
                )
                continue

            t_cfg = time.time()
            f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
            df_f = pd.read_csv(f_csv)
            is_anom_all = df_f["is_annual_2_5_anomaly"].to_numpy(dtype=np.int8)
            block_id_all = df_f["block_id"].to_numpy(dtype=np.int32)

            folds_all, clean_all = compute_anomaly_group_folds(
                is_anom_all=is_anom_all,
                block_id_all=block_id_all,
                n_splits=N_SPLITS,
                seed=SEED,
            )

            norm_mask = (is_anom_all == 0) & clean_all
            anom_mask = (is_anom_all == 1) & clean_all
            if int(np.sum(anom_mask)) < 20:
                anom_mask = (is_anom_all == 1)
            if int(np.sum(norm_mask)) < 20:
                norm_mask = (is_anom_all == 0)

            valid_norm = np.flatnonzero(norm_mask)
            valid_anom = np.flatnonzero(anom_mask)
            norm_idx = valid_norm if len(valid_norm) <= max_norm else np.sort(rng.choice(valid_norm, size=max_norm, replace=False))
            anom_idx = valid_anom if len(valid_anom) <= max_anom else np.sort(rng.choice(valid_anom, size=max_anom, replace=False))

            sel_idx = np.sort(np.concatenate([norm_idx, anom_idx]))
            y_sel = is_anom_all[sel_idx].astype(np.int32)
            folds_sel = folds_all[sel_idx].astype(np.int8)

            if ds_name != "Dataset_Completo":
                npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
                dados = np.load(npz_path)
                scaler_min = dados["scaler_min_per_fold"]
                scaler_max = dados["scaler_max_per_fold"]
                c_min = np.mean(scaler_min, axis=0)[None, :, None]
                c_max = np.mean(scaler_max, axis=0)[None, :, None]
                denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

                x1d_raw = dados["X_1d"][sel_idx].astype(np.float32)
                x1d_sel = np.clip(2.0 * (x1d_raw - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
                del x1d_raw
                gc.collect()

                # Utiliza somente o GADF: (n_sel, 4, 32, 32) em float16
                x2d_sel = np.ascontiguousarray(dados["X_gadf"][sel_idx], dtype=np.float16)
                dados.close()
                del dados
                gc.collect()
            else:
                # Dataset_Completo construido pela concatenacao dos 6 datasets individuais (sem duplicar .npz)
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
                    x1d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :] = np.clip(
                        2.0 * (x1d_p - c_min) / denom - 1.0, -1.0, 1.0
                    ).astype(np.float32)
                    del x1d_p

                    # Somente GADF: (n_sel, 4, 32, 32)
                    x2d_sel[:, p_idx * 4 : (p_idx + 1) * 4, :, :] = p_npz["X_gadf"][sel_idx]
                    p_npz.close()
                    del p_npz
                    gc.collect()

            for arch_name in ARCHITECTURES:
                fold_metrics, oof_unsup, oof_prob, oof_mse = evaluate_architecture_5folds(
                    arch_name=arch_name,
                    x1d_all=x1d_sel,
                    x2d_all=x2d_sel,
                    y_all=y_sel,
                    folds_cv=folds_sel,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )

                oof_roc_unsup = float(roc_auc_score(y_sel, oof_unsup))
                oof_pr_unsup = float(average_precision_score(y_sel, oof_unsup))
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
                vn = int(np.sum((y_sel == 0) & (y_pred_opt == 0)))
                fp = int(np.sum((y_sel == 0) & (y_pred_opt == 1)))
                fn = int(np.sum((y_sel == 1) & (y_pred_opt == 0)))
                vp = int(np.sum((y_sel == 1) & (y_pred_opt == 1)))
                acc_opt = float((vn + vp) / len(y_sel))
                far_opt = float(fp / max(1, vn + fp))

                unsup_ratio = float(np.mean(oof_unsup[y_sel == 1]) / (np.mean(oof_unsup[y_sel == 0]) + 1e-8))

                summary_rows.append({
                    "dataset": ds_name,
                    "window": win_label,
                    "architecture": arch_name,
                    "n_normal_evaluated": int(np.sum(y_sel == 0)),
                    "n_anomaly_evaluated": int(np.sum(y_sel == 1)),
                    "VN": vn,
                    "FP": fp,
                    "FN": fn,
                    "VP": vp,
                    "unsup_anomaly_score_ratio": round(unsup_ratio, 4),
                    "oof_roc_auc_unsup": round(oof_roc_unsup, 4),
                    "oof_pr_auc_unsup": round(oof_pr_unsup, 4),
                    "oof_roc_auc_hybrid": round(oof_roc_hyb, 4),
                    "oof_pr_auc_hybrid": round(oof_pr_hyb, 4),
                    "oof_accuracy_at_best_f1": round(acc_opt, 4),
                    "oof_best_f1_hybrid": round(best_f1, 4),
                    "oof_precision_at_best_f1": round(best_prec, 4),
                    "oof_recall_at_best_f1": round(best_rec, 4),
                    "oof_false_alarm_rate_pct": round(100.0 * far_opt, 2),
                })

            pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT_CSV, index=False)

            df_curr = pd.DataFrame([r for r in summary_rows if r["dataset"] == ds_name and r["window"] == win_label])
            best_row = df_curr.sort_values("oof_roc_auc_hybrid", ascending=False).iloc[0]
            print(
                f"[OK] {ds_name:16s} | {win_label:5s} | Norm={best_row['n_normal_evaluated']:3d}, Anom={best_row['n_anomaly_evaluated']:3d} | "
                f"Melhor: {best_row['architecture']:24s} (ROC-AUC={best_row['oof_roc_auc_hybrid']:.4f}, "
                f"Acc={best_row['oof_accuracy_at_best_f1']*100:.1f}%, F1={best_row['oof_best_f1_hybrid']:.4f}, "
                f"Recall={best_row['oof_recall_at_best_f1']*100:.1f}%) | {time.time() - t_cfg:.1f}s",
                flush=True,
            )

    df_sum_25 = pd.DataFrame(summary_rows)
    df_sum_25.to_csv(SUMMARY_OUT_CSV, index=False)

    df_rank_25 = (
        df_sum_25.groupby("architecture", as_index=False)
        .agg(
            roc_auc_hybrid_mean=("oof_roc_auc_hybrid", "mean"),
            pr_auc_hybrid_mean=("oof_pr_auc_hybrid", "mean"),
            accuracy_hybrid_mean=("oof_accuracy_at_best_f1", "mean"),
            best_f1_hybrid_mean=("oof_best_f1_hybrid", "mean"),
            precision_hybrid_mean=("oof_precision_at_best_f1", "mean"),
            recall_hybrid_mean=("oof_recall_at_best_f1", "mean"),
            false_alarm_rate_pct_mean=("oof_false_alarm_rate_pct", "mean"),
            roc_auc_unsup_mean=("oof_roc_auc_unsup", "mean"),
            unsup_score_ratio_mean=("unsup_anomaly_score_ratio", "mean"),
        )
        .sort_values("roc_auc_hybrid_mean", ascending=False)
    )
    df_rank_25.to_csv(os.path.join(ANOM_RESULTS_DIR, "ranking_global_deteccao_ano_acima_2_5.csv"), index=False)

    print("\n=== RANKING GLOBAL DAS 7 ARQUITETURAS (ANO COMPLETO STDDEV >= 2.5 - 28 CONFIGURACOES EM 5 FOLDS) ===", flush=True)
    print(df_rank_25.to_string(index=False), flush=True)

    # Figura de Matrizes de Confusão (StdDev >= 2.5)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    axes = axes.flatten()
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        sub = df_sum_25[(df_sum_25["dataset"] == ds) & (df_sum_25["window"] == "15min")].sort_values("oof_roc_auc_hybrid", ascending=False)
        row = sub.iloc[0]
        vn, fp, fn, vp = int(row["VN"]), int(row["FP"]), int(row["FN"]), int(row["VP"])
        tot = vn + fp + fn + vp
        acc = (vn + vp) / tot * 100.0
        rec = vp / max(1, vp + fn) * 100.0
        f1 = row["oof_best_f1_hybrid"]

        cm_pct = np.array([
            [vn / max(1, vn + fp) * 100.0, fp / max(1, vn + fp) * 100.0],
            [fn / max(1, fn + vp) * 100.0, vp / max(1, fn + vp) * 100.0],
        ])
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        labels_txt = [
            [f"VN={vn}\n({cm_pct[0,0]:.1f}% Normais)", f"FP={fp}\n({cm_pct[0,1]:.1f}% F. Alarme)"],
            [f"FN={fn}\n({cm_pct[1,0]:.1f}% Perdidas)", f"VP={vp}\n({cm_pct[1,1]:.1f}% Detectadas)"],
        ]
        for r in range(2):
            for c in range(2):
                color_t = "white" if cm_pct[r, c] > 55 else "black"
                ax.text(c, r, labels_txt[r][c], ha="center", va="center", color=color_t, fontsize=9.5, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: Normal", "Pred: Anomalia"], fontweight="bold")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Real: Normal", "Real: Anomalia"], fontweight="bold")
        ax.set_title(
            f"{ds} (15min - StdDev >= 2.5)\n{row['architecture']}\nAcc={acc:.1f}% | F1={f1:.3f} | Recall={rec:.1f}%",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[7]
    best_arch_global = df_rank_25.iloc[0]["architecture"]
    sub_b = df_sum_25[df_sum_25["architecture"] == best_arch_global]
    vn, fp, fn, vp = int(sub_b["VN"].sum()), int(sub_b["FP"].sum()), int(sub_b["FN"].sum()), int(sub_b["VP"].sum())
    cm_pct = np.array([
        [vn / max(1, vn + fp) * 100.0, fp / max(1, vn + fp) * 100.0],
        [fn / max(1, fn + vp) * 100.0, vp / max(1, fn + vp) * 100.0],
    ])
    ax.imshow(cm_pct, cmap="YlGnBu", vmin=0, vmax=100)
    labels_txt = [
        [f"VN={vn}\n({cm_pct[0,0]:.1f}% Normais)", f"FP={fp}\n({cm_pct[0,1]:.1f}% F. Alarme)"],
        [f"FN={fn}\n({cm_pct[1,0]:.1f}% Perdidas)", f"VP={vp}\n({cm_pct[1,1]:.1f}% Detectadas)"],
    ]
    for r in range(2):
        for c in range(2):
            color_t = "white" if cm_pct[r, c] > 55 else "black"
            ax.text(c, r, labels_txt[r][c], ha="center", va="center", color=color_t, fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: Normal", "Pred: Anomalia"], fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real: Normal", "Real: Anomalia"], fontweight="bold")
    ax.set_title(
        f"ACUMULADO 28 CONFIGS (StdDev >= 2.5)\n{best_arch_global}\nAcc={(vn+vp)/(vn+fp+fn+vp)*100:.1f}% | Recall={vp/(vp+fn)*100:.1f}%",
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        "Matrizes de Confusão Out-of-Fold (5 Folds) — Planilha Anual Completa (StdDev 5min da Corrente >= 2.5)",
        fontsize=14,
        fontweight="bold",
    )
    p_cm_25 = os.path.join(ANOM_PLOTS_DIR, "matrizes_confusao_ano_acima_2_5_15min.png")
    plt.savefig(p_cm_25, dpi=140)
    plt.close(fig)
    if os.path.isdir(ART_DIR):
        shutil.copy2(p_cm_25, os.path.join(ART_DIR, "matrizes_confusao_ano_acima_2_5_15min.png"))
    print(f"\nConcluído em {time.time() - t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
