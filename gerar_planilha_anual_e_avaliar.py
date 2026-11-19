"""
Gera a Planilha Anual Completa de Anomalias (Equivalente à Metodologia do Operador Edcarlos)
baseada no Desvio Padrão de 5 em 5 minutos da Corrente (`BR_YAR_*` / `Corrente_YAR`) na faixa
[4.0, 8.0] para todo o ano de 2026 (01/01/2026 a 26/10/2026), além de validar a aderência
exata com o período informado pelo operador (05/04, 13/04, 17/04 e 21/04/2026) e atualizar
toda a avaliação dos 7 modelos (1D, 2D GAF, LSTM, Transformer e Híbridos) em 5 folds.
"""

import os
import gc
import time
import shutil
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
from avaliar_deteccao_anomalias_operador import (
    extract_latent_and_reconstructions,
    compute_auxiliary_signal_features,
    evaluate_architecture_5folds,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 4))

BASE_DIR = "Dataset_completo_bombas"
FOLDS_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")
RESULTS_DIR = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D")
ANOM_RESULTS_DIR = os.path.join(RESULTS_DIR, "Deteccao_Anomalias_Operador")
ANOM_PLOTS_DIR = os.path.join(ANOM_RESULTS_DIR, "Figuras_Anomalias_Operador")
ART_DIR = r"C:\Users\202510482\.gemini\antigravity\brain\820f487b-8644-4f3d-b47e-9fd7fab98fec"

OPERATOR_CSV_PATH = os.path.join(BASE_DIR, "relatorio_anomalias_operador.csv")
ANNUAL_CSV_PATH = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_5min_corrente.csv")
ANNUAL_XLSX_PATH = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_edcarlos.xlsx")
COMPARISON_APRIL_CSV = os.path.join(ANOM_RESULTS_DIR, "verificacao_periodo_edcarlos_abril2026.csv")

PUMPS_LIST = ["1305", "1307", "2307", "3307", "4307", "5307"]
EDCARLOS_DATES = ["2026-04-05", "2026-04-13", "2026-04-17", "2026-04-21"]

N_SPLITS = 5
MAX_NORMAL_SAMPLES = 300
MAX_ANOMALY_SAMPLES = 300


def build_annual_5min_current_anomaly_table():
    """
    Calcula o desvio padrão de 5 em 5 minutos da corrente (`BR_YAR_*` / `Corrente_YAR`)
    para todas as 6 bombas durante todo o ano de 2026:
    1. A partir da série de 10s (`Bomba_<ID>.csv`): `s_10s_block` (desvio padrão amostral de 30 pontos)
       e `s_10s_rollmax` (máximo do desvio padrão móvel de 5 min dentro do bloco).
    2. A partir da série bruta irregular (`bomba_<ID>_corrente.csv`, quando disponível): `s_raw_block`.
    3. Compara detalhadamente o período de 4 dias do Edcarlos (`05/04`, `13/04`, `17/04`, `21/04/2026`).
    4. Gera a planilha anual completa consolidada (CSV e Excel `.xlsx`).
    """
    df_op = pd.read_csv(OPERATOR_CSV_PATH)
    df_op["Timestamp"] = pd.to_datetime(df_op["Timestamp"])
    df_op["Nome_Bomba"] = df_op["Nome_Bomba"].astype(str).str.strip()

    annual_records_10s = []
    annual_records_hybrid = []
    april_verification_rows = []
    pump_summary_rows = []

    # Dicionário de lookup rápido dos pontos originais digitalizados pelo Edcarlos nas 4 datas de abril
    edcarlos_lookup = {}
    for _, row in df_op.iterrows():
        edcarlos_lookup[(row["Timestamp"], str(row["Nome_Bomba"]))] = float(row["StdDev estimado"])

    for pid in PUMPS_LIST:
        f_10s = os.path.join(BASE_DIR, "Bombas_por_ID", f"Bomba_{pid}.csv")
        df_10s = pd.read_csv(f_10s)
        df_10s["Timestamp"] = pd.to_datetime(df_10s["Timestamp"])
        yar_col = [c for c in df_10s.columns if "YAR" in c][0]

        s_yar = df_10s.set_index("Timestamp")[yar_col]
        # Desvio padrão amostral (ddof=1) em blocos fixos de 5 minutos (30 amostras de 10s)
        std_5m_10s = s_yar.resample("5min", closed="left", label="left").std(ddof=1)
        # Desvio padrão móvel de 5 minutos (máximo no bloco de 5 min - reproduz o envelope visual da curva no SIGMA)
        std_5m_roll = s_yar.rolling("5min").std(ddof=1).resample("5min", closed="left", label="left").max()

        # Tenta carregar a série bruta sem interpolação (bomba_<ID>_corrente.csv) se disponível
        raw_path = rf"C:\Users\202510482\Downloads\Bombas_Polpa_Project-main\Bombas_Polpa_Project-main\data\bomba_{pid}_corrente.csv"
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path)
            df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], format="mixed")
            std_5m_raw = (
                df_raw.set_index("Timestamp")["Corrente_YAR"]
                .resample("5min", closed="left", label="left")
                .std(ddof=1)
                .reindex(std_5m_10s.index)
            )
        else:
            std_5m_raw = std_5m_10s.copy()

        # Verifica os 4 dias de abril de 2026 frente à planilha do Edcarlos
        for dt_str in EDCARLOS_DATES:
            op_sub = df_op[(df_op["Nome_Bomba"] == pid) & (df_op["Timestamp"].dt.strftime("%Y-%m-%d") == dt_str)]
            b10_d = std_5m_10s[std_5m_10s.index.strftime("%Y-%m-%d") == dt_str]
            r10_d = std_5m_roll[std_5m_roll.index.strftime("%Y-%m-%d") == dt_str]
            raw_d = std_5m_raw[std_5m_raw.index.strftime("%Y-%m-%d") == dt_str]

            ts_op = set(op_sub["Timestamp"])
            ts_b10 = set(b10_d[(b10_d >= 4.0) & (b10_d <= 8.0)].index)
            ts_r10 = set(r10_d[(r10_d >= 4.0) & (r10_d <= 8.0)].index)
            ts_raw = set(raw_d[(raw_d >= 4.0) & (raw_d <= 8.0)].index)

            # Calcula MAE e correlação nos timestamps em comum
            if len(op_sub) > 0:
                op_series = op_sub.set_index("Timestamp")["StdDev estimado"]
                b10_aligned = b10_d.reindex(op_series.index)
                raw_aligned = raw_d.reindex(op_series.index)
                mae_b10 = float((op_series - b10_aligned).abs().mean())
                mae_raw = float((op_series - raw_aligned).abs().mean())
            else:
                mae_b10 = np.nan
                mae_raw = np.nan

            april_verification_rows.append({
                "Data": dt_str,
                "Nome_Bomba": pid,
                "Pontos_Edcarlos_Planilha": len(ts_op),
                "Pontos_Calculados_10s_Bloco5m_[4,8]": len(ts_b10),
                "Pontos_Calculados_10s_RollingMax5m_[4,8]": len(ts_r10),
                "Pontos_Calculados_Bruto_Bloco5m_[4,8]": len(ts_raw),
                "Intersecao_Exata_Timestamps_(Bruto_vs_Edcarlos)": len(ts_op & ts_raw),
                "Intersecao_Exata_Timestamps_(10s_Roll_vs_Edcarlos)": len(ts_op & ts_r10),
                "Taxa_Concordancia_Bruto_pct": round(100.0 * len(ts_op & ts_raw) / max(1, len(ts_op)), 1) if len(ts_op) > 0 else 100.0,
                "Taxa_Concordancia_10s_Roll_pct": round(100.0 * len(ts_op & ts_r10) / max(1, len(ts_op)), 1) if len(ts_op) > 0 else 100.0,
                "Media_StdDev_Edcarlos": round(float(op_sub["StdDev estimado"].mean()), 4) if len(op_sub) > 0 else np.nan,
                "Media_StdDev_Calculado_10s": round(float(b10_d[(b10_d >= 4.0) & (b10_d <= 8.0)].mean()), 4) if len(ts_b10) > 0 else np.nan,
                "MAE_Nos_Pontos_Coincidentes_10s": round(mae_b10, 4) if not np.isnan(mae_b10) else np.nan,
            })

        # Filtra para todo o ano os blocos de 5 minutos onde StdDev_5min da corrente está em [4.0, 8.0]
        mask_10s_48 = (std_5m_10s >= 4.0) & (std_5m_10s <= 8.0)
        for ts, val in std_5m_10s[mask_10s_48].items():
            dt_s = ts.strftime("%Y-%m-%d")
            if dt_s in EDCARLOS_DATES and (ts, pid) in edcarlos_lookup:
                obs = "Coincide com planilha original do operador (Edcarlos) + Calculado via StdDev 5min [4.0, 8.0]"
            else:
                obs = "Calculado via StdDev 5min da corrente (BR_YAR) na faixa [4.0, 8.0]"
            annual_records_10s.append({
                "Timestamp": ts,
                "Nome_Bomba": pid,
                "StdDev estimado": round(float(val), 4),
                "StdDev_Corrente_5min_Exato": round(float(val), 4),
                "Observacao": obs,
            })

        # Também cria a versão Integrada (que preserva 100% dos pontos digitalizados pelo Edcarlos nos 4 dias de abril
        # e usa o cálculo automático `std_5m_10s in [4.0, 8.0]` para todos os demais dias do ano!)
        annual_records_hybrid.extend([
            r for r in annual_records_10s if r["Nome_Bomba"] == pid and r["Timestamp"].strftime("%Y-%m-%d") not in EDCARLOS_DATES
        ])
        for _, row in df_op[df_op["Nome_Bomba"] == pid].iterrows():
            ts = row["Timestamp"]
            calc_v = float(std_5m_10s.get(ts, np.nan))
            annual_records_hybrid.append({
                "Timestamp": ts,
                "Nome_Bomba": pid,
                "StdDev estimado": round(float(row["StdDev estimado"]), 4),
                "StdDev_Corrente_5min_Exato": round(calc_v, 4) if not np.isnan(calc_v) else round(float(row["StdDev estimado"]), 4),
                "Observacao": str(row["Observacao"]),
            })

        pump_summary_rows.append({
            "Bomba": f"BP {pid}",
            "Nome_Bomba": pid,
            "Total_Blocos_5min_No_Ano": len(std_5m_10s),
            "Pontos_Edcarlos_(4_dias_Abril)": int((df_op["Nome_Bomba"] == pid).sum()),
            "Pontos_Ano_Completo_Calculados_[4,8]": int(mask_10s_48.sum()),
            "Pontos_Ano_Completo_Integrado_(Edcarlos+Calculado)": int(
                sum(1 for r in annual_records_hybrid if r["Nome_Bomba"] == pid)
            ),
            "Taxa_Ocorrencia_Anual_pct": round(100.0 * float(mask_10s_48.mean()), 3),
            "Minimo_StdDev_[4,8]": round(float(std_5m_10s[mask_10s_48].min()), 4) if mask_10s_48.any() else np.nan,
            "Media_StdDev_[4,8]": round(float(std_5m_10s[mask_10s_48].mean()), 4) if mask_10s_48.any() else np.nan,
            "Maximo_StdDev_[4,8]": round(float(std_5m_10s[mask_10s_48].max()), 4) if mask_10s_48.any() else np.nan,
            "Pontos_Acima_de_8_(Transientes_Partida/Parada)": int((std_5m_10s > 8.0).sum()),
        })

    df_annual_10s = pd.DataFrame(annual_records_10s).sort_values(["Timestamp", "Nome_Bomba"]).reset_index(drop=True)
    df_annual_hyb = pd.DataFrame(annual_records_hybrid).sort_values(["Timestamp", "Nome_Bomba"]).reset_index(drop=True)
    df_april_ver = pd.DataFrame(april_verification_rows)
    df_pump_sum = pd.DataFrame(pump_summary_rows)

    os.makedirs(ANOM_RESULTS_DIR, exist_ok=True)
    df_annual_hyb.to_csv(ANNUAL_CSV_PATH, index=False)
    df_april_ver.to_csv(COMPARISON_APRIL_CSV, index=False)
    df_pump_sum.to_csv(os.path.join(ANOM_RESULTS_DIR, "resumo_anual_5min_corrente_por_bomba.csv"), index=False)

    # Salva arquivo Excel (.xlsx) no exato formato das planilhas do Edcarlos (abas: Ocorrencias 4 a 8, Resumo, Verificacao_Abril, Parametros)
    df_params = pd.DataFrame([
        {"Item": "Periodo analisado", "Valor": "01/01/2026 00:00:00 a 26/10/2026 00:00:00 (298 dias / Ano Completo)"},
        {"Item": "Variavel monitorada", "Valor": "Corrente eletrica do motor da bomba (BR_YAR_BP* / Corrente_YAR)"},
        {"Item": "Intervalo nominal", "Valor": "5 minutos (blocos de 30 amostras de 10 segundos)"},
        {"Item": "Metrica calculada", "Valor": "Desvio padrao amostral (ddof=1) da corrente em cada janela de 5 minutos"},
        {"Item": "Faixa aplicada", "Valor": "4,0 a 8,0 inclusive (mesmo criterio da planilha do operador Edcarlos)"},
        {"Item": "Total de ocorrencias no ano (Calculado Puro 10s)", "Valor": str(len(df_annual_10s))},
        {"Item": "Total de ocorrencias no ano (Integrado com 4 dias do Operador)", "Valor": str(len(df_annual_hyb))},
    ])

    with pd.ExcelWriter(ANNUAL_XLSX_PATH, engine="openpyxl") as writer:
        df_annual_hyb.to_excel(writer, sheet_name="Ocorrencias 4 a 8 (Ano)", index=False)
        df_annual_10s.to_excel(writer, sheet_name="Calculado Puro 10s (Ano)", index=False)
        df_pump_sum.to_excel(writer, sheet_name="Resumo Anual por Bomba", index=False)
        df_april_ver.to_excel(writer, sheet_name="Verificacao Abril Edcarlos", index=False)
        df_params.to_excel(writer, sheet_name="Parametros", index=False)

    print(f"[OK] Planilha anual completa salva em:\n  - CSV : {ANNUAL_CSV_PATH} ({len(df_annual_hyb)} registros)\n  - XLSX: {ANNUAL_XLSX_PATH}")
    print("\n=== RESUMO ANUAL POR BOMBA (STDDEV CORRENTE 5 MIN NA FAIXA [4.0, 8.0]) ===")
    print(df_pump_sum.to_string(index=False))
    print("\n=== VERIFICACAO DO PERIODO DO EDCARLOS (05/04, 13/04, 17/04 E 21/04/2026) ===")
    print(df_april_ver[df_april_ver["Pontos_Edcarlos_Planilha"] > 0].to_string(index=False))

    return df_annual_hyb, df_pump_sum, df_april_ver


def map_and_evaluate_annual_anomalies(df_annual: pd.DataFrame):
    """
    Mapeia a planilha anual completa de anomalias (5 em 5 minutos na faixa [4.0, 8.0])
    para as janelas de 15min, 30min, 1h e 2h das 6 bombas e do Dataset Completo,
    e executa a avaliação Out-of-Fold (5 folds) das 7 arquiteturas.
    """
    mapping_rows = []
    rng = np.random.default_rng(SEED)
    summary_rows = []
    fold_rows = []

    for ds_name in DATASETS:
        pump_id = ds_name.replace("Bomba_", "")
        sub_op = df_annual if ds_name == "Dataset_Completo" else df_annual[df_annual["Nome_Bomba"] == pump_id]

        op_grouped = (
            sub_op.groupby("Timestamp", as_index=False)
            .agg(n_records=("StdDev estimado", "count"), max_std=("StdDev estimado", "max"))
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
            is_anom_annual = has_match.astype(np.int8)

            n_pts = np.zeros(n_win, dtype=np.int16)
            max_std = np.zeros(n_win, dtype=np.float32)
            for w_i in np.flatnonzero(has_match):
                l_i, r_i = idx_l[w_i], idx_r[w_i]
                n_pts[w_i] = int(np.sum(n_rec_arr[l_i:r_i]))
                max_std[w_i] = float(np.max(max_std_arr[l_i:r_i]))

            df_f["is_annual_5min_anomaly"] = is_anom_annual
            df_f["n_annual_5min_points"] = n_pts
            df_f["max_annual_5min_stddev"] = np.round(max_std, 4)
            df_f.to_csv(f_csv, index=False)

            orig_fold_dist = df_f.loc[is_anom_annual == 1, "fold"].value_counts().sort_index().to_dict()
            mapping_rows.append({
                "dataset": ds_name,
                "window": win_label,
                "total_windows": n_win,
                "annual_normal_windows": int(np.sum(is_anom_annual == 0)),
                "annual_anomaly_windows": int(np.sum(is_anom_annual == 1)),
                "annual_anomaly_rate_pct": round(100.0 * float(np.mean(is_anom_annual)), 2),
                "orig_5folds_anomaly_distribution": str(orig_fold_dist),
            })

    df_map_annual = pd.DataFrame(mapping_rows)
    df_map_annual.to_csv(os.path.join(ANOM_RESULTS_DIR, "resumo_janelas_anomalias_ano_completo.csv"), index=False)
    print("\n=== DISTRIBUICAO DE JANELAS ANOMALAS NO ANO COMPLETO (POR BOMBA, JANELA E NOS 5 FOLDS ORIGINAIS) ===")
    print(df_map_annual.to_string(index=False))

    # Agora avalia as 7 arquiteturas nas 28 configurações usando os 5 folds originais (`fold` de `StratifiedGroupKFold`)
    # pois no ano inteiro TODOS os 5 folds originais possuem dezenas/centenas de anomalias!
    print("\nIniciando avaliação Out-of-Fold nos 5 Folds Estratificados Originais (Ano Completo - Todas as 6 Bombas + Dataset Completo)...")

    for ds_name in DATASETS:
        for win_label in WINDOWS:
            t_cfg = time.time()
            f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
            df_f = pd.read_csv(f_csv)
            is_anom_all = df_f["is_annual_5min_anomaly"].to_numpy(dtype=np.int8)

            npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
            dados = np.load(npz_path)

            X_1d_all = dados["X_1d"]
            X_gasf_all = dados["X_gasf"]
            X_gadf_all = dados["X_gadf"]
            folds_orig = dados["fold"]
            strata_all = dados["stratum"]
            clean_all = ~dados["is_boundary_overlap"]
            scaler_min = dados["scaler_min_per_fold"]
            scaler_max = dados["scaler_max_per_fold"]

            # Subamostra estratificada de janelas normais + subamostra estratificada (ou 100%) das janelas anômalas
            norm_mask = (is_anom_all == 0) & clean_all
            anom_mask = (is_anom_all == 1) & clean_all
            if int(np.sum(anom_mask)) < 10:
                anom_mask = (is_anom_all == 1)

            norm_idx = stratified_subsample_indices(norm_mask, strata_all, MAX_NORMAL_SAMPLES, rng)
            anom_idx = stratified_subsample_indices(anom_mask, strata_all, MAX_ANOMALY_SAMPLES, rng)

            sel_idx = np.sort(np.concatenate([norm_idx, anom_idx]))
            y_sel = is_anom_all[sel_idx].astype(np.int32)
            folds_sel = folds_orig[sel_idx].astype(np.int8)

            # Garante que cada um dos 5 folds tenha tanto normais quanto anomalias; se algum fold tiver 0 anomalias (ex: Bomba_5307 2h), usa StratifiedKFold
            has_both_all_folds = all(
                (np.sum((folds_sel == f) & (y_sel == 1)) > 0) and (np.sum((folds_sel == f) & (y_sel == 0)) > 0)
                for f in range(N_SPLITS)
            )
            if not has_both_all_folds:
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                folds_sel = np.zeros(len(sel_idx), dtype=np.int8)
                for f_i, (_, va_i) in enumerate(skf.split(sel_idx, y_sel)):
                    folds_sel[va_i] = f_i

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
                    folds_cv=folds_sel,
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

            df_curr = pd.DataFrame([r for r in summary_rows if r["dataset"] == ds_name and r["window"] == win_label])
            best_row = df_curr.sort_values("oof_roc_auc_hybrid", ascending=False).iloc[0]
            print(
                f"[OK] {ds_name:16s} | {win_label:5s} | AnomAval={best_row['n_anomaly_evaluated']:3d} | "
                f"Melhor: {best_row['architecture']:24s} (ROC-AUC={best_row['oof_roc_auc_hybrid']:.4f}, "
                f"Acc={best_row['oof_accuracy_at_best_f1']*100:.1f}%, F1={best_row['oof_best_f1_hybrid']:.4f}, "
                f"Recall={best_row['oof_recall_at_best_f1']*100:.1f}%) | {time.time() - t_cfg:.1f}s"
            )

    df_sum_annual = pd.DataFrame(summary_rows)
    df_sum_annual.to_csv(os.path.join(ANOM_RESULTS_DIR, "resumo_deteccao_ano_completo_por_bomba_janela.csv"), index=False)

    df_rank_annual = (
        df_sum_annual.groupby("architecture", as_index=False)
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
    df_rank_annual.to_csv(os.path.join(ANOM_RESULTS_DIR, "ranking_global_deteccao_ano_completo.csv"), index=False)

    print("\n=== RANKING GLOBAL DAS 7 ARQUITETURAS (ANO COMPLETO - 28 CONFIGURACOES EM 5 FOLDS) ===")
    print(df_rank_annual.to_string(index=False))

    # Gera figura de Matrizes de Confusão do Ano Completo (7 Datasets na janela de 15min + Resumo Acumulado)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    axes = axes.flatten()
    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        sub = df_sum_annual[(df_sum_annual["dataset"] == ds) & (df_sum_annual["window"] == "15min")].sort_values("oof_roc_auc_hybrid", ascending=False)
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
            f"{ds} (15min - Ano Todo)\n{row['architecture']}\nAcc={acc:.1f}% | F1={f1:.3f} | Recall={rec:.1f}%",
            fontsize=10,
            fontweight="bold",
        )

    # 8º subplot: Matriz Acumulada da melhor arquitetura global no ano todo
    ax = axes[7]
    best_arch_global = df_rank_annual.iloc[0]["architecture"]
    sub_b = df_sum_annual[df_sum_annual["architecture"] == best_arch_global]
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
        f"ACUMULADO 28 CONFIGS (Ano Todo)\n{best_arch_global}\nAcc={(vn+vp)/(vn+fp+fn+vp)*100:.1f}% | Recall={vp/(vp+fn)*100:.1f}%",
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        "Matrizes de Confusão Out-of-Fold (5 Folds) — Planilha Anual Completa (StdDev 5min da Corrente em [4.0, 8.0])",
        fontsize=14,
        fontweight="bold",
    )
    p_cm_ann = os.path.join(ANOM_PLOTS_DIR, "matrizes_confusao_ano_completo_15min.png")
    plt.savefig(p_cm_ann, dpi=140)
    plt.close(fig)
    shutil.copy2(p_cm_ann, os.path.join(ART_DIR, "matrizes_confusao_ano_completo_15min.png"))


if __name__ == "__main__":
    t0 = time.time()
    df_annual_hyb, df_pump_sum, df_april_ver = build_annual_5min_current_anomaly_table()
    map_and_evaluate_annual_anomalies(df_annual_hyb)
    print(f"\nConcluído em {time.time() - t0:.1f}s")
