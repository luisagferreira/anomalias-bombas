"""
Script para divisão dos dados de bombas em 5 folds via Validação Cruzada Estratificada
(StratifiedGroupKFold) com janelas deslizantes (50% de sobreposição) de:
- 15 minutos (90 amostras de 10s, passo de 45 amostras)
- 30 minutos (180 amostras de 10s, passo de 90 amostras)
- 1 hora (360 amostras de 10s, passo de 180 amostras)
- 2 horas (720 amostras de 10s, passo de 360 amostras)

Executado separadamente para cada uma das 6 bombas (1305, 1307, 2307, 3307, 4307, 5307)
e para o conjunto de dados completo (todas as 6 bombas simultaneamente).
"""

import os
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import StratifiedGroupKFold

# Configurações gerais
BASE_DIR = "Dataset_completo_bombas"
FULL_CSV_PATH = os.path.join(BASE_DIR, "dataset_completo_processado.csv")
BOMBAS_DIR = os.path.join(BASE_DIR, "Bombas_por_ID")
OUTPUT_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")

N_SPLITS = 5
N_QUANTILES = 10
RANDOM_STATE = 42
BLOCK_DURATION_HOURS = 24  # Blocos temporais de 24h (8640 amostras de 10s) para agrupar janelas vizinhas

# Definição das janelas (amostragem = 10 segundos)
WINDOWS_CONFIG = {
    "15min": {"window_size": 90, "stride": 45},    # 15 min = 900s (90 * 10s), passo = 7.5 min (45 * 10s)
    "30min": {"window_size": 180, "stride": 90},   # 30 min = 1800s (180 * 10s), passo = 15 min (90 * 10s)
    "1h":    {"window_size": 360, "stride": 180},  # 1 hora = 3600s (360 * 10s), passo = 30 min (180 * 10s)
    "2h":    {"window_size": 720, "stride": 360},  # 2 horas = 7200s (720 * 10s), passo = 1 hora (360 * 10s)
}

# Mapeamento das 6 bombas e seus sufixos de colunas no dataset completo
PUMPS_MAP = {
    "Bomba_1305": "147A1305",
    "Bomba_1307": "145A1307",
    "Bomba_2307": "145A2307",
    "Bomba_3307": "145A3307",
    "Bomba_4307": "145A4307",
    "Bomba_5307": "145A5307",
}


def compute_yload_strata(yload_means: np.ndarray, n_quantiles: int = 10) -> np.ndarray:
    """
    Divide a média da carga (BR_YLOAD*) das janelas em faixas/quantis (bins) para estratificação.
    Trata janelas com carga zero (bomba desligada, <= 1e-6) como estrato 0 dedicado se houver
    muitos zeros, e divide os valores positivos nos quantis restantes.
    """
    strata = np.zeros(len(yload_means), dtype=np.int32)
    zero_mask = yload_means <= 1e-6
    pos_mask = ~zero_mask

    if np.sum(zero_mask) > 0 and np.sum(pos_mask) > 0:
        n_pos_bins = max(1, n_quantiles - 1)
        pos_bins = pd.qcut(
            yload_means[pos_mask],
            q=n_pos_bins,
            labels=False,
            duplicates="drop",
        )
        strata[pos_mask] = pos_bins.astype(np.int32) + 1
    else:
        strata = pd.qcut(
            yload_means,
            q=n_quantiles,
            labels=False,
            duplicates="drop",
        ).astype(np.int32)

    return strata


def process_dataset(
    dataset_name: str,
    timestamps: np.ndarray,
    data_matrix: np.ndarray,
    feature_cols: list[str],
    yload_indices: list[int],
    summary_records: list[dict],
):
    """
    Gera as janelas deslizantes (15m, 30m, 1h, 2h) com 50% de sobreposição,
    realiza a divisão estratificada em 5 folds agrupada por blocos temporais,
    e salva tanto o CSV de metadados/folds quanto o arquivo .npz com os tensores 3D.
    """
    dataset_out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    n_samples = data_matrix.shape[0]
    block_step_samples = BLOCK_DURATION_HOURS * 360  # 24h * 360 amostras/h = 8640 amostras

    for win_label, cfg in WINDOWS_CONFIG.items():
        t0 = time.time()
        win_size = cfg["window_size"]
        stride = cfg["stride"]

        # Índices iniciais e finais de cada janela deslizante
        start_indices = np.arange(0, n_samples - win_size + 1, stride, dtype=np.int32)
        end_indices = start_indices + win_size
        n_windows = len(start_indices)
        window_ids = np.arange(n_windows, dtype=np.int32)

        # Extração vetorizada das janelas via sliding_window_view
        # sliding_window_view na dimensão 0 retorna (n_samples - win_size + 1, n_features, win_size)
        # Transpomos para (n_windows, win_size, n_features)
        all_windows_view = sliding_window_view(data_matrix, window_shape=win_size, axis=0)
        X_windows = np.ascontiguousarray(
             np.transpose(all_windows_view[::stride], (0, 2, 1)),
            dtype=np.float32,
        )

        # Cálculo das estatísticas de BR_YLOAD* em cada janela para estratificação
        yload_windows = X_windows[:, :, yload_indices]  # (n_windows, win_size, n_yload_cols)
        yload_mean = yload_windows.mean(axis=(1, 2)).astype(np.float32)
        yload_std = yload_windows.std(axis=(1, 2)).astype(np.float32)
        yload_min = yload_windows.min(axis=(1, 2)).astype(np.float32)
        yload_max = yload_windows.max(axis=(1, 2)).astype(np.float32)

        # Médias de cada variável na janela para enriquecer o CSV
        feature_means = X_windows.mean(axis=1)  # (n_windows, n_features)

        # Estratificação por quantis da carga média (BR_YLOAD*)
        strata = compute_yload_strata(yload_mean, n_quantiles=N_QUANTILES)

        # Agrupamento em blocos temporais contíguos para evitar vazamento entre folds
        block_ids = (start_indices // block_step_samples).astype(np.int32)

        # Divisão em 5 folds com StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        folds = np.full(n_windows, -1, dtype=np.int8)
        for fold_idx, (_, val_idx) in enumerate(sgkf.split(window_ids, strata, groups=block_ids)):
            folds[val_idx] = fold_idx

        # Identifica janelas na fronteira exata entre blocos que caíram em folds diferentes
        # Como o overlap é de 50% (stride = win_size / 2), a janela w sobrepõe apenas com w-1 e w+1
        is_boundary_overlap = np.zeros(n_windows, dtype=bool)
        diff_prev = np.zeros(n_windows, dtype=bool)
        diff_next = np.zeros(n_windows, dtype=bool)
        diff_prev[1:] = folds[1:] != folds[:-1]
        diff_next[:-1] = folds[:-1] != folds[1:]
        is_boundary_overlap = diff_prev | diff_next

        # Montagem do DataFrame de janelas + folds para salvar em CSV
        df_windows = pd.DataFrame({
            "window_id": window_ids,
            "start_idx": start_indices,
            "end_idx": end_indices,
            "start_timestamp": timestamps[start_indices],
            "end_timestamp": timestamps[end_indices - 1],
            "block_id": block_ids,
            "yload_mean": np.round(yload_mean, 4),
            "yload_std": np.round(yload_std, 4),
            "yload_min": np.round(yload_min, 4),
            "yload_max": np.round(yload_max, 4),
            "stratum": strata,
            "fold": folds,
            "is_boundary_overlap": is_boundary_overlap,
        })

        # Adiciona a média de cada sensor na janela ao CSV
        for feat_idx, feat_name in enumerate(feature_cols):
            df_windows[f"mean_{feat_name}"] = np.round(feature_means[:, feat_idx], 4)

        csv_filename = f"folds_{dataset_name}_{win_label}.csv"
        csv_path = os.path.join(dataset_out_dir, csv_filename)
        df_windows.to_csv(csv_path, index=False)

        # Salva os tensores prontos para modelos sequenciais em .npz
        npz_filename = f"sequencias_{dataset_name}_{win_label}.npz"
        npz_path = os.path.join(dataset_out_dir, npz_filename)
        np.savez(
            npz_path,
            X=X_windows,
            fold=folds,
            stratum=strata.astype(np.int8),
            window_id=window_ids,
            start_idx=start_indices,
            end_idx=end_indices,
            block_id=block_ids,
            yload_mean=yload_mean,
            is_boundary_overlap=is_boundary_overlap,
            feature_names=np.array(feature_cols),
        )

        elapsed = time.time() - t0

        # Registra estatísticas no resumo
        for f_id in range(N_SPLITS):
            mask_f = folds == f_id
            summary_records.append({
                "dataset": dataset_name,
                "window": win_label,
                "window_size_steps": win_size,
                "stride_steps": stride,
                "fold": f_id,
                "n_windows_total": int(np.sum(mask_f)),
                "pct_windows": round(100.0 * np.mean(mask_f), 2),
                "n_clean_windows": int(np.sum(mask_f & (~is_boundary_overlap))),
                "yload_mean_avg": round(float(yload_mean[mask_f].mean()), 4),
                "yload_mean_std": round(float(yload_mean[mask_f].std()), 4),
                "n_strata": int(len(np.unique(strata[mask_f]))),
            })

        print(
            f"[OK] {dataset_name:16s} | Janela: {win_label:5s} "
            f"| X shape: {str(X_windows.shape):20s} | Tempo: {elapsed:.2f}s"
        )


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BOMBAS_DIR, exist_ok=True)

    print("1. Carregando dataset completo...")
    df_full = pd.read_csv(FULL_CSV_PATH)
    print(f"   Linhas originais: {len(df_full):,}")

    # Verifica e garante a existência do arquivo individual da Bomba_5307
    bomba_5307_path = os.path.join(BOMBAS_DIR, "Bomba_5307.csv")
    cols_5307 = [
        "Timestamp",
        "BR_YAR_BP145A5307",
        "BR_YL_LIT145A5307",
        "BR_YLOAD_BP145A5307",
        "BR_YP_PIT145A5307",
    ]
    if not os.path.exists(bomba_5307_path):
        print(f"2. Extraindo e salvando {bomba_5307_path}...")
        df_full[cols_5307].to_csv(bomba_5307_path, index=False)
    else:
        print(f"2. Arquivo {bomba_5307_path} já existe.")

    timestamps = df_full["Timestamp"].to_numpy()
    summary_records = []

    # 3. Processar cada uma das 6 bombas separadamente
    print("\n3. Gerando folds estratificados para cada bomba separadamente...")
    for pump_name, suffix in PUMPS_MAP.items():
        pump_cols = [
            f"BR_YAR_BP{suffix}",
            f"BR_YL_LIT{suffix}",
            f"BR_YLOAD_BP{suffix}",
            f"BR_YP_PIT{suffix}",
        ]
        data_matrix = df_full[pump_cols].to_numpy(dtype=np.float32)
        yload_indices = [2]  # Índice de BR_YLOAD_BP*
        process_dataset(
            dataset_name=pump_name,
            timestamps=timestamps,
            data_matrix=data_matrix,
            feature_cols=pump_cols,
            yload_indices=yload_indices,
            summary_records=summary_records,
        )

    # 4. Processar o conjunto de dados completo (todas as 6 bombas simultaneamente)
    print("\n4. Gerando folds estratificados para o conjunto de dados completo...")
    all_feature_cols = [c for c in df_full.columns if c != "Timestamp"]
    full_matrix = df_full[all_feature_cols].to_numpy(dtype=np.float32)
    full_yload_indices = [i for i, c in enumerate(all_feature_cols) if "YLOAD" in c]

    process_dataset(
        dataset_name="Dataset_Completo",
        timestamps=timestamps,
        data_matrix=full_matrix,
        feature_cols=all_feature_cols,
        yload_indices=full_yload_indices,
        summary_records=summary_records,
    )

    # 5. Salvar relatório consolidado da estratificação
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUTPUT_DIR, "resumo_estratificacao_folds.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n5. Resumo salvo em: {summary_path}")
    print(f"Concluído com sucesso em {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
