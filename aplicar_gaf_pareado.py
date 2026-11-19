"""
Script para aplicação do Gramian Angular Field (GASF e GADF) pareado com as séries
temporais 1D originais para todas as 6 bombas separadamente e para o Dataset Completo,
nas janelas de 15min, 30min, 1h e 2h.

Características:
- Redução espacial via PAA (Piecewise Aggregate Approximation) exata para M = 32 (32x32).
- Normalização Global por sensor em [-1, 1] calculada estritamente nos folds de treino
  (evitando vazamento de dados e preservando a escala física entre bomba ligada/desligada).
- Pareamento garantido por índice de janela (0..N-1) dentro de cada arquivo .npz:
    * X_1d   : shape (N, Canais, L)      em float32 (série temporal 1D original)
    * X_gasf : shape (N, Canais, 32, 32) em float16 (Gramian Angular Summation Field)
    * X_gadf : shape (N, Canais, 32, 32) em float16 (Gramian Angular Difference Field)
    * fold, stratum, window_id, start_idx, end_idx, block_id, yload_mean, is_boundary_overlap
- Geração de figuras .png demonstrativas em Dataset_completo_bombas/Folds_Stratified_CV/Figuras_GAF/
"""

import os
import gc
import time
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# Configurações gerais
BASE_DIR = "Dataset_completo_bombas"
FULL_CSV_PATH = os.path.join(BASE_DIR, "dataset_completo_processado.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Folds_Stratified_CV")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "Figuras_GAF")

GAF_SIZE = 32  # Resolução espacial MxM via PAA (32x32)
N_SPLITS = 5

WINDOWS_CONFIG = {
    "15min": {"window_size": 90, "stride": 45},
    "30min": {"window_size": 180, "stride": 90},
    "1h":    {"window_size": 360, "stride": 180},
    "2h":    {"window_size": 720, "stride": 360},
}

PUMPS_MAP = {
    "Bomba_1305": "147A1305",
    "Bomba_1307": "145A1307",
    "Bomba_2307": "145A2307",
    "Bomba_3307": "145A3307",
    "Bomba_4307": "145A4307",
    "Bomba_5307": "145A5307",
}


def has_complete_gaf_npz(npz_path: str) -> bool:
    """Verifica rapidamente via cabeçalho ZIP se o .npz já contém GADF e scalers."""
    if not os.path.isfile(npz_path):
        return False
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
        return {"X_1d.npy", "X_gadf.npy", "scaler_min_per_fold.npy", "scaler_max_per_fold.npy"}.issubset(names)
    except Exception:
        return False


def build_paa_weight_matrix(input_length: int, output_size: int = 32) -> np.ndarray:
    """
    Constrói a matriz de transformação linear exata (L, M) para Piecewise Aggregate
    Approximation (PAA), lidando inclusive com casos em que L não é múltiplo inteiro de M.
    Ao multiplicar X_1d (N, C, L) @ W (L, M), obtém-se X_paa (N, C, M).
    """
    W = np.zeros((input_length, output_size), dtype=np.float32)
    segment_len = input_length / float(output_size)

    for m in range(output_size):
        start_pos = m * segment_len
        end_pos = (m + 1) * segment_len

        for t in range(input_length):
            overlap_start = max(start_pos, float(t))
            overlap_end = min(end_pos, float(t + 1))
            if overlap_end > overlap_start:
                W[t, m] = (overlap_end - overlap_start) / segment_len

    return W


def compute_global_fold_normalized_paa(
    X_1d: np.ndarray,
    folds: np.ndarray,
    clean_mask: np.ndarray,
    output_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Aplica PAA reduzindo (N, C, L) -> (N, C, M).
    2. Normaliza cada canal globalmente para [-1, 1] usando o min e max calculados
       apenas nos folds de treino (fold != k) para cada fold de validação k.
    Retorna:
       X_norm : (N, C, M) em [-1, 1] (float32)
       scaler_min : (5, C) min por fold e canal
       scaler_max : (5, C) max por fold e canal
    """
    N, C, L = X_1d.shape
    W_paa = build_paa_weight_matrix(L, output_size)
    X_paa = np.matmul(X_1d, W_paa)  # (N, C, M)

    # Min e Max por janela em cada canal sobre a série 1D completa para referência global
    win_min = X_1d.min(axis=2)  # (N, C)
    win_max = X_1d.max(axis=2)  # (N, C)

    scaler_min = np.zeros((N_SPLITS, C), dtype=np.float32)
    scaler_max = np.zeros((N_SPLITS, C), dtype=np.float32)
    X_norm = np.empty_like(X_paa, dtype=np.float32)

    for k in range(N_SPLITS):
        train_mask = (folds != k) & clean_mask
        val_mask = folds == k

        c_min = win_min[train_mask].min(axis=0)  # (C,)
        c_max = win_max[train_mask].max(axis=0)  # (C,)
        denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

        scaler_min[k] = c_min
        scaler_max[k] = c_max

        # Escala para [-1, 1] nas janelas pertencentes ao fold k
        scaled = 2.0 * (X_paa[val_mask] - c_min[None, :, None]) / denom[None, :, None] - 1.0
        X_norm[val_mask] = np.clip(scaled, -1.0, 1.0)

    return X_norm, scaler_min, scaler_max


def compute_gasf_gadf_batch(
    X_norm: np.ndarray,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula GASF e GADF vetorizados em lotes para economizar RAM, retornando
    tensores (N, C, M, M) em float16.
    Para x em [-1, 1] e phi = arccos(x) em [0, pi]:
      cos(phi) = x
      sin(phi) = sqrt(1 - x^2)
      GASF[u, v] = cos(phi_u + phi_v) = x_u * x_v - sin(phi_u) * sin(phi_v)
      GADF[u, v] = sin(phi_u - phi_v) = sin(phi_u) * x_v - x_u * sin(phi_v)
    """
    N, C, M = X_norm.shape
    X_gasf = np.empty((N, C, M, M), dtype=np.float16)
    X_gadf = np.empty((N, C, M, M), dtype=np.float16)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = X_norm[start:end]  # (B, C, M) em float32
        s = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))  # sin(phi) >= 0

        # Produto externo no último eixo: (B, C, M, 1) * (B, C, 1, M) -> (B, C, M, M)
        xx = x[:, :, :, None] * x[:, :, None, :]
        ss = s[:, :, :, None] * s[:, :, None, :]
        sx = s[:, :, :, None] * x[:, :, None, :]
        xs = x[:, :, :, None] * s[:, :, None, :]

        X_gasf[start:end] = (xx - ss).astype(np.float16)
        X_gadf[start:end] = (sx - xs).astype(np.float16)

    return X_gasf, X_gadf


def plot_pump_gaf_showcase(
    dataset_name: str,
    win_label: str,
    X_1d: np.ndarray,
    X_norm: np.ndarray,
    X_gasf: np.ndarray,
    X_gadf: np.ndarray,
    feature_cols: list[str],
    strata: np.ndarray,
    folds: np.ndarray,
    timestamps_start: np.ndarray,
    win_idx: int,
    out_path: str,
):
    """
    Gera uma figura detalhada mostrando para uma janela representativa os 4 sensores
    da bomba: Série 1D original, Série PAA normalizada em [-1, 1], matriz GASF (32x32)
    e matriz GADF (32x32).
    """
    n_channels = min(4, len(feature_cols))
    L = X_1d.shape[2]
    M = X_norm.shape[2]

    fig, axes = plt.subplots(n_channels, 4, figsize=(16, 3.2 * n_channels), constrained_layout=True)
    if n_channels == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"{dataset_name} | Janela: {win_label} (L={L} -> PAA={M}x{M}) | "
        f"Window ID: {win_idx} ({timestamps_start[win_idx]}) | "
        f"Estrato: {strata[win_idx]} | Fold: {folds[win_idx]}",
        fontsize=13,
        fontweight="bold",
    )

    t_1d = np.arange(L)
    t_paa = np.linspace(0, L - 1, M)

    for c in range(n_channels):
        ch_name = feature_cols[c]

        # Coluna 0: Série Temporal 1D original (comprimento L)
        ax0 = axes[c, 0]
        ax0.plot(t_1d, X_1d[win_idx, c], color="#1f77b4", linewidth=1.5)
        ax0.set_title(f"{ch_name}\nSérie 1D Original (L={L})", fontsize=10)
        ax0.set_xlabel("Passo (10s)")
        ax0.set_ylabel("Valor Físico")
        ax0.grid(True, alpha=0.3)

        # Coluna 1: Série PAA Normalizada Globalmente em [-1, 1] (M=32)
        ax1 = axes[c, 1]
        ax1.plot(t_paa, X_norm[win_idx, c], color="#ff7f0e", marker="o", markersize=3, linewidth=1.2)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_title(f"{ch_name}\nPAA Normalizado [-1, 1] (M={M})", fontsize=10)
        ax1.set_xlabel("Segmento PAA")
        ax1.set_ylabel("Escala [-1, 1]")
        ax1.grid(True, alpha=0.3)

        # Coluna 2: GASF (32x32)
        ax2 = axes[c, 2]
        im_gasf = ax2.imshow(
            X_gasf[win_idx, c].astype(np.float32),
            cmap="rainbow",
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )
        ax2.set_title(f"GASF 32x32\n({ch_name})", fontsize=10)
        fig.colorbar(im_gasf, ax=ax2, fraction=0.046, pad=0.04)

        # Coluna 3: GADF (32x32)
        ax3 = axes[c, 3]
        im_gadf = ax3.imshow(
            X_gadf[win_idx, c].astype(np.float32),
            cmap="coolwarm",
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )
        ax3.set_title(f"GADF 32x32\n({ch_name})", fontsize=10)
        fig.colorbar(im_gadf, ax=ax3, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_multi_window_comparison(
    pump_name: str,
    win_samples: dict,
    out_path: str,
):
    """
    Gera figura comparando as 4 escalas temporais (15min, 30min, 1h, 2h) para a variável
    BR_YLOAD* (Carga) da bomba: Série 1D + GASF + GADF.
    """
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(
        f"Comparação Multiescala GAF (15min, 30min, 1h, 2h) - {pump_name} (Sensor Carga: BR_YLOAD)",
        fontsize=13,
        fontweight="bold",
    )

    for row_idx, (win_label, sample) in enumerate(win_samples.items()):
        x_1d = sample["x_1d"]
        gasf = sample["gasf"]
        gadf = sample["gadf"]
        L = len(x_1d)

        ax0 = axes[row_idx, 0]
        ax0.plot(np.arange(L), x_1d, color="#2ca02c", linewidth=1.4)
        ax0.set_title(f"Janela {win_label} - Série 1D (L={L} passos)", fontsize=10)
        ax0.set_xlabel("Passo (10s)")
        ax0.set_ylabel("Carga (YLOAD)")
        ax0.grid(True, alpha=0.3)

        ax1 = axes[row_idx, 1]
        im1 = ax1.imshow(gasf, cmap="rainbow", origin="lower", vmin=-1.0, vmax=1.0)
        ax1.set_title(f"GASF 32x32 ({win_label})", fontsize=10)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[row_idx, 2]
        im2 = ax2.imshow(gadf, cmap="coolwarm", origin="lower", vmin=-1.0, vmax=1.0)
        ax2.set_title(f"GADF 32x32 ({win_label})", fontsize=10)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def process_dataset_gaf(
    dataset_name: str,
    timestamps: np.ndarray,
    data_matrix: np.ndarray,
    feature_cols: list[str],
):
    dataset_out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    n_samples = data_matrix.shape[0]
    win_samples_for_plot = {}

    for win_label, cfg in WINDOWS_CONFIG.items():
        t0 = time.time()
        win_size = cfg["window_size"]
        stride = cfg["stride"]

        npz_path = os.path.join(dataset_out_dir, f"sequencias_{dataset_name}_{win_label}.npz")
        csv_path = os.path.join(dataset_out_dir, f"folds_{dataset_name}_{win_label}.csv")
        df_meta = pd.read_csv(csv_path)
        dynamic_idx = int(np.argmax(df_meta["yload_std"].to_numpy()))
        yload_ch = 2 if dataset_name != "Dataset_Completo" else 12

        if has_complete_gaf_npz(npz_path):
            n_w = len(df_meta)
            n_c = len(feature_cols)
            print(
                f"[JÁ GERADO] {dataset_name:16s} | {win_label:5s} | "
                f"X_1d: ({n_w}, {n_c}, {win_size}) | GASF/GADF: ({n_w}, {n_c}, 32, 32)"
            )
            # Carrega apenas a amostra dynamic_idx para o gráfico multiescala
            start_i = int(df_meta["start_idx"].iloc[dynamic_idx])
            x_1d_sample = data_matrix[start_i : start_i + win_size, yload_ch].copy()
            W_paa = build_paa_weight_matrix(win_size, GAF_SIZE)
            x_paa_s = x_1d_sample @ W_paa
            c_min = float(data_matrix[:, yload_ch].min())
            c_max = float(data_matrix[:, yload_ch].max())
            x_n = np.clip(2.0 * (x_paa_s - c_min) / max(c_max - c_min, 1e-8) - 1.0, -1.0, 1.0)
            s_n = np.sqrt(np.clip(1.0 - x_n * x_n, 0.0, 1.0))
            win_samples_for_plot[win_label] = {
                "x_1d": x_1d_sample,
                "gasf": (x_n[:, None] * x_n[None, :] - s_n[:, None] * s_n[None, :]).astype(np.float32),
                "gadf": (s_n[:, None] * x_n[None, :] - x_n[:, None] * s_n[None, :]).astype(np.float32),
            }
            continue

        folds = df_meta["fold"].to_numpy(dtype=np.int8)
        strata = df_meta["stratum"].to_numpy(dtype=np.int8)
        window_ids = df_meta["window_id"].to_numpy(dtype=np.int32)
        start_indices = df_meta["start_idx"].to_numpy(dtype=np.int32)
        end_indices = df_meta["end_idx"].to_numpy(dtype=np.int32)
        block_ids = df_meta["block_id"].to_numpy(dtype=np.int32)
        yload_mean = df_meta["yload_mean"].to_numpy(dtype=np.float32)
        is_boundary_overlap = df_meta["is_boundary_overlap"].to_numpy(dtype=bool)
        clean_mask = ~is_boundary_overlap

        # Constrói o tensor 1D no formato (N, Canais, L)
        all_windows_view = sliding_window_view(data_matrix, window_shape=win_size, axis=0)
        X_1d = np.ascontiguousarray(all_windows_view[::stride], dtype=np.float32)

        # Redução PAA (L -> 32) + Normalização Global por sensor nos folds de treino
        X_norm, scaler_min, scaler_max = compute_global_fold_normalized_paa(
            X_1d=X_1d,
            folds=folds,
            clean_mask=clean_mask,
            output_size=GAF_SIZE,
        )

        # Cálculo vetorizado de GASF e GADF (N, C, 32, 32) em float16
        X_gasf, X_gadf = compute_gasf_gadf_batch(X_norm, batch_size=512)

        if os.path.exists(npz_path):
            os.remove(npz_path)

        np.savez(
            npz_path,
            X_1d=X_1d,
            X_gasf=X_gasf,
            X_gadf=X_gadf,
            fold=folds,
            stratum=strata,
            window_id=window_ids,
            start_idx=start_indices,
            end_idx=end_indices,
            block_id=block_ids,
            yload_mean=yload_mean,
            is_boundary_overlap=is_boundary_overlap,
            feature_names=np.array(feature_cols),
            scaler_min_per_fold=scaler_min,
            scaler_max_per_fold=scaler_max,
        )

        if win_label == "15min" or (dataset_name == "Bomba_1305" and win_label == "1h"):
            fig_path = os.path.join(FIGURES_DIR, f"gaf_canais_{dataset_name}_{win_label}.png")
            plot_pump_gaf_showcase(
                dataset_name=dataset_name,
                win_label=win_label,
                X_1d=X_1d,
                X_norm=X_norm,
                X_gasf=X_gasf,
                X_gadf=X_gadf,
                feature_cols=feature_cols,
                strata=strata,
                folds=folds,
                timestamps_start=timestamps[start_indices],
                win_idx=dynamic_idx,
                out_path=fig_path,
            )

        # Guarda amostra do canal YLOAD para a figura multiescala (15m, 30m, 1h, 2h)
        win_samples_for_plot[win_label] = {
            "x_1d": X_1d[dynamic_idx, yload_ch].copy(),
            "gasf": X_gasf[dynamic_idx, yload_ch].astype(np.float32),
            "gadf": X_gadf[dynamic_idx, yload_ch].astype(np.float32),
        }

        elapsed = time.time() - t0
        print(
            f"[OK] {dataset_name:16s} | {win_label:5s} | "
            f"X_1d: {str(X_1d.shape):18s} | GASF/GADF: {str(X_gasf.shape):20s} | {elapsed:.2f}s"
        )
        del X_1d, X_norm, X_gasf, X_gadf
        gc.collect()

    # Salva figura comparativa multiescala (15min, 30min, 1h, 2h)
    multiscale_fig_path = os.path.join(FIGURES_DIR, f"gaf_multiescala_{dataset_name}.png")
    plot_multi_window_comparison(dataset_name, win_samples_for_plot, multiscale_fig_path)


def main():
    t_start = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("1. Carregando dataset completo...")
    df_full = pd.read_csv(FULL_CSV_PATH)
    timestamps = df_full["Timestamp"].to_numpy()

    print("\n2. Verificando/Gerando tensores pareados (1D + GASF 32x32 + GADF 32x32) para cada bomba...")
    for pump_name, suffix in PUMPS_MAP.items():
        expected_npz = [
            os.path.join(OUTPUT_DIR, pump_name, f"sequencias_{pump_name}_{w}.npz")
            for w in WINDOWS_CONFIG
        ]
        if all(has_complete_gaf_npz(p) for p in expected_npz):
            for w, cfg in WINDOWS_CONFIG.items():
                n_w = (len(df_full) - cfg["window_size"]) // cfg["stride"] + 1
                print(
                    f"[JÁ GERADO] {pump_name:16s} | {w:5s} | "
                    f"X_1d: ({n_w}, 4, {cfg['window_size']}) | GASF/GADF: ({n_w}, 4, 32, 32)"
                )
            continue

        pump_cols = [
            f"BR_YAR_BP{suffix}",
            f"BR_YL_LIT{suffix}",
            f"BR_YLOAD_BP{suffix}",
            f"BR_YP_PIT{suffix}",
        ]
        data_matrix = df_full[pump_cols].to_numpy(dtype=np.float32)
        process_dataset_gaf(
            dataset_name=pump_name,
            timestamps=timestamps,
            data_matrix=data_matrix,
            feature_cols=pump_cols,
        )

    del df_full
    gc.collect()
    print("\n3. Dataset_Completo agora é construído dinamicamente pela concatenação das 6 bombas (sem duplicar .npz em disco).")

    print(f"\n4. Figuras demonstrativas salvas em: {FIGURES_DIR}")
    print(f"Concluído com sucesso em {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
