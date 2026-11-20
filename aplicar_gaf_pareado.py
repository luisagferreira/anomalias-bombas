"""
Script para aplicação do Gramian Angular Difference Field (somente GADF) pareado com as
séries temporais 1D originais para todas as 6 bombas separadamente, nas janelas de
15min, 30min, 1h e 2h.

Correções aplicadas para redução de volume e memória:
1. Removido completamente o cálculo e o salvamento de `X_gasf` (mantendo somente GADF).
2. Substituído `np.savez` (sem compressão) por `np.savez_compressed` (compressão ZIP).
3. Salvo `X_paa` (N, Canais, M) em float16 (junto aos scalers por fold) para permitir
   futuro cálculo on-the-fly na GPU sem vazamento entre folds.
4. Adicionadas flags de CLI:
   - `--force`       : regrava os `.npz` mesmo que já existam em disco.
   - `--on-the-fly`  : salva apenas `X_1d` + `X_paa` + scalers (sem pré-materializar `X_gadf`,
                       reduzindo o volume total de ~3,1 GB para ~0,55 GB).

Características:
- Redução espacial via PAA (Piecewise Aggregate Approximation) exata para M = 32 (32x32).
- Normalização Global por sensor em [-1, 1] calculada estritamente nos folds de treino
  (evitando vazamento de dados e preservando a escala física entre bomba ligada/desligada).
- Pareamento garantido por índice de janela (0..N-1) dentro de cada arquivo .npz:
    * X_1d   : shape (N, Canais, L)      em float32 (série temporal 1D original)
    * X_paa  : shape (N, Canais, 32)     em float16 (série reduzida via PAA)
    * X_gadf : shape (N, Canais, 32, 32) em float16 (Gramian Angular Difference Field)
    * fold, stratum, window_id, start_idx, end_idx, block_id, yload_mean, is_boundary_overlap
- Geração de figuras .png demonstrativas em Dataset_completo_bombas/Folds_Stratified_CV/Figuras_GAF/
"""

import os
import gc
import time
import zipfile
import argparse
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


def has_complete_gaf_npz(npz_path: str, require_gadf: bool = True) -> bool:
    """Verifica rapidamente via cabeçalho ZIP se o .npz já contém GADF (sem GASF) e scalers."""
    if not os.path.isfile(npz_path):
        return False
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
        required = {"X_1d.npy", "scaler_min_per_fold.npy", "scaler_max_per_fold.npy"}
        if require_gadf:
            required.add("X_gadf.npy")
        return required.issubset(names) and ("X_gasf.npy" not in names)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Aplica PAA reduzindo (N, C, L) -> (N, C, M).
    2. Normaliza cada canal globalmente para [-1, 1] usando o min e max calculados
       apenas nos folds de treino (fold != k) para cada fold de validação k.
    Retorna:
       X_paa      : (N, C, M) bruto em float32 (para normalização exata por fold on-the-fly)
       X_norm     : (N, C, M) normalizado em [-1, 1] (float32)
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

    return X_paa, X_norm, scaler_min, scaler_max


def compute_gadf_batch(
    X_norm: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Calcula SOMENTE o GADF vetorizado em lotes para economizar RAM e disco, retornando
    tensor (N, C, M, M) em float16.
    Para x em [-1, 1] e phi = arccos(x) em [0, pi]:
      cos(phi) = x
      sin(phi) = sqrt(1 - x^2)
      GADF[u, v] = sin(phi_u - phi_v) = sin(phi_u) * x_v - x_u * sin(phi_v)
    """
    N, C, M = X_norm.shape
    X_gadf = np.empty((N, C, M, M), dtype=np.float16)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = X_norm[start:end]  # (B, C, M) em float32
        s = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))  # sin(phi) >= 0

        # Produto externo no último eixo: (B, C, M, 1) * (B, C, 1, M) -> (B, C, M, M)
        sx = s[:, :, :, None] * x[:, :, None, :]
        xs = x[:, :, :, None] * s[:, :, None, :]
        X_gadf[start:end] = (sx - xs).astype(np.float16)

    return X_gadf


def compute_single_gadf(x_norm_1d: np.ndarray) -> np.ndarray:
    """Calcula a matriz GADF (M, M) para um único vetor 1D normalizado em [-1, 1]."""
    s = np.sqrt(np.clip(1.0 - x_norm_1d * x_norm_1d, 0.0, 1.0))
    return (s[:, None] * x_norm_1d[None, :] - x_norm_1d[:, None] * s[None, :]).astype(np.float32)


def plot_pump_gadf_showcase(
    dataset_name: str,
    win_label: str,
    X_1d: np.ndarray,
    X_norm: np.ndarray,
    feature_cols: list[str],
    strata: np.ndarray,
    folds: np.ndarray,
    timestamps_start: np.ndarray,
    win_idx: int,
    out_path: str,
    X_gadf: np.ndarray | None = None,
):
    """
    Gera uma figura detalhada mostrando para uma janela representativa os 4 sensores
    da bomba: Série 1D original, Série PAA normalizada em [-1, 1] e matriz GADF (32x32).
    """
    n_channels = min(4, len(feature_cols))
    L = X_1d.shape[2]
    M = X_norm.shape[2]

    fig, axes = plt.subplots(n_channels, 3, figsize=(13, 3.2 * n_channels), constrained_layout=True)
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

        # Coluna 2: GADF (32x32)
        ax2 = axes[c, 2]
        gadf_mat = (
            X_gadf[win_idx, c].astype(np.float32)
            if X_gadf is not None
            else compute_single_gadf(X_norm[win_idx, c])
        )
        im_gadf = ax2.imshow(
            gadf_mat,
            cmap="coolwarm",
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )
        ax2.set_title(f"GADF 32x32\n({ch_name})", fontsize=10)
        fig.colorbar(im_gadf, ax=ax2, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_multi_window_comparison(
    pump_name: str,
    win_samples: dict,
    out_path: str,
):
    """
    Gera figura comparando as 4 escalas temporais (15min, 30min, 1h, 2h) para a variável
    BR_YLOAD* (Carga) da bomba: Série 1D + GADF.
    """
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12), constrained_layout=True)
    fig.suptitle(
        f"Comparação Multiescala GADF (15min, 30min, 1h, 2h) - {pump_name} (Sensor Carga: BR_YLOAD)",
        fontsize=13,
        fontweight="bold",
    )

    for row_idx, (win_label, sample) in enumerate(win_samples.items()):
        x_1d = sample["x_1d"]
        gadf = sample["gadf"]
        L = len(x_1d)

        ax0 = axes[row_idx, 0]
        ax0.plot(np.arange(L), x_1d, color="#2ca02c", linewidth=1.4)
        ax0.set_title(f"Janela {win_label} - Série 1D (L={L} passos)", fontsize=10)
        ax0.set_xlabel("Passo (10s)")
        ax0.set_ylabel("Carga (YLOAD)")
        ax0.grid(True, alpha=0.3)

        ax1 = axes[row_idx, 1]
        im1 = ax1.imshow(gadf, cmap="coolwarm", origin="lower", vmin=-1.0, vmax=1.0)
        ax1.set_title(f"GADF 32x32 ({win_label})", fontsize=10)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def process_dataset_gaf(
    dataset_name: str,
    timestamps: np.ndarray,
    data_matrix: np.ndarray,
    feature_cols: list[str],
    force: bool = False,
    on_the_fly: bool = False,
):
    dataset_out_dir = os.path.join(OUTPUT_DIR, dataset_name)
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

        if (not force) and has_complete_gaf_npz(npz_path, require_gadf=(not on_the_fly)):
            n_w = len(df_meta)
            n_c = len(feature_cols)
            sz_mb = os.path.getsize(npz_path) / 1e6
            print(
                f"[JÁ GERADO] {dataset_name:16s} | {win_label:5s} | "
                f"X_1d: ({n_w}, {n_c}, {win_size}) | GADF: ({n_w}, {n_c}, 32, 32) | {sz_mb:.1f} MB"
            )
            start_i = int(df_meta["start_idx"].iloc[dynamic_idx])
            x_1d_sample = data_matrix[start_i : start_i + win_size, yload_ch].copy()
            W_paa = build_paa_weight_matrix(win_size, GAF_SIZE)
            x_paa_s = x_1d_sample @ W_paa
            c_min = float(data_matrix[:, yload_ch].min())
            c_max = float(data_matrix[:, yload_ch].max())
            x_n = np.clip(2.0 * (x_paa_s - c_min) / max(c_max - c_min, 1e-8) - 1.0, -1.0, 1.0)
            win_samples_for_plot[win_label] = {
                "x_1d": x_1d_sample,
                "gadf": compute_single_gadf(x_n),
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
        X_paa, X_norm, scaler_min, scaler_max = compute_global_fold_normalized_paa(
            X_1d=X_1d,
            folds=folds,
            clean_mask=clean_mask,
            output_size=GAF_SIZE,
        )

        # Calcula SOMENTE GADF (sem GASF) quando não estiver no modo --on-the-fly
        X_gadf = None if on_the_fly else compute_gadf_batch(X_norm, batch_size=2048)

        if os.path.exists(npz_path):
            os.remove(npz_path)

        save_payload = dict(
            X_1d=X_1d,
            X_paa=X_paa.astype(np.float16),
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
        if X_gadf is not None:
            save_payload["X_gadf"] = X_gadf

        # Compressão ZIP nativa (reduz ~50% em X_gadf e ~76% em X_1d)
        np.savez_compressed(npz_path, **save_payload)

        if win_label == "15min" or (dataset_name == "Bomba_1305" and win_label == "1h"):
            fig_path = os.path.join(FIGURES_DIR, f"gaf_canais_{dataset_name}_{win_label}.png")
            plot_pump_gadf_showcase(
                dataset_name=dataset_name,
                win_label=win_label,
                X_1d=X_1d,
                X_norm=X_norm,
                feature_cols=feature_cols,
                strata=strata,
                folds=folds,
                timestamps_start=timestamps[start_indices],
                win_idx=dynamic_idx,
                out_path=fig_path,
                X_gadf=X_gadf,
            )

        # Guarda amostra do canal YLOAD para a figura multiescala (15m, 30m, 1h, 2h)
        gadf_dyn = (
            X_gadf[dynamic_idx, yload_ch].astype(np.float32)
            if X_gadf is not None
            else compute_single_gadf(X_norm[dynamic_idx, yload_ch])
        )
        win_samples_for_plot[win_label] = {
            "x_1d": X_1d[dynamic_idx, yload_ch].copy(),
            "gadf": gadf_dyn,
        }

        elapsed = time.time() - t0
        sz_mb = os.path.getsize(npz_path) / 1e6
        gadf_info = str(X_gadf.shape) if X_gadf is not None else "on-the-fly (X_paa)"
        print(
            f"[OK] {dataset_name:16s} | {win_label:5s} | "
            f"X_1d: {str(X_1d.shape):18s} | GADF: {gadf_info:20s} | {sz_mb:6.1f} MB | {elapsed:.2f}s"
        )
        del X_1d, X_paa, X_norm, X_gadf, save_payload
        gc.collect()

    # Salva figura comparativa multiescala (15min, 30min, 1h, 2h)
    multiscale_fig_path = os.path.join(FIGURES_DIR, f"gaf_multiescala_{dataset_name}.png")
    plot_multi_window_comparison(dataset_name, win_samples_for_plot, multiscale_fig_path)


def parse_args():
    p = argparse.ArgumentParser(description="Gera tensores pareados 1D + GADF (sem GASF) comprimidos em .npz")
    p.add_argument("--force", action="store_true", help="Regrava os arquivos .npz mesmo se já existirem")
    p.add_argument(
        "--on-the-fly",
        action="store_true",
        help="Salva apenas X_1d + X_paa + scalers (sem pré-gravar X_gadf no disco; ~0,55 GB total)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("1. Carregando dataset completo...")
    df_full = pd.read_csv(FULL_CSV_PATH)
    timestamps = df_full["Timestamp"].to_numpy()

    modo = "ON-THE-FLY (X_1d + X_paa)" if args.on_the_fly else "COMPRIMIDO (X_1d + X_paa + GADF 32x32, sem GASF)"
    print(f"\n2. Verificando/Gerando tensores pareados [{modo}] para cada bomba...")
    for pump_name, suffix in PUMPS_MAP.items():
        expected_npz = [
            os.path.join(OUTPUT_DIR, pump_name, f"sequencias_{pump_name}_{w}.npz")
            for w in WINDOWS_CONFIG
        ]
        if (not args.force) and all(has_complete_gaf_npz(p, require_gadf=(not args.on_the_fly)) for p in expected_npz):
            for w, cfg in WINDOWS_CONFIG.items():
                p_path = os.path.join(OUTPUT_DIR, pump_name, f"sequencias_{pump_name}_{w}.npz")
                n_w = (len(df_full) - cfg["window_size"]) // cfg["stride"] + 1
                sz_mb = os.path.getsize(p_path) / 1e6
                print(
                    f"[JÁ GERADO] {pump_name:16s} | {w:5s} | "
                    f"X_1d: ({n_w}, 4, {cfg['window_size']}) | GADF: ({n_w}, 4, 32, 32) | {sz_mb:6.1f} MB"
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
            force=args.force,
            on_the_fly=args.on_the_fly,
        )

    del df_full
    gc.collect()
    print("\n3. Dataset_Completo é construído dinamicamente pela concatenação das 6 bombas (sem duplicar .npz em disco).")
    print(f"\n4. Figuras demonstrativas salvas em: {FIGURES_DIR}")
    print(f"Concluído com sucesso em {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
