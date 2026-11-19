"""
Pipeline completo de Detecção de Anomalias via Autoencoders (Reconstrução 1D e 2D GAF)
com Validação Cruzada Estratificada (5 folds) para as 6 bombas separadamente e para o
Dataset Completo, nas janelas de 15min, 30min, 1h e 2h.

Arquiteturas avaliadas:
1. AE_CNN1D                 : Autoencoder Convolucional 1D (apenas sinal 1D)
2. AE_LSTM1D                : Autoencoder Recorrente Patch-LSTM 1D (apenas sinal 1D)
3. AE_Transformer1D         : Autoencoder Patch-Transformer 1D (apenas sinal 1D)
4. AE_CNN2D_GAF             : Autoencoder Convolucional 2D nas imagens GASF+GADF (32x32)
                              + cabeça de reconstrução cruzada 1D
5. Hybrid_CNN1D_CNN2D       : Fusão Multimodal (Encoder CNN-1D no sinal + Encoder CNN-2D no GAF)
6. Hybrid_LSTM1D_CNN2D      : Fusão Multimodal (Encoder LSTM-1D no sinal + Encoder CNN-2D no GAF)
7. Hybrid_Transformer_CNN2D : Fusão Multimodal (Encoder Transformer-1D no sinal + Encoder CNN-2D no GAF)
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configurações de reprodutibilidade e Dispositivo (GPU CUDA automática se disponível)
SEED = 42
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
PLOTS_DIR = os.path.join(RESULTS_DIR, "Figuras_Modelos")

INDIVIDUAL_PUMPS = [
    "Bomba_1305",
    "Bomba_1307",
    "Bomba_2307",
    "Bomba_3307",
    "Bomba_4307",
    "Bomba_5307",
]
DATASETS = INDIVIDUAL_PUMPS + ["Dataset_Completo"]
WINDOWS = ["15min", "30min", "1h", "2h"]
N_SPLITS = 5

# Hiperparâmetros enxutos para execução rápida em CPU nas 28 configurações x 5 folds x 7 modelos
MAX_TRAIN_SAMPLES_PER_FOLD = 600   # Subamostragem estratificada por fold de treino (60 por estrato)
MAX_VAL_SAMPLES_PER_FOLD = 200     # Subamostragem estratificada por fold de validação (20 por estrato)
BATCH_SIZE = 128
EPOCHS = 6
LR = 3e-3
LATENT_DIM = 32
N_PATCHES = 15  # 90, 180, 360 e 720 são todos divisíveis exatamente por 15!


def stratified_subsample_indices(
    mask: np.ndarray,
    strata: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Seleciona até `max_samples` índices dentre aqueles com `mask == True`,
    preservando rigorosamente a proporção de cada estrato (quantil de carga 0..9).
    """
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) <= max_samples:
        return np.sort(valid_indices)

    valid_strata = strata[valid_indices]
    unique_s, counts_s = np.unique(valid_strata, return_counts=True)
    proportions = counts_s / counts_s.sum()
    quotas = np.maximum(1, np.round(proportions * max_samples).astype(int))

    selected = []
    for s_val, quota in zip(unique_s, quotas):
        s_idx = valid_indices[valid_strata == s_val]
        if len(s_idx) <= quota:
            selected.append(s_idx)
        else:
            chosen = rng.choice(s_idx, size=quota, replace=False)
            selected.append(chosen)

    selected_idx = np.concatenate(selected)
    if len(selected_idx) > max_samples:
        selected_idx = rng.choice(selected_idx, size=max_samples, replace=False)
    return np.sort(selected_idx)


# ==============================================================================
# BLOCOS CODIFICADORES (ENCODERS) E DECODIFICADORES (DECODERS)
# ==============================================================================

class EncoderCNN1D(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, latent_dim),
            nn.GELU(),
        )

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        return self.net(x_1d)


class EncoderLSTM1D(nn.Module):
    """
    Patch-LSTM 1D: divide a sequência (B, C, L) em N_PATCHES=15 patches temporais
    de dimensão (C * patch_len), processando-os com uma LSTM bidirecional rápida.
    """
    def __init__(self, in_channels: int, seq_len: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.n_patches = N_PATCHES
        self.patch_len = seq_len // N_PATCHES
        patch_dim = in_channels * self.patch_len
        self.proj = nn.Linear(patch_dim, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.GELU(),
        )

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        B, C, L = x_1d.shape
        # (B, C, n_patches, patch_len) -> (B, n_patches, C * patch_len)
        x_p = x_1d.view(B, C, self.n_patches, self.patch_len).permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        h = torch.relu(self.proj(x_p))
        out, _ = self.lstm(h)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


class EncoderTransformer1D(nn.Module):
    """
    Patch-Transformer 1D: projeta N_PATCHES=15 patches temporais em d_model=32,
    adiciona codificação posicional aprendível e aplica Multi-Head Self-Attention.
    """
    def __init__(self, in_channels: int, seq_len: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.n_patches = N_PATCHES
        self.patch_len = seq_len // N_PATCHES
        patch_dim = in_channels * self.patch_len
        d_model = 32

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(d_model * self.n_patches, latent_dim),
            nn.GELU(),
        )

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        B, C, L = x_1d.shape
        x_p = x_1d.view(B, C, self.n_patches, self.patch_len).permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        tokens = self.patch_embed(x_p) + self.pos_embed
        z = self.transformer(tokens)
        return self.fc(z.reshape(B, -1))


class EncoderCNN2D(nn.Module):
    """
    Encoder CNN 2D para as matrizes GASF + GADF empilhadas (B, 2*C, 32, 32).
    """
    def __init__(self, in_channels_2d: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels_2d, 16, kernel_size=3, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),              # -> 8x8
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),                                       # -> 4x4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
            nn.GELU(),
        )

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        return self.net(x_2d)


class Decoder1D(nn.Module):
    """
    Decodificador 1D que reconstrói o sinal (B, C, L) em [-1, 1] a partir do vetor latente.
    """
    def __init__(self, out_channels: int, seq_len: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_channels * seq_len),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.out_channels, self.seq_len)


class Decoder2D(nn.Module):
    """
    Decodificador 2D que reconstrói as matrizes GASF + GADF (B, 2*C, 32, 32) em [-1, 1].
    """
    def __init__(self, out_channels_2d: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * 8 * 8),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),           # 8x8 -> 16x16
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, out_channels_2d, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 16, 8, 8)
        return self.deconv(h)


# ==============================================================================
# MODELOS UNIMODAIS (1D ou 2D) E HÍBRIDOS MULTIMODAIS (1D + 2D)
# ==============================================================================

class AutoencoderModel(nn.Module):
    def __init__(self, arch_name: str, n_channels: int, seq_len: int):
        super().__init__()
        self.arch_name = arch_name
        c_2d = n_channels  # Somente GADF (B, C, 32, 32)

        if arch_name == "AE_CNN1D":
            self.mode = "1d"
            self.enc1d = EncoderCNN1D(n_channels, seq_len, LATENT_DIM)
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)

        elif arch_name == "AE_LSTM1D":
            self.mode = "1d"
            self.enc1d = EncoderLSTM1D(n_channels, seq_len, LATENT_DIM)
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)

        elif arch_name == "AE_Transformer1D":
            self.mode = "1d"
            self.enc1d = EncoderTransformer1D(n_channels, seq_len, LATENT_DIM)
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)

        elif arch_name == "AE_CNN2D_GAF":
            self.mode = "2d"
            self.enc2d = EncoderCNN2D(c_2d, LATENT_DIM)
            self.dec2d = Decoder2D(c_2d, LATENT_DIM)
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)

        elif arch_name == "Hybrid_CNN1D_CNN2D":
            self.mode = "hybrid"
            self.enc1d = EncoderCNN1D(n_channels, seq_len, LATENT_DIM)
            self.enc2d = EncoderCNN2D(c_2d, LATENT_DIM)
            self.fusion = nn.Sequential(nn.Linear(2 * LATENT_DIM, LATENT_DIM), nn.GELU())
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)
            self.dec2d = Decoder2D(c_2d, LATENT_DIM)

        elif arch_name == "Hybrid_LSTM1D_CNN2D":
            self.mode = "hybrid"
            self.enc1d = EncoderLSTM1D(n_channels, seq_len, LATENT_DIM)
            self.enc2d = EncoderCNN2D(c_2d, LATENT_DIM)
            self.fusion = nn.Sequential(nn.Linear(2 * LATENT_DIM, LATENT_DIM), nn.GELU())
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)
            self.dec2d = Decoder2D(c_2d, LATENT_DIM)

        elif arch_name == "Hybrid_Transformer_CNN2D":
            self.mode = "hybrid"
            self.enc1d = EncoderTransformer1D(n_channels, seq_len, LATENT_DIM)
            self.enc2d = EncoderCNN2D(c_2d, LATENT_DIM)
            self.fusion = nn.Sequential(nn.Linear(2 * LATENT_DIM, LATENT_DIM), nn.GELU())
            self.dec1d = Decoder1D(n_channels, seq_len, LATENT_DIM)
            self.dec2d = Decoder2D(c_2d, LATENT_DIM)
        else:
            raise ValueError(f"Arquitetura desconhecida: {arch_name}")

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
        if self.mode == "1d":
            z = self.enc1d(x_1d)
            rec_1d = self.dec1d(z)
            return rec_1d, None
        elif self.mode == "2d":
            z = self.enc2d(x_2d)
            rec_1d = self.dec1d(z)
            rec_2d = self.dec2d(z)
            return rec_1d, rec_2d
        else:
            z1 = self.enc1d(x_1d)
            z2 = self.enc2d(x_2d)
            z = self.fusion(torch.cat([z1, z2], dim=-1))
            rec_1d = self.dec1d(z)
            rec_2d = self.dec2d(z)
            return rec_1d, rec_2d


ARCHITECTURES = [
    "AE_CNN1D",
    "AE_LSTM1D",
    "AE_Transformer1D",
    "AE_CNN2D_GAF",
    "Hybrid_CNN1D_CNN2D",
    "Hybrid_LSTM1D_CNN2D",
    "Hybrid_Transformer_CNN2D",
]


def train_and_evaluate_model(
    arch_name: str,
    X_1d_train: torch.Tensor,
    X_2d_train: torch.Tensor,
    X_1d_val: torch.Tensor,
    X_2d_val: torch.Tensor,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Treina o autoencoder por `epochs` épocas (em GPU CUDA ou CPU) e avalia no conjunto de validação.
    Retorna métricas de reconstrução 1D, 2D e detecção de anomalias (limiar P95/P99).
    """
    n_channels = X_1d_train.shape[1]
    seq_len = X_1d_train.shape[2]

    torch.manual_seed(SEED)
    model = AutoencoderModel(arch_name, n_channels, seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = TensorDataset(X_1d_train, X_2d_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    model.train()
    for _ in range(epochs):
        for b_1d, b_2d in train_loader:
            b_1d = b_1d.to(device, non_blocking=True)
            b_2d = b_2d.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                rec_1d, rec_2d = model(b_1d, b_2d)
                loss_1d = criterion(rec_1d, b_1d)
                if rec_2d is not None:
                    loss_2d = criterion(rec_2d, b_2d)
                    loss = loss_1d + 0.5 * loss_2d
                else:
                    loss = loss_1d
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Avaliação em modo inferência em lotes (evita estouro de VRAM em datasets grandes)
    model.eval()
    eval_batch = max(batch_size, 256)
    with torch.no_grad():
        tr_mse_list = []
        for i in range(0, len(X_1d_train), eval_batch):
            b1 = X_1d_train[i : i + eval_batch].to(device, non_blocking=True)
            b2 = X_2d_train[i : i + eval_batch].to(device, non_blocking=True)
            r1, _ = model(b1, b2)
            tr_mse_list.append(((r1 - b1) ** 2).mean(dim=(1, 2)).cpu().numpy())
        tr_sample_mse_1d = np.concatenate(tr_mse_list, axis=0)
        thresh_p95 = float(np.percentile(tr_sample_mse_1d, 95))
        thresh_p99 = float(np.percentile(tr_sample_mse_1d, 99))

        val_mse_1d_list, val_mae_1d_list = [], []
        val_mse_2d_list, val_mae_2d_list = [], []
        for i in range(0, len(X_1d_val), eval_batch):
            b1 = X_1d_val[i : i + eval_batch].to(device, non_blocking=True)
            b2 = X_2d_val[i : i + eval_batch].to(device, non_blocking=True)
            r1, r2 = model(b1, b2)
            d1 = r1 - b1
            val_mse_1d_list.append((d1 ** 2).mean(dim=(1, 2)).cpu().numpy())
            val_mae_1d_list.append(torch.abs(d1).mean(dim=(1, 2)).cpu().numpy())
            if r2 is not None:
                d2 = r2 - b2
                val_mse_2d_list.append((d2 ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
                val_mae_2d_list.append(torch.abs(d2).mean(dim=(1, 2, 3)).cpu().numpy())

        val_sample_mse_1d = np.concatenate(val_mse_1d_list, axis=0)
        val_sample_mae_1d = np.concatenate(val_mae_1d_list, axis=0)

        mse_1d = float(np.mean(val_sample_mse_1d))
        rmse_1d = float(np.sqrt(mse_1d))
        mae_1d = float(np.mean(val_sample_mae_1d))

        if len(val_mse_2d_list) > 0:
            mse_2d = float(np.mean(np.concatenate(val_mse_2d_list, axis=0)))
            mae_2d = float(np.mean(np.concatenate(val_mae_2d_list, axis=0)))
        else:
            mse_2d = np.nan
            mae_2d = np.nan

        pct_anom_p95 = float(100.0 * np.mean(val_sample_mse_1d > thresh_p95))
        pct_anom_p99 = float(100.0 * np.mean(val_sample_mse_1d > thresh_p99))

    return {
        "mse_1d": mse_1d,
        "rmse_1d": rmse_1d,
        "mae_1d": mae_1d,
        "mse_2d_gaf": mse_2d,
        "mae_2d_gaf": mae_2d,
        "threshold_p95": thresh_p95,
        "threshold_p99": thresh_p99,
        "val_anomaly_rate_p95_pct": pct_anom_p95,
        "val_anomaly_rate_p99_pct": pct_anom_p99,
    }


def generate_summary_plots(df_summary: pd.DataFrame):
    """
    Gera figuras comparativas de desempenho (RMSE_1D e MAE_1D) entre as 7 arquiteturas
    nas diferentes janelas e bombas.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Figura 1: Comparação geral das 7 arquiteturas por tamanho de janela (média nos datasets)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    pivot_rmse = df_summary.pivot_table(
        index="architecture",
        columns="window",
        values="rmse_1d_mean",
        aggfunc="mean",
    )[WINDOWS]
    pivot_rmse = pivot_rmse.reindex(ARCHITECTURES)

    x = np.arange(len(ARCHITECTURES))
    width = 0.18
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, w_label in enumerate(WINDOWS):
        vals = pivot_rmse[w_label].to_numpy()
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"Janela {w_label}", color=colors[i])

    ax.set_title("Comparação das 7 Arquiteturas (1D, 2D GAF e Híbridas) por Janela Temporal", fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE de Reconstrução 1D (Menor = Melhor)")
    ax.set_xticks(x)
    ax.set_xticklabels(ARCHITECTURES, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "comparacao_arquiteturas_por_janela.png"), dpi=140)
    plt.close(fig)

    # Figura 2: Heatmap Dataset x Arquitetura (RMSE médio)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    hm_data = df_summary.pivot_table(
        index="dataset",
        columns="architecture",
        values="rmse_1d_mean",
        aggfunc="mean",
    )[ARCHITECTURES].reindex(DATASETS)

    im = ax.imshow(hm_data.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(ARCHITECTURES)))
    ax.set_xticklabels(ARCHITECTURES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title("RMSE 1D Médio (5 Folds) por Bomba/Dataset e Arquitetura", fontsize=13, fontweight="bold")

    for r in range(len(DATASETS)):
        for c in range(len(ARCHITECTURES)):
            val = hm_data.iloc[r, c]
            ax.text(c, r, f"{val:.4f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="RMSE 1D")
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap_rmse_bombas_vs_modelos.png"), dpi=140)
    plt.close(fig)


def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)
    all_fold_records = []

    print("Iniciando avaliação dos 7 modelos (1D, 2D GAF, LSTM, Transformer e Híbridos) em 5 Folds...")

    for ds_name in DATASETS:
        for win_label in WINDOWS:
            t_cfg = time.time()
            if ds_name != "Dataset_Completo":
                npz_path = os.path.join(FOLDS_DIR, ds_name, f"sequencias_{ds_name}_{win_label}.npz")
                dados = np.load(npz_path)
                X_1d_all = dados["X_1d"]                          # (N, 4, L) float32
                X_gadf_all = dados["X_gadf"]                      # (N, 4, 32, 32) float16
                folds_all = dados["fold"]                         # (N,) int8
                strata_all = dados["stratum"]                     # (N,) int8
                clean_all = ~dados["is_boundary_overlap"]         # (N,) bool
                scaler_min = dados["scaler_min_per_fold"]         # (5, 4) float32
                scaler_max = dados["scaler_max_per_fold"]         # (5, 4) float32
            else:
                # Dataset_Completo construido pela concatenacao dos 6 datasets individuais (sem duplicar .npz)
                f_csv = os.path.join(FOLDS_DIR, ds_name, f"folds_{ds_name}_{win_label}.csv")
                df_meta = pd.read_csv(f_csv)
                folds_all = df_meta["fold"].to_numpy(dtype=np.int8)
                strata_all = df_meta["stratum"].to_numpy(dtype=np.int8)
                clean_all = ~df_meta["is_boundary_overlap"].to_numpy(dtype=bool)
                x1d_parts, xgadf_parts, smin_parts, smax_parts = [], [], [], []
                for p_name in INDIVIDUAL_PUMPS:
                    p_npz = np.load(os.path.join(FOLDS_DIR, p_name, f"sequencias_{p_name}_{win_label}.npz"))
                    x1d_parts.append(p_npz["X_1d"])
                    xgadf_parts.append(p_npz["X_gadf"])
                    smin_parts.append(p_npz["scaler_min_per_fold"])
                    smax_parts.append(p_npz["scaler_max_per_fold"])
                    p_npz.close()
                X_1d_all = np.concatenate(x1d_parts, axis=1)
                X_gadf_all = np.concatenate(xgadf_parts, axis=1)
                scaler_min = np.concatenate(smin_parts, axis=1)
                scaler_max = np.concatenate(smax_parts, axis=1)
                dados = None
                del x1d_parts, xgadf_parts, smin_parts, smax_parts

            for val_fold in range(N_SPLITS):
                train_mask = (folds_all != val_fold) & clean_all
                val_mask = (folds_all == val_fold) & clean_all

                tr_idx = stratified_subsample_indices(train_mask, strata_all, MAX_TRAIN_SAMPLES_PER_FOLD, rng)
                va_idx = stratified_subsample_indices(val_mask, strata_all, MAX_VAL_SAMPLES_PER_FOLD, rng)

                # Normaliza X_1d para [-1, 1] usando os limites globais do fold de treino
                c_min = scaler_min[val_fold][None, :, None]
                c_max = scaler_max[val_fold][None, :, None]
                denom = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)

                x1d_tr = np.clip(2.0 * (X_1d_all[tr_idx] - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)
                x1d_va = np.clip(2.0 * (X_1d_all[va_idx] - c_min) / denom - 1.0, -1.0, 1.0).astype(np.float32)

                # Utiliza somente GADF nos canais: (B, C, 32, 32) em float32
                x2d_tr = X_gadf_all[tr_idx].astype(np.float32)
                x2d_va = X_gadf_all[va_idx].astype(np.float32)

                t_x1d_tr = torch.from_numpy(x1d_tr)
                t_x2d_tr = torch.from_numpy(x2d_tr)
                t_x1d_va = torch.from_numpy(x1d_va)
                t_x2d_va = torch.from_numpy(x2d_va)

                for arch_name in ARCHITECTURES:
                    metrics = train_and_evaluate_model(
                        arch_name=arch_name,
                        X_1d_train=t_x1d_tr,
                        X_2d_train=t_x2d_tr,
                        X_1d_val=t_x1d_va,
                        X_2d_val=t_x2d_va,
                    )
                    all_fold_records.append({
                        "dataset": ds_name,
                        "window": win_label,
                        "fold": val_fold,
                        "architecture": arch_name,
                        "n_train": len(tr_idx),
                        "n_val": len(va_idx),
                        **metrics,
                    })

            if dados is not None:
                dados.close()
            del dados, X_1d_all, X_gadf_all
            gc.collect()

            # Resumo rápido da configuração atual no console
            df_curr = pd.DataFrame([r for r in all_fold_records if r["dataset"] == ds_name and r["window"] == win_label])
            best_row = df_curr.groupby("architecture")["rmse_1d"].mean().sort_values()
            best_arch = best_row.index[0]
            best_rmse = best_row.iloc[0]
            print(
                f"[OK] {ds_name:16s} | {win_label:5s} | 5 Folds x 7 Modelos | "
                f"Melhor: {best_arch:24s} (RMSE_1D={best_rmse:.4f}) | {time.time() - t_cfg:.1f}s"
            )

    # Salva resultados detalhados por fold (980 linhas = 7 datasets x 4 janelas x 5 folds x 7 arquiteturas)
    df_folds = pd.DataFrame(all_fold_records)
    folds_csv_path = os.path.join(RESULTS_DIR, "metricas_detalhadas_por_fold.csv")
    df_folds.to_csv(folds_csv_path, index=False)

    # Calcula média e desvio padrão entre os 5 folds para cada (dataset, window, architecture)
    df_summary = (
        df_folds.groupby(["dataset", "window", "architecture"], as_index=False)
        .agg(
            rmse_1d_mean=("rmse_1d", "mean"),
            rmse_1d_std=("rmse_1d", "std"),
            mae_1d_mean=("mae_1d", "mean"),
            mae_1d_std=("mae_1d", "std"),
            mse_1d_mean=("mse_1d", "mean"),
            mse_2d_gaf_mean=("mse_2d_gaf", "mean"),
            mae_2d_gaf_mean=("mae_2d_gaf", "mean"),
            threshold_p99_mean=("threshold_p99", "mean"),
            val_anomaly_rate_p99_pct=("val_anomaly_rate_p99_pct", "mean"),
        )
    )
    summary_csv_path = os.path.join(RESULTS_DIR, "resumo_modelos_5folds.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    # Ranking global por arquitetura
    df_ranking = (
        df_summary.groupby("architecture", as_index=False)
        .agg(
            rmse_1d_global=("rmse_1d_mean", "mean"),
            mae_1d_global=("mae_1d_mean", "mean"),
            mse_2d_gaf_global=("mse_2d_gaf_mean", "mean"),
            anomaly_rate_p99_pct=("val_anomaly_rate_p99_pct", "mean"),
        )
        .sort_values("rmse_1d_global")
    )
    ranking_csv_path = os.path.join(RESULTS_DIR, "ranking_global_arquiteturas.csv")
    df_ranking.to_csv(ranking_csv_path, index=False)

    generate_summary_plots(df_summary)

    print("\n=== RANKING GLOBAL DAS ARQUITETURAS (MÉDIA DE TODAS AS BOMBAS, JANELAS E FOLDS) ===")
    print(df_ranking.to_string(index=False))
    print(f"\nArquivos salvos em: {RESULTS_DIR}")
    print(f"Tempo total: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
