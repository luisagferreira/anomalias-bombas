"""
Treina as 7 arquiteturas de Autoencoders na GPU (5 Folds via StratifiedGroupKFold com
exclusão de fronteira `clean_mask`) registrando época a época (1 a 25) as curvas de
Perda (Loss) de Treinamento e de Validação:
  1. Loss de Reconstrução Não Supervisionada (MSE nas janelas normais: 1D + 0.5 * 2D GADF)
  2. Loss Total Híbrida (Reconstrução + BCE Calibrado)

Gera:
  - `historico_perda_loss_epocas_5folds.csv`
  - `curvas_perda_loss_7_arquiteturas.png` (Treino vs. Validação ± 1 DP para as 7 arquiteturas)
  - `curvas_perda_loss_por_bomba.png` (Convergência Treino vs. Validação para cada bomba e Dataset Completo)
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from treinar_modelos_1d_2d_gaf import (
    AutoencoderModel,
    ARCHITECTURES,
    DATASETS,
    SEED,
    LR,
    LATENT_DIM,
)
from avaliar_deteccao_anomalias_operador import (
    extract_latent_and_reconstructions,
    compute_auxiliary_signal_features,
)
from avaliar_histograma_erro_operador_gpu import (
    compute_clean_group_folds,
    load_dataset_tensors,
    N_SPLITS,
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

ARCH_COLORS = {
    "AE_CNN1D": "#0284C7",
    "AE_LSTM1D": "#2563EB",
    "AE_Transformer1D": "#7C3AED",
    "AE_CNN2D_GAF": "#D97706",
    "Hybrid_CNN1D_CNN2D": "#059669",
    "Hybrid_LSTM1D_CNN2D": "#DC2626",
    "Hybrid_Transformer_CNN2D": "#0F172A",
}


def parse_args():
    has_cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description="Geração das Curvas de Perda (Loss) Treino vs Validação em GPU")
    p.add_argument("--device", type=str, default="cuda" if has_cuda else "cpu")
    p.add_argument("--window", type=str, default="15min")
    p.add_argument("--epochs", type=int, default=25 if has_cuda else 10)
    p.add_argument("--batch-size", type=int, default=512 if has_cuda else 128)
    p.add_argument("--max-samples", type=int, default=6000 if has_cuda else 400)
    return p.parse_args()


def track_loss_5folds(
    arch_name: str,
    x1d_all: np.ndarray,
    x2d_all: np.ndarray,
    y_all: np.ndarray,
    folds_cv: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> list[dict]:
    """Treina nos 5 folds e registra train_rec_loss, val_rec_loss, train_tot_loss, val_tot_loss a cada época."""
    n_samples, n_channels, seq_len = x1d_all.shape
    c_2d = x2d_all.shape[1]
    aux_dim = 5 * n_channels + c_2d
    use_amp = (device.type == "cuda")
    unique_folds = np.unique(folds_cv)
    records = []

    for val_fold in unique_folds:
        tr_idx = np.flatnonzero(folds_cv != val_fold)
        va_idx = np.flatnonzero(folds_cv == val_fold)

        t_x1d_tr = torch.from_numpy(x1d_all[tr_idx]).to(device, non_blocking=True)
        t_x2d_tr = torch.from_numpy(x2d_all[tr_idx]).to(device, non_blocking=True)
        t_y_tr = torch.from_numpy(y_all[tr_idx].astype(np.float32)).unsqueeze(1).to(device, non_blocking=True)
        t_x1d_va = torch.from_numpy(x1d_all[va_idx]).to(device, non_blocking=True)
        t_x2d_va = torch.from_numpy(x2d_all[va_idx]).to(device, non_blocking=True)
        t_y_va = torch.from_numpy(y_all[va_idx].astype(np.float32)).unsqueeze(1).to(device, non_blocking=True)

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.15)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_pos_tr = float(np.sum(y_all[tr_idx] == 1))
        n_neg_tr = float(np.sum(y_all[tr_idx] == 0))
        pos_weight = torch.tensor([min(50.0, max(1.0, n_neg_tr / max(1.0, n_pos_tr)))], dtype=torch.float32, device=device)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_tr = len(tr_idx)
        n_va = len(va_idx)

        for ep in range(1, epochs + 1):
            model.train()
            clf_head.train()
            perm = torch.randperm(n_tr, device=device)

            ep_tr_rec_sum = 0.0
            ep_tr_rec_cnt = 0
            ep_tr_tot_sum = 0.0
            ep_tr_tot_cnt = 0

            for start_b in range(0, n_tr, batch_size):
                b_idx = perm[start_b : start_b + batch_size]
                if len(b_idx) <= 1:
                    continue
                b1 = t_x1d_tr[b_idx].float()
                b2 = t_x2d_tr[b_idx].float()
                by = t_y_tr[b_idx]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b1, b2)
                    norm_b = (by.squeeze(1) < 0.5)
                    n_norm_b = int(norm_b.sum().item())
                    if n_norm_b > 0:
                        loss_rec_1d = ((rec_1d[norm_b] - b1[norm_b]) ** 2).mean()
                        if rec_2d is not None:
                            loss_rec_2d = ((rec_2d[norm_b] - b2[norm_b]) ** 2).mean()
                            loss_rec = loss_rec_1d + 0.5 * loss_rec_2d
                        else:
                            loss_rec = loss_rec_1d
                        ep_tr_rec_sum += float(loss_rec.item()) * n_norm_b
                        ep_tr_rec_cnt += n_norm_b
                    else:
                        loss_rec = torch.tensor(0.0, device=device)

                    aux_f = compute_auxiliary_signal_features(b1, rec_1d.detach(), rec_2d.detach() if rec_2d is not None else None, b2)
                    logits = clf_head(torch.cat([z, aux_f], dim=-1))
                    loss_cls = bce_loss_fn(logits, by)
                    loss_tot = loss_rec + 0.25 * loss_cls

                scaler.scale(loss_tot).backward()
                scaler.step(optimizer)
                scaler.update()

                ep_tr_tot_sum += float(loss_tot.item()) * len(b_idx)
                ep_tr_tot_cnt += len(b_idx)

            scheduler.step()

            # Avaliação da Loss no Fold de Validação ao fim da época
            model.eval()
            clf_head.eval()
            ep_va_rec_sum = 0.0
            ep_va_rec_cnt = 0
            ep_va_tot_sum = 0.0
            ep_va_tot_cnt = 0

            with torch.no_grad():
                for start_b in range(0, n_va, batch_size):
                    end_b = min(n_va, start_b + batch_size)
                    b1 = t_x1d_va[start_b:end_b].float()
                    b2 = t_x2d_va[start_b:end_b].float()
                    by = t_y_va[start_b:end_b]
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        z, rec_1d, rec_2d = extract_latent_and_reconstructions(model, b1, b2)
                        norm_b = (by.squeeze(1) < 0.5)
                        n_norm_b = int(norm_b.sum().item())
                        if n_norm_b > 0:
                            l1 = ((rec_1d[norm_b] - b1[norm_b]) ** 2).mean()
                            if rec_2d is not None:
                                l2 = ((rec_2d[norm_b] - b2[norm_b]) ** 2).mean()
                                l_rec = l1 + 0.5 * l2
                            else:
                                l_rec = l1
                            ep_va_rec_sum += float(l_rec.item()) * n_norm_b
                            ep_va_rec_cnt += n_norm_b
                        else:
                            l_rec = torch.tensor(0.0, device=device)

                        aux_f = compute_auxiliary_signal_features(b1, rec_1d, rec_2d, b2)
                        logits = clf_head(torch.cat([z, aux_f], dim=-1))
                        l_cls = bce_loss_fn(logits, by)
                        l_tot = l_rec + 0.25 * l_cls

                    ep_va_tot_sum += float(l_tot.item()) * (end_b - start_b)
                    ep_va_tot_cnt += (end_b - start_b)

            records.append({
                "architecture": arch_name,
                "fold": int(val_fold),
                "epoch": ep,
                "train_rec_loss": ep_tr_rec_sum / max(1, ep_tr_rec_cnt),
                "val_rec_loss": ep_va_rec_sum / max(1, ep_va_rec_cnt),
                "train_total_loss": ep_tr_tot_sum / max(1, ep_tr_tot_cnt),
                "val_total_loss": ep_va_tot_sum / max(1, ep_va_tot_cnt),
            })

        del t_x1d_tr, t_x2d_tr, t_y_tr, t_x1d_va, t_x2d_va, t_y_va, model, clf_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return records


def plot_loss_by_architecture(df_loss: pd.DataFrame, out_path: str):
    """Plota um painel 2x4 com as curvas de Perda (Treino vs Validação ± 1 DP) para as 7 arquiteturas + comparativo."""
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 9.2), dpi=300, facecolor="white")
    axes = axes.flatten()

    fig.suptitle(
        "Curvas de Perda (Loss de Reconstrução MSE) no Treinamento e Validação por Arquitetura (5 Folds em GPU)",
        fontsize=14.8, fontweight="bold", color="#0F172A", y=0.98,
    )
    fig.text(
        0.5, 0.942,
        "Linha Sólida: Loss de Treino (Janelas Normais) | Linha Tracejada: Loss de Validação Out-of-Fold | Sombra: ±1 Desvio Padrão entre os 5 Folds",
        ha="center", va="center", fontsize=10.2, color="#475569",
    )

    agg = (
        df_loss.groupby(["architecture", "epoch"], as_index=False)
        .agg(
            tr_rec_mean=("train_rec_loss", "mean"),
            tr_rec_std=("train_rec_loss", "std"),
            va_rec_mean=("val_rec_loss", "mean"),
            va_rec_std=("val_rec_loss", "std"),
        )
    )

    for idx, arch_name in enumerate(ARCHITECTURES):
        ax = axes[idx]
        sub = agg[agg["architecture"] == arch_name].sort_values("epoch")
        ep = sub["epoch"].to_numpy()
        tr_m = sub["tr_rec_mean"].to_numpy()
        tr_s = sub["tr_rec_std"].fillna(0).to_numpy()
        va_m = sub["va_rec_mean"].to_numpy()
        va_s = sub["va_rec_std"].fillna(0).to_numpy()

        c = ARCH_COLORS.get(arch_name, "#1D4ED8")
        ax.plot(ep, tr_m, color=c, lw=2.2, label=f"Treino (final={tr_m[-1]:.4f})")
        ax.fill_between(ep, np.maximum(0, tr_m - tr_s), tr_m + tr_s, color=c, alpha=0.16)

        ax.plot(ep, va_m, color="#DC2626", lw=2.0, linestyle="--", label=f"Validação (final={va_m[-1]:.4f})")
        ax.fill_between(ep, np.maximum(0, va_m - va_s), va_m + va_s, color="#DC2626", alpha=0.13)

        ax.set_title(f"{idx + 1}. {arch_name}", fontsize=10.8, fontweight="bold", color="#0F172A")
        ax.set_xlabel("Época", fontsize=9.0)
        ax.set_ylabel("Loss de Reconstrução (MSE)", fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # 8º subplot: Comparativo direto da Loss de Validação das 7 Arquiteturas
    ax_all = axes[7]
    for arch_name in ARCHITECTURES:
        sub = agg[agg["architecture"] == arch_name].sort_values("epoch")
        ax_all.plot(
            sub["epoch"],
            sub["va_rec_mean"],
            lw=2.0,
            color=ARCH_COLORS.get(arch_name, "#1D4ED8"),
            label=f"{arch_name} ({sub['va_rec_mean'].iloc[-1]:.4f})",
        )
    ax_all.set_title("8. Comparativo: Loss de Validação (7 Modelos)", fontsize=10.8, fontweight="bold", color="#0F172A")
    ax_all.set_xlabel("Época", fontsize=9.0)
    ax_all.set_ylabel("Val Loss (MSE)", fontsize=9.0)
    ax_all.grid(True, linestyle=":", alpha=0.55)
    ax_all.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

    plt.tight_layout(rect=[0, 0.01, 1, 0.925])
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvo: {out_path}", flush=True)


def plot_loss_by_pump(df_loss: pd.DataFrame, out_path: str):
    """Plota as curvas de perda (Treino vs Validação) para cada uma das 6 bombas + Dataset Completo."""
    ds_list = [d for d in DATASETS if d in df_loss["dataset"].unique()]
    cols = 2
    rows = (len(ds_list) + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 4.1 * rows), dpi=300, facecolor="white")
    axes = axes.flatten()

    fig.suptitle(
        "Convergência da Perda (Loss de Reconstrução e Loss Híbrida) por Bomba e Dataset Completo (GPU RTX 5080)",
        fontsize=14.2, fontweight="bold", color="#0F172A", y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "Comparação das curvas de Treinamento e Validação (5 Folds) para modelos Unimodais 1D, 2D GADF e Híbridos Multimodais",
        ha="center", va="center", fontsize=10.0, color="#475569",
    )

    rep_archs = ["AE_LSTM1D", "AE_CNN2D_GAF", "Hybrid_Transformer_CNN2D"]

    for idx, ds_name in enumerate(ds_list):
        ax = axes[idx]
        sub_ds = df_loss[df_loss["dataset"] == ds_name]
        agg = (
            sub_ds.groupby(["architecture", "epoch"], as_index=False)
            .agg(
                tr_mean=("train_rec_loss", "mean"),
                va_mean=("val_rec_loss", "mean"),
                va_std=("val_rec_loss", "std"),
            )
        )

        for arch_name in rep_archs:
            sa = agg[agg["architecture"] == arch_name].sort_values("epoch")
            if len(sa) == 0:
                continue
            c = ARCH_COLORS[arch_name]
            ep = sa["epoch"].to_numpy()
            tr_m = sa["tr_mean"].to_numpy()
            va_m = sa["va_mean"].to_numpy()
            va_s = sa["va_std"].fillna(0).to_numpy()

            ax.plot(ep, tr_m, color=c, lw=1.9, linestyle="-", label=f"{arch_name} [Treino]")
            ax.plot(ep, va_m, color=c, lw=1.9, linestyle="--", label=f"{arch_name} [Val={va_m[-1]:.4f}]")
            ax.fill_between(ep, np.maximum(0, va_m - va_s), va_m + va_s, color=c, alpha=0.10)

        ax.set_title(f"{ds_name} — Convergência Treino vs. Validação (5 Folds)", fontsize=10.6, fontweight="bold", color="#0F172A")
        ax.set_xlabel("Época", fontsize=9.0)
        ax.set_ylabel("Loss de Reconstrução (MSE)", fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", ncol=2, fontsize=7.3, framealpha=0.92)

    for j in range(len(ds_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvo: {out_path}", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("================================================================================", flush=True)
    print(f"GERAÇÃO DAS CURVAS DE PERDA (LOSS) DE TREINAMENTO E VALIDAÇÃO EM GPU", flush=True)
    print(f"Dispositivo : {device} ({gpu_name}) | epochs={args.epochs} | batch_size={args.batch_size}", flush=True)
    print("================================================================================", flush=True)

    rng = np.random.default_rng(SEED)
    all_loss_records = []

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
            continue

        valid_norm = np.flatnonzero(norm_mask)
        valid_anom = np.flatnonzero(anom_mask)
        norm_idx = valid_norm if len(valid_norm) <= args.max_samples else np.sort(rng.choice(valid_norm, size=args.max_samples, replace=False))
        anom_idx = valid_anom if len(valid_anom) <= args.max_samples else np.sort(rng.choice(valid_anom, size=args.max_samples, replace=False))

        sel_idx = np.sort(np.concatenate([norm_idx, anom_idx]))
        y_sel = is_anom_all[sel_idx].astype(np.int32)
        folds_sel = folds_all[sel_idx]

        x1d_sel, x2d_sel = load_dataset_tensors(ds_name, args.window, sel_idx)

        for arch_name in ARCHITECTURES:
            recs = track_loss_5folds(
                arch_name=arch_name,
                x1d_all=x1d_sel,
                x2d_all=x2d_sel,
                y_all=y_sel,
                folds_cv=folds_sel,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            for r in recs:
                r["dataset"] = ds_name
                r["window"] = args.window
            all_loss_records.extend(recs)

        print(f"[OK] {ds_name:16s} | 7 arquiteturas x 5 folds x {args.epochs} épocas concluídas em {time.time() - t_ds:.1f}s", flush=True)
        del x1d_sel, x2d_sel
        gc.collect()

    df_loss = pd.DataFrame(all_loss_records)
    csv_out = os.path.join(RESULTS_DIR, "historico_perda_loss_epocas_5folds.csv")
    df_loss.to_csv(csv_out, index=False)

    fig_arch_out = os.path.join(PLOTS_DIR, "curvas_perda_loss_7_arquiteturas.png")
    fig_pump_out = os.path.join(PLOTS_DIR, "curvas_perda_loss_por_bomba.png")
    plot_loss_by_architecture(df_loss, fig_arch_out)
    plot_loss_by_pump(df_loss, fig_pump_out)

    print(f"\nConcluído em {time.time() - t0:.1f}s! CSV salvo em: {csv_out}", flush=True)


if __name__ == "__main__":
    main()
