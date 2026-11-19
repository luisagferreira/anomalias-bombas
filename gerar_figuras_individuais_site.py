"""
Gera todas as figuras individuais (uma imagem por gráfico/bomba/arquitetura em 300 DPI)
para exibição no formato de visualizador imagem-por-imagem no portal GitHub Pages (`docs/index.html`).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANOM_DIR = os.path.join(
    "Dataset_completo_bombas",
    "Resultados_Modelos_1D_2D",
    "Deteccao_Anomalias_Operador",
)
PLOTS_DIR = os.path.join(ANOM_DIR, "Figuras_Anomalias_Operador")
os.makedirs(PLOTS_DIR, exist_ok=True)

ARCH_COLORS = {
    "AE_CNN1D": "#0284C7",
    "AE_LSTM1D": "#2563EB",
    "AE_Transformer1D": "#7C3AED",
    "AE_CNN2D_GAF": "#D97706",
    "Hybrid_CNN1D_CNN2D": "#059669",
    "Hybrid_LSTM1D_CNN2D": "#DC2626",
    "Hybrid_Transformer_CNN2D": "#0F172A",
}
ARCHITECTURES = list(ARCH_COLORS.keys())
DATASETS = [
    "Bomba_1305",
    "Bomba_1307",
    "Bomba_2307",
    "Bomba_3307",
    "Bomba_4307",
    "Bomba_5307",
    "Dataset_Completo",
]


def main():
    # ==========================================================================
    # 1. FIGURA 10: 8 gráficos individuais de Loss por Arquitetura + Comparativo
    # ==========================================================================
    df_loss = pd.read_csv(os.path.join(ANOM_DIR, "historico_perda_loss_epocas_5folds.csv"))
    agg_arch = df_loss.groupby(["architecture", "epoch"], as_index=False).agg(
        tr_m=("train_rec_loss", "mean"),
        tr_s=("train_rec_loss", "std"),
        va_m=("val_rec_loss", "mean"),
        va_s=("val_rec_loss", "std"),
    )

    for idx, arch in enumerate(ARCHITECTURES, 1):
        sub = agg_arch[agg_arch["architecture"] == arch].sort_values("epoch")
        ep = sub["epoch"].to_numpy()
        tr_m, tr_s = sub["tr_m"].to_numpy(), sub["tr_s"].fillna(0).to_numpy()
        va_m, va_s = sub["va_m"].to_numpy(), sub["va_s"].fillna(0).to_numpy()
        c = ARCH_COLORS[arch]

        fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300, facecolor="white")
        ax.plot(ep, tr_m, color=c, lw=2.5, label=f"Loss de Treino — Janelas Normais (final = {tr_m[-1]:.4f})")
        ax.fill_between(ep, np.maximum(0, tr_m - tr_s), tr_m + tr_s, color=c, alpha=0.18, label="±1 Desvio Padrão (Treino nos 5 Folds)")
        ax.plot(ep, va_m, color="#DC2626", lw=2.3, linestyle="--", label=f"Loss de Validação Out-of-Fold (final = {va_m[-1]:.4f})")
        ax.fill_between(ep, np.maximum(0, va_m - va_s), va_m + va_s, color="#DC2626", alpha=0.14, label="±1 Desvio Padrão (Validação nos 5 Folds)")
        ax.set_title(f"Figura 10 ({idx}/8) — Curva de Perda (Loss MSE): {arch}", fontsize=13.0, fontweight="bold", color="#0F172A", pad=12)
        ax.set_xlabel("Época de Treinamento (GPU RTX 5080)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Loss de Reconstrução (MSE)", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.94)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"fig10_loss_{arch}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300, facecolor="white")
    for arch in ARCHITECTURES:
        sub = agg_arch[agg_arch["architecture"] == arch].sort_values("epoch")
        v_end = float(sub["va_m"].iloc[-1])
        ax.plot(sub["epoch"], sub["va_m"], lw=2.3, color=ARCH_COLORS[arch], label=f"{arch} (Val MSE final = {v_end:.4f})")
    ax.set_title("Figura 10 (8/8) — Comparativo Direto da Loss de Validação entre as 7 Arquiteturas", fontsize=13.0, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Época de Treinamento (GPU RTX 5080)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Loss de Validação Out-of-Fold (MSE)", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.2, framealpha=0.94)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig10_loss_Comparativo_7_Modelos.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ==========================================================================
    # 2. FIGURA 11: 7 gráficos individuais de Loss por Bomba + Dataset Completo
    # ==========================================================================
    rep_archs = ["AE_LSTM1D", "AE_CNN2D_GAF", "Hybrid_Transformer_CNN2D"]
    for idx, ds_name in enumerate(DATASETS, 1):
        sub_ds = df_loss[df_loss["dataset"] == ds_name]
        agg_p = sub_ds.groupby(["architecture", "epoch"], as_index=False).agg(
            tr_m=("train_rec_loss", "mean"),
            va_m=("val_rec_loss", "mean"),
            va_s=("val_rec_loss", "std"),
        )
        fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300, facecolor="white")
        for arch in rep_archs:
            sa = agg_p[agg_p["architecture"] == arch].sort_values("epoch")
            if len(sa) == 0:
                continue
            c = ARCH_COLORS[arch]
            ep = sa["epoch"].to_numpy()
            tr_m, va_m, va_s = sa["tr_m"].to_numpy(), sa["va_m"].to_numpy(), sa["va_s"].fillna(0).to_numpy()
            ax.plot(ep, tr_m, color=c, lw=2.2, linestyle="-", label=f"{arch} — Treino (final={tr_m[-1]:.4f})")
            ax.plot(ep, va_m, color=c, lw=2.2, linestyle="--", label=f"{arch} — Validação (final={va_m[-1]:.4f})")
            ax.fill_between(ep, np.maximum(0, va_m - va_s), va_m + va_s, color=c, alpha=0.12)
        ax.set_title(f"Figura 11 ({idx}/7) — Convergência da Perda (Treino vs. Validação 5 Folds): {ds_name}", fontsize=13.0, fontweight="bold", color="#0F172A", pad=12)
        ax.set_xlabel("Época de Treinamento (GPU RTX 5080)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Loss de Reconstrução (MSE)", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", ncol=2, fontsize=8.8, framealpha=0.94)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"fig11_loss_{ds_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ==========================================================================
    # 3. FIGURA 12: 7 Matrizes de Confusão Individuais por Bomba (Janela 15min)
    # ==========================================================================
    df_28 = pd.read_csv(os.path.join(ANOM_DIR, "resumo_deteccao_ano_acima_2_5_por_bomba_janela.csv"))
    df_15 = df_28[df_28["window"] == "15min"]

    for idx, ds_name in enumerate(DATASETS, 1):
        sub = df_15[df_15["dataset"] == ds_name].sort_values("oof_best_f1_hybrid", ascending=False)
        best = sub.iloc[0]
        arch = best["architecture"]
        vn, fp, fn, vp = int(best["VN"]), int(best["FP"]), int(best["FN"]), int(best["VP"])
        cm_pct = np.array([
            [100.0 * vn / max(1, vn + fp), 100.0 * fp / max(1, vn + fp)],
            [100.0 * fn / max(1, fn + vp), 100.0 * vp / max(1, fn + vp)],
        ])

        fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=300, facecolor="white")
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Porcentagem por Classe Real (%)", fontsize=9.5, fontweight="bold")

        labels_txt = [
            [
                f"VN (Normal Correto)\n{vn:,} janelas\n({cm_pct[0, 0]:.2f}%)".replace(",", "."),
                f"FP (Falso Alarme)\n{fp:,} janelas\n({cm_pct[0, 1]:.2f}%)".replace(",", "."),
            ],
            [
                f"FN (Omissão)\n{fn:,} janelas\n({cm_pct[1, 0]:.2f}%)".replace(",", "."),
                f"VP (Anomalia Detectada)\n{vp:,} janelas\n({cm_pct[1, 1]:.2f}%)".replace(",", "."),
            ],
        ]
        for r in range(2):
            for c in range(2):
                color_t = "white" if cm_pct[r, c] > 55 else "#0F172A"
                ax.text(c, r, labels_txt[r][c], ha="center", va="center", fontsize=11.0, fontweight="bold", color=color_t)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predito: Normal (0)", "Predito: Anomalia (1)"], fontsize=10.5, fontweight="bold")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Real: Normal (0)", "Real: Anomalia (1)"], fontsize=10.5, fontweight="bold")
        roc_v = float(best["oof_roc_auc_hybrid"])
        f1_v = float(best["oof_best_f1_hybrid"])
        pr_v = float(best["oof_precision_at_best_f1"])
        rc_v = float(best["oof_recall_at_best_f1"])
        ax.set_title(
            f"Figura 12 ({idx}/7) — Matriz de Confusão 5-Folds: {ds_name} ({arch})\n"
            f"ROC-AUC = {roc_v:.4f} | F1 = {f1_v:.4f} | Precisão = {pr_v:.4f} | Recall = {rc_v:.4f}",
            fontsize=11.5, fontweight="bold", color="#0F172A", pad=14,
        )
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"fig12_cm_{ds_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ==========================================================================
    # 4. FIGURA 13: 4 Gráficos Individuais por Escala de Janela
    # ==========================================================================
    wins_order = ["15min", "30min", "1h", "2h"]
    agg_win = df_28.groupby(["window", "architecture"], as_index=False).agg(
        roc=("oof_roc_auc_hybrid", "mean"),
        pr=("oof_pr_auc_hybrid", "mean"),
        f1=("oof_best_f1_hybrid", "mean"),
        far=("oof_false_alarm_rate_pct", "mean"),
    )

    metrics_fig13 = [
        ("fig13a_janela_roc_auc.png", "roc", "ROC-AUC Médio", (0.93, 1.00), "Figura 13A (1/4) — Comparação de ROC-AUC por Escala de Janela Temporal"),
        ("fig13b_janela_pr_auc.png", "pr", "PR-AUC Médio", (0.86, 0.98), "Figura 13B (2/4) — Comparação de PR-AUC por Escala de Janela Temporal"),
        ("fig13c_janela_f1_score.png", "f1", "F1-Score Médio", (0.80, 0.93), "Figura 13C (3/4) — Comparação de F1-Score por Escala de Janela Temporal"),
        ("fig13d_janela_falsos_alarmes.png", "far", "Taxa de Falsos Alarmes (%)", (0.0, 9.5), "Figura 13D (4/4) — Taxa de Falsos Alarmes (%) por Escala de Janela Temporal"),
    ]

    for fname, col, ylabel, ylim, title in metrics_fig13:
        fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=300, facecolor="white")
        x = np.arange(len(wins_order))
        w = 0.11
        for i, arch in enumerate(ARCHITECTURES):
            vals = [float(agg_win[(agg_win["window"] == wn) & (agg_win["architecture"] == arch)][col].iloc[0]) for wn in wins_order]
            offset = (i - 3) * w
            bars = ax.bar(x + offset, vals, width=w * 0.92, color=ARCH_COLORS[arch], label=arch)
            for b in bars:
                h = b.get_height()
                fmt = f"{h:.1f}%" if col == "far" else f"{h:.3f}"
                ax.text(b.get_x() + b.get_width() / 2, h + (ylim[1] - ylim[0]) * 0.01, fmt, ha="center", va="bottom", fontsize=6.8, rotation=90, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["15 min (L=90)", "30 min (L=180)", "1 hora (L=360)", "2 horas (L=720)"], fontsize=10.5, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="#0F172A", pad=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.55)
        ax.legend(loc="upper right" if col == "far" else "lower right", ncol=2, fontsize=8.5, framealpha=0.94)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ==========================================================================
    # 5. FIGURA 14: 4 Gráficos Individuais para as 11 Formulações de Rótulos
    # ==========================================================================
    df_11 = pd.read_csv(os.path.join(ANOM_DIR, "resumo_global_11_abordagens_rotulos.csv"))
    labels_11 = df_11["approach_label"].tolist()
    y_pos = np.arange(len(labels_11))

    # 14A: ROC-AUC e PR-AUC
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300, facecolor="white")
    h = 0.36
    b1 = ax.barh(y_pos - h / 2, df_11["roc_auc_mean"], height=h, color="#1D4ED8", label="ROC-AUC Médio")
    b2 = ax.barh(y_pos + h / 2, df_11["pr_auc_mean"], height=h, color="#0284C7", label="PR-AUC Médio")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_11, fontsize=9.2, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.20, 1.08)
    ax.set_xlabel("Score (ROC-AUC / PR-AUC)", fontsize=10.5, fontweight="bold")
    ax.set_title("Figura 14A (1/4) — ROC-AUC e PR-AUC nas 11 Formulações de Rótulos (GPU)", fontsize=12.5, fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=9.5)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width()
            ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", ha="left", fontsize=8.0, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig14a_rotulos_roc_pr.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 14B: F1-Score, Precisão e Recall
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300, facecolor="white")
    h = 0.26
    b1 = ax.barh(y_pos - h, df_11["best_f1_mean"], height=h, color="#16A34A", label="Melhor F1-Score")
    b2 = ax.barh(y_pos, df_11["precision_mean"], height=h, color="#2563EB", label="Precisão")
    b3 = ax.barh(y_pos + h, df_11["recall_mean"], height=h, color="#D97706", label="Recall (Sensibilidade)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_11, fontsize=9.2, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.20, 1.05)
    ax.set_xlabel("Métrica (0 a 1)", fontsize=10.5, fontweight="bold")
    ax.set_title("Figura 14B (2/4) — F1-Score, Precisão e Recall nas 11 Formulações de Rótulos", fontsize=12.5, fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=9.2)
    for b in b1:
        v = b.get_width()
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"F1={v:.3f}", va="center", ha="left", fontsize=7.8, fontweight="bold", color="#15803D")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig14b_rotulos_f1_prec_rec.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 14C: Correlação de Spearman
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300, facecolor="white")
    b1 = ax.barh(y_pos, df_11["spearman_corr_mean"], height=0.55, color="#7C3AED", label="Correlação de Spearman (ρ) com a Severidade/Risco")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_11, fontsize=9.2, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.78)
    ax.set_xlabel("Coeficiente de Correlação de Spearman (ρ)", fontsize=10.5, fontweight="bold")
    ax.set_title("Figura 14C (3/4) — Correlação de Spearman entre Probabilidade do Modelo e Risco Real", fontsize=12.5, fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=9.5)
    for b in b1:
        v = b.get_width()
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", ha="left", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig14c_rotulos_spearman.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 14D: Taxa de Falsos Alarmes (%) e MAE (%)
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300, facecolor="white")
    h = 0.36
    b1 = ax.barh(y_pos - h / 2, df_11["false_alarm_rate_pct_mean"], height=h, color="#DC2626", label="Taxa de Falsos Alarmes — FAR (%)")
    b2 = ax.barh(y_pos + h / 2, df_11["prob_mae_pct_mean"], height=h, color="#64748B", label="Erro Médio Absoluto da Probabilidade — MAE (%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_11, fontsize=9.2, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 42.0)
    ax.set_xlabel("Porcentagem (%) — Menor é Melhor", fontsize=10.5, fontweight="bold")
    ax.set_title("Figura 14D (4/4) — Taxa de Falsos Alarmes (%) e MAE da Probabilidade (%) por Formulação", fontsize=12.5, fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=9.2)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width()
            ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", ha="left", fontsize=8.0, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig14d_rotulos_far_mae.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Todas as 30 figuras individuais (Fig 10, 11, 12, 13 e 14) foram geradas com sucesso!")


if __name__ == "__main__":
    main()
