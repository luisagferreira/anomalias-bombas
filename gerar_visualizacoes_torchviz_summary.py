"""
Gera as visualizações das 7 arquiteturas de Autoencoders na pasta `Figuras_Arquiteturas`
utilizando oficialmente `torchviz` (grafo computacional autograd do PyTorch) e
`torchinfo.summary` (tabela de camadas, shapes dos tensores, parâmetros e Mult-Adds).
"""

import os
import glob
import tempfile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from torchviz import make_dot
from torchinfo import summary

from treinar_modelos_1d_2d_gaf import AutoencoderModel, ARCHITECTURES

# Garante que o binário `dot` do Graphviz seja encontrado mesmo sem reiniciar o terminal
GRAPHVIZ_CANDIDATES = [
    r"C:\Users\202510482\.gemini\antigravity\brain\820f487b-8644-4f3d-b47e-9fd7fab98fec\scratch\graphviz\Graphviz-16.1.1-win64\bin",
    r"C:\Program Files\Graphviz\bin",
]
for gv_bin in GRAPHVIZ_CANDIDATES:
    if os.path.isdir(gv_bin) and gv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = gv_bin + os.pathsep + os.environ.get("PATH", "")

OUTPUT_DIR = os.path.abspath(os.path.join(".", "Dataset_completo_bombas", "Figuras_Arquiteturas"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARCH_TITLES = {
    "AE_CNN1D": "Arquitetura 1: AE_CNN1D (Autoencoder Convolucional 1D)",
    "AE_LSTM1D": "Arquitetura 2: AE_LSTM1D (Patch-LSTM Bidirecional 1D)",
    "AE_Transformer1D": "Arquitetura 3: AE_Transformer1D (Patch-Transformer 1D)",
    "AE_CNN2D_GAF": "Arquitetura 4: AE_CNN2D_GAF (Autoencoder Convolucional 2D — GADF)",
    "Hybrid_CNN1D_CNN2D": "Arquitetura 5: Hybrid_CNN1D_CNN2D (Fusão Multimodal CNN-1D + CNN-2D)",
    "Hybrid_LSTM1D_CNN2D": "Arquitetura 6: Hybrid_LSTM1D_CNN2D (Fusão Multimodal LSTM-1D + CNN-2D)",
    "Hybrid_Transformer_CNN2D": "Arquitetura 7: Hybrid_Transformer_CNN2D (Fusão Multimodal Transformer-1D + CNN-2D)",
}


def render_torchviz_png(model: torch.nn.Module, x_1d: torch.Tensor, x_2d: torch.Tensor, tmp_dir: str, tag: str) -> str:
    """Gera o grafo computacional via `torchviz.make_dot` e renderiza em PNG de alta resolução."""
    model.eval()
    rec_1d, rec_2d = model(x_1d, x_2d)
    targets = (rec_1d, rec_2d) if rec_2d is not None else rec_1d

    dot = make_dot(
        targets,
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    dot.attr(dpi="260", rankdir="TB", bgcolor="white", margin="0.15", nodesep="0.22", ranksep="0.35")
    dot.attr("node", fontname="DejaVu Sans", fontsize="9")
    dot.attr("edge", color="#475569", arrowsize="0.65")

    base_path = os.path.join(tmp_dir, f"viz_{tag}")
    rendered_path = dot.render(base_path, format="png", cleanup=True)
    return rendered_path


def get_torchinfo_summary_text(arch_name: str, n_channels: int = 4, seq_len: int = 90) -> tuple[str, object]:
    """Executa `torchinfo.summary` e retorna o texto formatado e o objeto ModelStatistics."""
    model = AutoencoderModel(arch_name, n_channels=n_channels, seq_len=seq_len)
    model.eval()
    x_1d = torch.randn(1, n_channels, seq_len)
    x_2d = torch.randn(1, n_channels, 32, 32)
    stats = summary(
        model,
        input_data=(x_1d, x_2d),
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        verbose=0,
    )
    return str(stats), stats


def create_composite_figure(
    arch_name: str,
    viz_png_path: str,
    summary_text: str,
    stats_c4,
    stats_c24,
    out_filename: str,
):
    """Combina o grafo `torchviz` (esquerda) e o `torchinfo.summary` (direita) em uma figura limpa."""
    img = mpimg.imread(viz_png_path)
    img_h, img_w = img.shape[0], img.shape[1]
    aspect = img_w / max(1, img_h)

    summary_lines = summary_text.splitlines()
    max_line_len = max((len(line) for line in summary_lines), default=80)
    n_lines = len(summary_lines)

    fig_h = max(9.5, min(16.0, 2.2 + n_lines * 0.23))
    viz_col_w = max(5.5, min(9.0, fig_h * aspect * 0.85))
    sum_col_w = max(9.5, max_line_len * 0.088)
    fig_w = viz_col_w + sum_col_w + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[viz_col_w, sum_col_w], wspace=0.06, top=0.88, bottom=0.05, left=0.02, right=0.98)

    title_str = ARCH_TITLES.get(arch_name, arch_name)
    fig.suptitle(title_str, fontsize=14.5, fontweight="bold", color="#0F172A", y=0.965)
    subtitle_str = (
        f"Visualização PyTorch (torchviz + torchinfo.summary) | "
        f"Bomba Individual (C=4, L=90): {stats_c4.trainable_params:,} params | "
        f"Dataset Completo (C=24, L=90): {stats_c24.trainable_params:,} params"
    ).replace(",", ".")
    fig.text(0.5, 0.922, subtitle_str, ha="center", va="center", fontsize=10.0, color="#475569")

    # Painel Esquerdo: torchviz
    ax_viz = fig.add_subplot(gs[0, 0])
    ax_viz.imshow(img)
    ax_viz.axis("off")
    ax_viz.set_title("Grafo Computacional Autograd (torchviz)", fontsize=11.0, fontweight="bold", color="#1E293B", pad=8)

    # Painel Direito: torchinfo.summary
    ax_sum = fig.add_subplot(gs[0, 1])
    ax_sum.axis("off")
    ax_sum.set_title("Resumo Estrutural de Camadas e Tensores (torchinfo.summary)", fontsize=11.0, fontweight="bold", color="#1E293B", pad=8)

    box = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.3,
        transform=ax_sum.transAxes, zorder=1,
    )
    ax_sum.add_patch(box)

    ax_sum.text(
        0.02, 0.97, summary_text,
        transform=ax_sum.transAxes,
        ha="left", va="top",
        fontsize=7.6,
        fontfamily="DejaVu Sans Mono",
        color="#0F172A",
        linespacing=1.28,
        zorder=2,
    )

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvo: {out_filename}")


def create_hybrids_panel(hybrid_imgs: dict[str, str], stats_map: dict[str, tuple]):
    """Atualiza `arquitetura_Hibridos_Multimodais.png` reunindo os 3 modelos híbridos com torchviz."""
    hybrids = ["Hybrid_CNN1D_CNN2D", "Hybrid_LSTM1D_CNN2D", "Hybrid_Transformer_CNN2D"]
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 9.5), dpi=300, facecolor="white")
    fig.suptitle(
        "Arquiteturas 5, 6 e 7: Modelos Híbridos Multimodais (1D + 2D GADF) — Grafo PyTorch (torchviz)",
        fontsize=15.0, fontweight="bold", color="#0F172A", y=0.97,
    )
    fig.text(
        0.5, 0.925,
        "Comparação dos grafos computacionais autograd e contagem de parâmetros (torchinfo.summary) com fusão no Espaço Latente Z = 32",
        ha="center", va="center", fontsize=10.5, color="#475569",
    )

    for ax, h_name in zip(axes, hybrids):
        img = mpimg.imread(hybrid_imgs[h_name])
        ax.imshow(img)
        ax.axis("off")
        s4, s24 = stats_map[h_name]
        p4_str = f"{s4.trainable_params:,}".replace(",", ".")
        p24_str = f"{s24.trainable_params:,}".replace(",", ".")
        ax.set_title(
            f"{h_name}\nC=4: {p4_str} params | C=24: {p24_str} params",
            fontsize=11.0, fontweight="bold", color="#1E293B", pad=8,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    out_path = os.path.join(OUTPUT_DIR, "arquitetura_Hibridos_Multimodais.png")
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Salvo: arquitetura_Hibridos_Multimodais.png")


def main():
    print(f"Gerando visualizações torchviz + torchinfo.summary em: {OUTPUT_DIR}")
    all_summaries_txt = []
    hybrid_imgs = {}
    stats_map = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for arch_name in ARCHITECTURES:
            model_c4 = AutoencoderModel(arch_name, n_channels=4, seq_len=90)
            x1d_c4 = torch.randn(1, 4, 90)
            x2d_c4 = torch.randn(1, 4, 32, 32)

            viz_png = render_torchviz_png(model_c4, x1d_c4, x2d_c4, tmp_dir, arch_name)
            sum_txt_c4, stats_c4 = get_torchinfo_summary_text(arch_name, n_channels=4, seq_len=90)
            sum_txt_c24, stats_c24 = get_torchinfo_summary_text(arch_name, n_channels=24, seq_len=90)

            stats_map[arch_name] = (stats_c4, stats_c24)
            if arch_name.startswith("Hybrid_"):
                hybrid_imgs[arch_name] = viz_png

            out_name = f"arquitetura_{arch_name}.png"
            create_composite_figure(arch_name, viz_png, sum_txt_c4, stats_c4, stats_c24, out_name)

            all_summaries_txt.append(f"{'=' * 90}\nMODELO: {arch_name} (C=4, L=90 | GADF 4x32x32)\n{'=' * 90}\n{sum_txt_c4}\n")
            all_summaries_txt.append(f"{'-' * 90}\nMODELO: {arch_name} — Dataset Completo (C=24, L=90 | GADF 24x32x32)\n{'-' * 90}\n{sum_txt_c24}\n\n")

        create_hybrids_panel(hybrid_imgs, stats_map)

    txt_out = os.path.join(OUTPUT_DIR, "resumo_torchinfo_todas_arquiteturas.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(all_summaries_txt))
    print(f"[OK] Salvo: resumo_torchinfo_todas_arquiteturas.txt")


if __name__ == "__main__":
    main()
