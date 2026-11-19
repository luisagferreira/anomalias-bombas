"""
Script para geração programática de diagramas técnicos de alta resolução (300 DPI)
mostrando fielmente o fluxo de trabalho, as decisões de arquitetura (dimensões de entrada,
camadas, neurônios, filtros, funções de ativação, espaço latente Z=32) e todas as 7
arquiteturas avaliadas no projeto de detecção de anomalias em bombas industriais.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from treinar_modelos_1d_2d_gaf import AutoencoderModel, ARCHITECTURES
    HAS_TORCH_MODELS = True
except Exception:
    HAS_TORCH_MODELS = False

OUTPUT_DIR = os.path.abspath(os.path.join(".", "Dataset_completo_bombas", "Figuras_Arquiteturas"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename: str):
    out = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(out):
        try:
            os.remove(out)
        except OSError:
            pass
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {filename}")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9.5

# Paleta de cores padronizada (estilo IEEE / Elsevier)
C_INPUT = "#E0F2FE"
C_INPUT_BD = "#0284C7"
C_ENCODER = "#DBEAFE"
C_ENCODER_BD = "#2563EB"
C_SPECIAL = "#F3E8FF"
C_SPECIAL_BD = "#9333EA"
C_LATENT = "#FEF3C7"
C_LATENT_BD = "#D97706"
C_DECODER = "#DCFCE7"
C_DECODER_BD = "#16A34A"
C_OUTPUT = "#FEE2E2"
C_OUTPUT_BD = "#DC2626"
C_NOTE = "#F8FAFC"
C_NOTE_BD = "#64748B"


def get_param_count(arch_name: str, c: int, l: int = 90) -> str:
    if not HAS_TORCH_MODELS:
        return "N/D"
    m = AutoencoderModel(arch_name, c, l)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return f"{n:,}".replace(",", ".")


def draw_box(ax, x, y, w, h, title, subtitle="", dims="", bg="#FFFFFF", bd="#000000", lw=1.6, title_fs=9.2):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=bg, edgecolor=bd, linewidth=lw, zorder=3
    )
    ax.add_patch(box)

    if subtitle and dims:
        ax.text(x, y + h * 0.27, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#0F172A", zorder=4)
        ax.text(x, y - h * 0.02, subtitle, ha="center", va="center", fontsize=7.6, color="#334155", zorder=4)
        ax.text(x, y - h * 0.32, dims, ha="center", va="center", fontsize=7.5, fontweight="bold", color=bd, zorder=4)
    elif subtitle:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#0F172A", zorder=4)
        ax.text(x, y - h * 0.19, subtitle, ha="center", va="center", fontsize=7.8, color="#334155", zorder=4)
    elif dims:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#0F172A", zorder=4)
        ax.text(x, y - h * 0.20, dims, ha="center", va="center", fontsize=7.8, fontweight="bold", color=bd, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#0F172A", zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#475569", rad=0.0, ls="-"):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_length=0.45,head_width=0.25",
            color=color, lw=1.7, shrinkA=2, shrinkB=2,
            connectionstyle=f"arc3,rad={rad}", linestyle=ls
        ),
        zorder=2
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.16
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.4, color=color, fontweight="bold", zorder=5)


def draw_decisions_panel(ax, x, y, w, h, title, bullets):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=C_NOTE, edgecolor=C_NOTE_BD, linewidth=1.2, linestyle="--", zorder=2
    )
    ax.add_patch(box)
    ax.text(x - w / 2 + 0.2, y + h / 2 - 0.22, title, ha="left", va="top", fontsize=9.0, fontweight="bold", color="#1E293B", zorder=4)
    text_body = "\n".join([f"• {b}" for b in bullets])
    ax.text(x - w / 2 + 0.2, y + h / 2 - 0.52, text_body, ha="left", va="top", fontsize=8.0, color="#334155", linespacing=1.45, zorder=4)


# ==============================================================================
# 1. AE_CNN1D
# ==============================================================================
def gerar_ae_cnn1d():
    p4 = get_param_count("AE_CNN1D", 4, 90)
    p24 = get_param_count("AE_CNN1D", 24, 90)

    fig, ax = plt.subplots(figsize=(16.5, 7.2), dpi=300)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-1.6, 5.2)
    ax.axis("off")

    ax.text(8.0, 4.75, "Arquitetura 1: AE_CNN1D (Autoencoder Convolucional 1D)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(8.0, 4.30, f"Extração local de padrões temporais | Espaço Latente Z = 32 | Parâmetros (L=90): {p4} (C=4) / {p24} (C=24)", ha="center", va="center", fontsize=9.8, color="#475569")

    draw_box(ax, 1.0, 2.7, 1.7, 1.45, "Entrada 1D", "Sinal Normalizado\n[-1, 1]", "(B, C, L)", C_INPUT, C_INPUT_BD)
    draw_arrow(ax, 1.85, 2.7, 2.45, 2.7)

    draw_box(ax, 3.35, 2.7, 1.8, 1.45, "Conv1D Bloco 1", "16 filtros, k=5, s=2, p=2\nBatchNorm1d + GELU", "(B, 16, L/2)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 4.25, 2.7, 4.85, 2.7)

    draw_box(ax, 5.75, 2.7, 1.8, 1.45, "Conv1D Bloco 2", "32 filtros, k=5, s=2, p=2\nBatchNorm1d + GELU", "(B, 32, L/4)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 6.65, 2.7, 7.25, 2.7)

    draw_box(ax, 8.15, 2.7, 1.8, 1.45, "Pooling + Flatten", "AdaptiveAvgPool1d(8)\nFlatten (32 × 8)", "(B, 256)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 9.05, 2.7, 9.65, 2.7)

    draw_box(ax, 10.5, 2.7, 1.7, 1.6, "Espaço Latente Z", "Linear(256 → 32)\n+ GELU", "(B, 32)", C_LATENT, C_LATENT_BD, lw=2.2)
    draw_arrow(ax, 11.35, 2.7, 11.95, 2.7)

    draw_box(ax, 12.85, 2.7, 1.8, 1.45, "Decoder1D - Oculta", "Linear(32 → 64)\n+ GELU", "(B, 64)", C_DECODER, C_DECODER_BD)
    draw_arrow(ax, 13.75, 2.7, 14.35, 2.7)

    draw_box(ax, 15.2, 2.7, 1.7, 1.45, "Saída Reconstruída", "Linear(64 → C×L)\n+ Tanh + Reshape", "(B, C, L)", C_DECODER, C_DECODER_BD)

    # Erro MSE
    draw_arrow(ax, 1.0, 1.95, 1.0, 1.25, color=C_OUTPUT_BD, ls="--")
    draw_arrow(ax, 15.2, 1.95, 15.2, 1.25, color=C_OUTPUT_BD, ls="--")
    ax.plot([1.0, 15.2], [1.25, 1.25], color=C_OUTPUT_BD, lw=1.6, linestyle="--", zorder=2)
    draw_box(ax, 8.1, 1.25, 5.2, 0.72, "Perda e Score de Anomalia: MSE(X_1D, X_rec_1D) = ||X - X_rec||²", "Treinado apenas com janelas normais (AdamW, lr=3e-3, wd=1e-4)", "", C_OUTPUT, C_OUTPUT_BD)

    draw_decisions_panel(ax, 8.0, -0.45, 15.6, 1.85, "Decisões de Arquitetura (AE_CNN1D)", [
        "Entradas: C = 4 sensores (bomba individual) ou C = 24 (Dataset Completo); L = 90 (15m), 180 (30m), 360 (1h) ou 720 (2h) com taxa de 10s.",
        "Kernel k=5 e Stride s=2: Captura correlações de 50 segundos por filtro reduzindo a resolução temporal pela metade a cada camada convolucional.",
        "AdaptiveAvgPool1d(8): Padroniza a saída para exatamente 8 posições (256 neurônios) independentemente do tamanho da janela L (90 a 720).",
        "Gargalo Latente Z=32 e Decoder Tanh: Força compressão informacional e garante reconstrução limitada ao mesmo intervalo [-1, 1] da entrada."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_AE_CNN1D.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 2. AE_LSTM1D
# ==============================================================================
def gerar_ae_lstm1d():
    p4 = get_param_count("AE_LSTM1D", 4, 90)
    p24 = get_param_count("AE_LSTM1D", 24, 90)

    fig, ax = plt.subplots(figsize=(16.5, 7.2), dpi=300)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-1.6, 5.2)
    ax.axis("off")

    ax.text(8.0, 4.75, "Arquitetura 2: AE_LSTM1D (Patch-LSTM Bidirecional Autoencoder)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(8.0, 4.30, f"Modelagem sequencial por patches temporais | Espaço Latente Z = 32 | Parâmetros (L=90): {p4} (C=4) / {p24} (C=24)", ha="center", va="center", fontsize=9.8, color="#475569")

    draw_box(ax, 1.0, 2.7, 1.7, 1.45, "Entrada 1D", "Sinal Normalizado\n[-1, 1]", "(B, C, L)", C_INPUT, C_INPUT_BD)
    draw_arrow(ax, 1.85, 2.7, 2.45, 2.7)

    draw_box(ax, 3.35, 2.7, 1.8, 1.45, "Patch Partition", "N_PATCHES = 15\nP = L / 15", "(B, 15, C × P)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 4.25, 2.7, 4.85, 2.7)

    draw_box(ax, 5.75, 2.7, 1.8, 1.45, "Projeção Linear", "Linear(C×P → 32)\n+ ReLU", "(B, 15, 32)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 6.65, 2.7, 7.25, 2.7)

    draw_box(ax, 8.15, 2.7, 1.8, 1.45, "Bi-LSTM + Pool", "1 camada, hidden=32\nBidirecional (64) + Mean", "(B, 64)", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 9.05, 2.7, 9.65, 2.7)

    draw_box(ax, 10.5, 2.7, 1.7, 1.6, "Espaço Latente Z", "Linear(64 → 32)\n+ GELU", "(B, 32)", C_LATENT, C_LATENT_BD, lw=2.2)
    draw_arrow(ax, 11.35, 2.7, 11.95, 2.7)

    draw_box(ax, 12.85, 2.7, 1.8, 1.45, "Decoder1D - Oculta", "Linear(32 → 64)\n+ GELU", "(B, 64)", C_DECODER, C_DECODER_BD)
    draw_arrow(ax, 13.75, 2.7, 14.35, 2.7)

    draw_box(ax, 15.2, 2.7, 1.7, 1.45, "Saída Reconstruída", "Linear(64 → C×L)\n+ Tanh + Reshape", "(B, C, L)", C_DECODER, C_DECODER_BD)

    draw_arrow(ax, 1.0, 1.95, 1.0, 1.25, color=C_OUTPUT_BD, ls="--")
    draw_arrow(ax, 15.2, 1.95, 15.2, 1.25, color=C_OUTPUT_BD, ls="--")
    ax.plot([1.0, 15.2], [1.25, 1.25], color=C_OUTPUT_BD, lw=1.6, linestyle="--", zorder=2)
    draw_box(ax, 8.1, 1.25, 5.2, 0.72, "Perda e Score de Anomalia: MSE(X_1D, X_rec_1D) = ||X - X_rec||²", "Melhor modelo global na avaliação anual (ROC-AUC = 0.9812 | Acurácia = 95.9%)", "", C_OUTPUT, C_OUTPUT_BD)

    draw_decisions_panel(ax, 8.0, -0.45, 15.6, 1.85, "Decisões de Arquitetura (AE_LSTM1D)", [
        "Divisão em 15 Patches: 90 (15m), 180 (30m), 360 (1h) e 720 (2h) são todos exatamente divisíveis por 15 (P = 6, 12, 24 e 48 amostras/patch).",
        "Benefício do Patching: Reduz sequências longas (até 720 passos) para apenas 15 tokens temporais, evitando dissipação de gradiente na LSTM.",
        "LSTM Bidirecional (hidden=32): Lê a dinâmica nos dois sentidos (passado → futuro e futuro → passado), concatenando 32 + 32 = 64 estados.",
        "Mean Pooling Temporal: Resume os 15 patches sem privilegiar apenas o último instante, capturando anomalias no início, meio ou fim da janela."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_AE_LSTM1D.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 3. AE_TRANSFORMER1D
# ==============================================================================
def gerar_ae_transformer1d():
    p4 = get_param_count("AE_Transformer1D", 4, 90)
    p24 = get_param_count("AE_Transformer1D", 24, 90)

    fig, ax = plt.subplots(figsize=(16.5, 7.2), dpi=300)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-1.6, 5.2)
    ax.axis("off")

    ax.text(8.0, 4.75, "Arquitetura 3: AE_Transformer1D (Patch-Transformer Autoencoder)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(8.0, 4.30, f"Atenção Global Multi-Cabeça sobre tokens temporais | Espaço Latente Z = 32 | Parâmetros (L=90): {p4} (C=4) / {p24} (C=24)", ha="center", va="center", fontsize=9.8, color="#475569")

    draw_box(ax, 1.0, 2.7, 1.7, 1.45, "Entrada 1D", "Sinal Normalizado\n[-1, 1]", "(B, C, L)", C_INPUT, C_INPUT_BD)
    draw_arrow(ax, 1.85, 2.7, 2.45, 2.7)

    draw_box(ax, 3.35, 2.7, 1.8, 1.45, "Patch Partition", "N_PATCHES = 15\nP = L / 15", "(B, 15, C × P)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 4.25, 2.7, 4.85, 2.7)

    draw_box(ax, 5.75, 2.7, 1.8, 1.45, "Patch + Pos Embed", "Linear(C×P → 32)\n+ PosEmbed(1,15,32)", "(B, 15, 32)", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 6.65, 2.7, 7.25, 2.7)

    draw_box(ax, 8.15, 2.7, 1.8, 1.45, "Transformer Encoder", "1 camada, nhead=4\nd_ff=64, GELU, LN", "(B, 15, 32)", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 9.05, 2.7, 9.65, 2.7)

    draw_box(ax, 10.5, 2.7, 1.7, 1.6, "Espaço Latente Z", "Flatten(480) →\nLinear(480→32)+GELU", "(B, 32)", C_LATENT, C_LATENT_BD, lw=2.2)
    draw_arrow(ax, 11.35, 2.7, 11.95, 2.7)

    draw_box(ax, 12.85, 2.7, 1.8, 1.45, "Decoder1D - Oculta", "Linear(32 → 64)\n+ GELU", "(B, 64)", C_DECODER, C_DECODER_BD)
    draw_arrow(ax, 13.75, 2.7, 14.35, 2.7)

    draw_box(ax, 15.2, 2.7, 1.7, 1.45, "Saída Reconstruída", "Linear(64 → C×L)\n+ Tanh + Reshape", "(B, C, L)", C_DECODER, C_DECODER_BD)

    draw_arrow(ax, 1.0, 1.95, 1.0, 1.25, color=C_OUTPUT_BD, ls="--")
    draw_arrow(ax, 15.2, 1.95, 15.2, 1.25, color=C_OUTPUT_BD, ls="--")
    ax.plot([1.0, 15.2], [1.25, 1.25], color=C_OUTPUT_BD, lw=1.6, linestyle="--", zorder=2)
    draw_box(ax, 8.1, 1.25, 5.2, 0.72, "Perda e Score de Anomalia: MSE(X_1D, X_rec_1D) = ||X - X_rec||²", "Atenção direta entre todos os 15 sub-intervalos da janela", "", C_OUTPUT, C_OUTPUT_BD)

    draw_decisions_panel(ax, 8.0, -0.45, 15.6, 1.85, "Decisões de Arquitetura (AE_Transformer1D)", [
        "Positional Embedding Aprendível (1, 15, 32): Como o mecanismo de atenção não tem noção intrínseca de ordem, injeta a posição temporal de cada patch.",
        "Multi-Head Self-Attention (d_model=32, 4 cabeças): Cada cabeça tem dimensão 8 e pode focar em diferentes padrões (ex: pico de corrente vs. nível).",
        "Feed-Forward Network enxuta (dim_feedforward=64, dropout=0.0): Evita overfitting na modelagem de regime normal com 1 única camada codificadora.",
        "Preservação posicional no Gargalo: Usa Flatten(15 × 32 = 480) → Linear(480 → 32) preservando a identidade de cada patch na compressão."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_AE_Transformer1D.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 4. AE_CNN2D_GAF
# ==============================================================================
def gerar_ae_cnn2d():
    p4 = get_param_count("AE_CNN2D_GAF", 4, 90)
    p24 = get_param_count("AE_CNN2D_GAF", 24, 90)

    fig, ax = plt.subplots(figsize=(16.5, 8.0), dpi=300)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-2.2, 5.6)
    ax.axis("off")

    ax.text(8.0, 5.15, "Arquitetura 4: AE_CNN2D_GAF (Autoencoder Convolucional 2D com GADF)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(8.0, 4.70, f"Imagens GADF (32×32) com Reconstrução Cruzada (2D + 1D) | Z = 32 | Parâmetros (L=90): {p4} (C=4) / {p24} (C=24)", ha="center", va="center", fontsize=9.8, color="#475569")

    draw_box(ax, 1.0, 3.05, 1.7, 1.45, "Entrada 2D GADF", "PAA(32) → Somente GADF\n(float16 [-1, 1])", "(B, C, 32, 32)", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 1.85, 3.05, 2.45, 3.05)

    draw_box(ax, 3.35, 3.05, 1.8, 1.45, "Conv2D Bloco 1", "16 filtros, k=3, s=2, p=1\nBatchNorm2d + GELU", "(B, 16, 16, 16)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 4.25, 3.05, 4.85, 3.05)

    draw_box(ax, 5.75, 3.05, 1.8, 1.45, "Conv2D Bloco 2", "32 filtros, k=3, s=2, p=1\nBatchNorm2d + GELU", "(B, 32, 8, 8)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 6.65, 3.05, 7.25, 3.05)

    draw_box(ax, 8.15, 3.05, 1.8, 1.45, "AdaptPool2D + Flat", "AdaptiveAvgPool2d(4,4)\nFlatten (32×4×4)", "(B, 512)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 9.05, 3.05, 9.65, 3.05)

    draw_box(ax, 10.5, 3.05, 1.7, 1.6, "Espaço Latente Z", "Linear(512 → 32)\n+ GELU", "(B, 32)", C_LATENT, C_LATENT_BD, lw=2.2)

    # Ramo Superior: Decoder 2D
    draw_arrow(ax, 11.35, 3.35, 12.1, 3.75)
    draw_box(ax, 13.0, 3.85, 1.8, 1.25, "Decoder2D - FC/Deconv1", "Linear(32→1024)→(16,8,8)\nConvTransp2d k=4,s=2", "(B, 16, 16, 16)", C_DECODER, C_DECODER_BD)
    draw_arrow(ax, 13.9, 3.85, 14.45, 3.85)
    draw_box(ax, 15.3, 3.85, 1.65, 1.25, "Saída GADF 2D", "ConvTransp2d(16→C)\nk=4,s=2 + Tanh", "(B, C, 32, 32)", C_DECODER, C_DECODER_BD)

    # Ramo Inferior: Cabeça de Reconstrução Cruzada 1D
    draw_arrow(ax, 11.35, 2.75, 12.1, 2.15)
    draw_box(ax, 13.0, 2.05, 1.8, 1.15, "Decoder1D (Cruzado)", "Linear(32 → 64) + GELU\nLinear(64 → C×L) + Tanh", "(B, C, L)", C_DECODER, C_DECODER_BD)
    draw_arrow(ax, 13.9, 2.05, 14.45, 2.05)
    draw_box(ax, 15.3, 2.05, 1.65, 1.15, "Loss Combinada", "MSE_2D + MSE_1D\nScore unificado", "Reconstrução", C_OUTPUT, C_OUTPUT_BD)

    draw_decisions_panel(ax, 8.0, -0.65, 15.6, 2.35, "Decisões de Arquitetura (AE_CNN2D_GAF - Somente GADF)", [
        "Codificação Polar GADF (sin(φ_i - φ_j)): Captura diretamente transições, variações direcionais e volatilidade temporal em matrizes 32×32.",
        "Redução de Canais (C em vez de 2C): Usando apenas GADF, cada bomba possui C=4 canais 2D e o Dataset Completo possui C=24 canais concatenados.",
        "Convoluções Transpostas (k=4, s=2, p=1): Escolhido kernel par (4×4) com stride 2 para dobrar exatamente a resolução (8×8 → 16×16 → 32×32) sem artefatos.",
        "Cabeça de Reconstrução Cruzada 1D: Além de reconstruir a imagem GADF 2D (Decoder2D), o vetor Z=32 reconstrói o sinal 1D original (Decoder1D)."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_AE_CNN2D_GAF.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 5. ARQUITETURAS HÍBRIDAS MULTIMODAIS (5, 6 e 7)
# ==============================================================================
def gerar_diagrama_hibridos():
    p_cnn = get_param_count("Hybrid_CNN1D_CNN2D", 4, 90)
    p_lstm = get_param_count("Hybrid_LSTM1D_CNN2D", 4, 90)
    p_trf = get_param_count("Hybrid_Transformer_CNN2D", 4, 90)

    fig, ax = plt.subplots(figsize=(16.5, 8.2), dpi=300)
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-2.2, 5.8)
    ax.axis("off")

    ax.text(8.0, 5.35, "Arquiteturas 5, 6 e 7: Modelos Híbridos Multimodais (Fusão 1D + 2D GADF)", ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(8.0, 4.90, f"Hybrid_CNN1D_CNN2D ({p_cnn} params)  |  Hybrid_LSTM1D_CNN2D ({p_lstm} params)  |  Hybrid_Transformer_CNN2D ({p_trf} params) [C=4, L=90]", ha="center", va="center", fontsize=9.5, color="#475569")

    # Ramo Superior: Encoder 1D
    draw_box(ax, 1.3, 3.8, 2.0, 1.3, "Entrada 1D", "Sinal Temporal Bruto", "(B, C, L)", C_INPUT, C_INPUT_BD)
    draw_arrow(ax, 2.3, 3.8, 3.1, 3.8)
    draw_box(ax, 4.4, 3.8, 2.6, 1.3, "Encoder 1D Modular", "CNN1D  ou  LSTM1D\nou  Transformer1D", "z1: (B, 32)", C_ENCODER, C_ENCODER_BD)

    # Ramo Inferior: Encoder 2D GADF
    draw_box(ax, 1.3, 1.9, 2.0, 1.3, "Entrada 2D GADF", "Somente GADF (32×32)", "(B, C, 32, 32)", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 2.3, 1.9, 3.1, 1.9)
    draw_box(ax, 4.4, 1.9, 2.6, 1.3, "Encoder 2D (CNN2D)", "Conv2D(16→32) + Pool(4,4)\n+ Linear(512→32)", "z2: (B, 32)", C_SPECIAL, C_SPECIAL_BD)

    # Convergência para Fusão
    draw_arrow(ax, 5.7, 3.8, 6.7, 3.05)
    draw_arrow(ax, 5.7, 1.9, 6.7, 2.65)
    draw_box(ax, 7.7, 2.85, 2.0, 1.55, "Fusão Multimodal", "Concat([z1, z2]) → (B, 64)\nLinear(64 → 32) + GELU", "Z_fundido: (B, 32)", C_LATENT, C_LATENT_BD, lw=2.2)

    # Divergência para Decodificação Dupla
    draw_arrow(ax, 8.7, 3.25, 9.7, 3.8)
    draw_arrow(ax, 8.7, 2.45, 9.7, 1.9)

    draw_box(ax, 10.9, 3.8, 2.4, 1.3, "Decoder 1D", "Linear(32→64→C×L)\n+ Tanh", "X_rec_1D: (B, C, L)", C_DECODER, C_DECODER_BD)
    draw_box(ax, 10.9, 1.9, 2.4, 1.3, "Decoder 2D", "Linear(32→1024)\n2× ConvTransp2D + Tanh", "X_rec_2D: (B, C, 32, 32)", C_DECODER, C_DECODER_BD)

    draw_arrow(ax, 12.1, 3.8, 13.1, 3.1)
    draw_arrow(ax, 12.1, 1.9, 13.1, 2.6)
    draw_box(ax, 14.5, 2.85, 2.7, 1.55, "Loss Híbrida (MSE)", "0.5 × MSE(1D)\n+ 0.5 × MSE(2D)", "Score de Anomalia\nMultimodal", C_OUTPUT, C_OUTPUT_BD)

    draw_decisions_panel(ax, 8.0, -0.65, 15.6, 2.35, "Decisões de Arquitetura dos Modelos Híbridos (5, 6 e 7)", [
        "Complementaridade 1D + 2D: O ramo 1D preserva dinâmica sequencial exata (amplitudes e tendências); o ramo 2D captura relações angulares par-a-par no tempo.",
        "Bottleneck Unificado (Z=32): A concatenação (32 + 32 = 64) é comprimida de volta para 32 dimensões via Linear(64→32) + GELU, forçando fusão semântica.",
        "Reconstrução Simultânea (Dual-Head): Um único vetor Z=32 precisa ser capaz de reconstruir simultaneamente a série temporal 1D e as imagens GAF 2D.",
        "Flexibilidade Modular: Permite comparar diretamente qual extrator 1D (Convolução, LSTM ou Self-Attention) combina melhor com a visão espacial 2D."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_Hibridos_Multimodais.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. PAINEL COMPARATIVO COMPLETO DAS 7 ARQUITETURAS
# ==============================================================================
def gerar_painel_7_arquiteturas():
    fig, ax = plt.subplots(figsize=(17.5, 9.5), dpi=300)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 10.0)
    ax.axis("off")

    ax.text(8.75, 9.55, "Resumo Comparativo das 7 Arquiteturas Avaliadas (Especificações Reais do Código)", ha="center", va="center", fontsize=14.5, fontweight="bold", color="#0F172A")
    ax.text(8.75, 9.15, "Latent Dim = 32  |  N_Patches = 15  |  Otimizador: AdamW (lr=3e-3, wd=1e-4)  |  Ativações: GELU / ReLU / Tanh", ha="center", va="center", fontsize=10, color="#475569")

    headers = ["Arquitetura", "Modalidade", "Estrutura do Encoder", "Gargalo", "Decoder(s)", "Params (C=4, L=90)", "Params (C=24, L=90)"]
    col_x = [0.3, 3.1, 4.7, 9.1, 10.4, 13.0, 15.2]
    col_w = [2.7, 1.5, 4.3, 1.2, 2.5, 2.1, 2.1]

    # Cabeçalho
    y_top = 8.45
    for cx, cw, h_txt in zip(col_x, col_w, headers):
        box = FancyBboxPatch((cx, y_top - 0.3), cw, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#1E293B", edgecolor="#0F172A")
        ax.add_patch(box)
        ax.text(cx + cw / 2, y_top, h_txt, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF")

    rows = [
        ("AE_CNN1D", "1D", "2× Conv1d(16,32, k=5,s=2) + BN + Pool(8)", "Z = 32", "Decoder1D (32→64→C×L)", get_param_count("AE_CNN1D", 4), get_param_count("AE_CNN1D", 24), C_INPUT),
        ("AE_LSTM1D", "1D", "15 Patches → Linear(32) → Bi-LSTM(32) + Mean", "Z = 32", "Decoder1D (32→64→C×L)", get_param_count("AE_LSTM1D", 4), get_param_count("AE_LSTM1D", 24), C_ENCODER),
        ("AE_Transformer1D", "1D", "15 Patches + Pos → 1× Transf(d=32, H=4, ff=64)", "Z = 32", "Decoder1D (32→64→C×L)", get_param_count("AE_Transformer1D", 4), get_param_count("AE_Transformer1D", 24), C_ENCODER),
        ("AE_CNN2D_GAF", "2D (GAF)", "2× Conv2d(16,32, k=3,s=2) + BN + Pool(4,4)", "Z = 32", "Decoder2D + Decoder1D", get_param_count("AE_CNN2D_GAF", 4), get_param_count("AE_CNN2D_GAF", 24), C_SPECIAL),
        ("Hybrid_CNN1D_CNN2D", "1D + 2D", "EncCNN1D + EncCNN2D → Fusion Linear(64→32)", "Z = 32", "Decoder1D + Decoder2D", get_param_count("Hybrid_CNN1D_CNN2D", 4), get_param_count("Hybrid_CNN1D_CNN2D", 24), C_LATENT),
        ("Hybrid_LSTM1D_CNN2D", "1D + 2D", "EncLSTM1D + EncCNN2D → Fusion Linear(64→32)", "Z = 32", "Decoder1D + Decoder2D", get_param_count("Hybrid_LSTM1D_CNN2D", 4), get_param_count("Hybrid_LSTM1D_CNN2D", 24), C_LATENT),
        ("Hybrid_Transformer_CNN2D", "1D + 2D", "EncTransf1D + EncCNN2D → Fusion Linear(64→32)", "Z = 32", "Decoder1D + Decoder2D", get_param_count("Hybrid_Transformer_CNN2D", 4), get_param_count("Hybrid_Transformer_CNN2D", 24), C_LATENT),
    ]

    for idx, row in enumerate(rows):
        ry = 7.45 - idx * 0.98
        bg_c = row[-1]
        for c_idx, (cx, cw, val) in enumerate(zip(col_x, col_w, row[:-1])):
            box = FancyBboxPatch((cx, ry - 0.36), cw, 0.72, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=bg_c if c_idx == 0 else "#F8FAFC", edgecolor="#CBD5E1", lw=1.0)
            ax.add_patch(box)
            fw = "bold" if c_idx in (0, 3, 5, 6) else "normal"
            ax.text(cx + cw / 2, ry, val, ha="center", va="center", fontsize=8.2, fontweight=fw, color="#0F172A")

    ax.text(8.75, 0.45, "Nota: Todos os modelos compartilham exatamente o mesmo espaço latente (Z = 32) para garantir comparação justa de capacidade representacional entre paradigmas.",
            ha="center", va="center", fontsize=9.0, color="#334155", style="italic")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "arquitetura_painel_comparativo_7_modelos.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 7. FLUXO DE TRABALHO END-TO-END E DECISÕES METODOLÓGICAS
# ==============================================================================
def gerar_fluxo_trabalho():
    fig, ax = plt.subplots(figsize=(17.0, 8.0), dpi=300)
    ax.set_xlim(-0.4, 16.6)
    ax.set_ylim(-2.1, 5.4)
    ax.axis("off")

    ax.text(8.1, 4.95, "Fluxo de Trabalho Completo e Decisões Metodológicas do Pipeline", ha="center", va="center", fontsize=14.5, fontweight="bold", color="#0F172A")
    ax.text(8.1, 4.50, "Da coleta no Historiador PI (10s) até o Treinamento Não Supervisionado em GPU e Detecção de Anomalias", ha="center", va="center", fontsize=10, color="#475569")

    # Etapa 1
    draw_box(ax, 1.2, 2.8, 2.1, 1.7, "1. Dados Brutos PI", "6 Bombas × 4 Sensores\nTaxa CSV: 10s (0,1 Hz)\nHardware: 961,5 Hz", "Jan–Out 2026", C_INPUT, C_INPUT_BD)
    draw_arrow(ax, 2.25, 2.8, 2.85, 2.8)

    # Etapa 2
    draw_box(ax, 3.9, 2.8, 2.1, 1.7, "2. Janelamento", "Escalas: 15m, 30m, 1h, 2h\nL = 90, 180, 360, 720\nDivisíveis por 15 patches", "N = 57.215 (15m)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 4.95, 2.8, 5.55, 2.8)

    # Etapa 3
    draw_box(ax, 6.6, 2.8, 2.1, 1.7, "3. Validação Cruzada", "StratifiedGroupKFold\nSomente Rótulo (0/1)\nBloqueio anti-vazamento", "5 Folds (block_id)", C_ENCODER, C_ENCODER_BD)
    draw_arrow(ax, 7.65, 2.8, 8.25, 2.8)

    # Etapa 4
    draw_box(ax, 9.3, 2.8, 2.1, 1.7, "4. Representações", "1D: (B, C, L) [-1, 1]\n2D: PAA(32) → GADF\nfloat16 (C×32×32)", "1D + 2D GADF", C_SPECIAL, C_SPECIAL_BD)
    draw_arrow(ax, 10.35, 2.8, 10.95, 2.8)

    # Etapa 5
    draw_box(ax, 12.0, 2.8, 2.1, 1.7, "5. Treino GPU (CUDA)", "7 Arquiteturas AE\nDataset_Completo Concat\nAdamW + AMP float16", "Z = 32 | MSE Loss", C_LATENT, C_LATENT_BD, lw=2.0)
    draw_arrow(ax, 13.05, 2.8, 13.65, 2.8)

    # Etapa 6
    draw_box(ax, 14.8, 2.8, 2.3, 1.7, "6. Inferência & Score", "Erro ||X - X_rec||²\nLimiar ROC / P95 / P99\nvs Critério ≥ 2,5 A", "Avaliação 5 Folds", C_DECODER, C_DECODER_BD)

    draw_decisions_panel(ax, 8.1, -0.55, 15.8, 2.35, "Justificativas das Principais Decisões do Fluxo de Trabalho", [
        "Dataset Completo por Concatenação: Constrói os 24 canais concatenando as 6 bombas individuais em memória, eliminando duplicação de arquivos .npz em disco.",
        "StratifiedGroupKFold por Rótulo (5 Folds): Agrupa por block_id (evitando data leakage temporal) e estratifica diretamente pelo rótulo de anomalia (0 ou 1).",
        "Uso Exclusivo do GADF (32×32 em float16): Reduz pela metade os canais 2D (de 2C para C), focando nas diferenças angulares (transições dinâmicas de corrente).",
        "Treinamento em GPU com AMP (float16): Otimiza os 7 modelos não supervisionados e híbridos com precisão mista na GPU NVIDIA RTX 5080."
    ])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fluxo_trabalho_e_decisoes_arquitetura.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Gerando diagramas completos (7 arquiteturas + fluxo de trabalho)...")
    gerar_ae_cnn1d()
    gerar_ae_lstm1d()
    gerar_ae_transformer1d()
    gerar_ae_cnn2d()
    gerar_diagrama_hibridos()
    gerar_painel_7_arquiteturas()
    gerar_fluxo_trabalho()
    print(f"Concluído! Todos os arquivos salvos em: {OUTPUT_DIR}")
