"""
Gerador Automático do Site Científico para GitHub Pages (`docs/`).

Este script lê dinamicamente os CSVs e figuras mais recentes da pasta
`Dataset_completo_bombas/Resultados_Modelos_1D_2D/Deteccao_Anomalias_Operador/`
e `Dataset_completo_bombas/Figuras_Arquiteturas/`, construindo um portal web
completo (`docs/index.html`, `docs/assets/style.css`, `docs/assets/app.js`,
`docs/data/resultados.json`) contendo:

1. Seção de Texto Acadêmico em Formato de Dissertação (Capítulos 1 a 6),
   contextualizando todo o problema industrial, fundamentação matemática,
   transformada GADF + PAA, validação cruzada sem data leakage, as 7 arquiteturas
   em PyTorch (torchviz + torchinfo.summary), as 11 formulações de rótulos e a
   triagem por histograma do erro de reconstrução com revisão do operador (FP/FN).
2. Seção Interativa de Resultados Experimentais (tabelas dinâmicas, gráficos
   Chart.js, filtros por bomba/janela e galeria de figuras em alta resolução).
"""

import os
import json
import shutil
from datetime import datetime
import pandas as pd

WORKSPACE = os.path.abspath(".")
DOCS_DIR = os.path.join(WORKSPACE, "docs")
IMG_OUT_DIR = os.path.join(DOCS_DIR, "assets", "img")
DATA_OUT_DIR = os.path.join(DOCS_DIR, "data")

ANOM_DIR = os.path.join(
    WORKSPACE,
    "Dataset_completo_bombas",
    "Resultados_Modelos_1D_2D",
    "Deteccao_Anomalias_Operador",
)
FIG_ANOM_DIR = os.path.join(ANOM_DIR, "Figuras_Anomalias_Operador")
FIG_ARCH_DIR = os.path.join(WORKSPACE, "Dataset_completo_bombas", "Figuras_Arquiteturas")
ART_DIR = r"C:\Users\202510482\.gemini\antigravity\brain\820f487b-8644-4f3d-b47e-9fd7fab98fec"


def load_csv_records(filename: str, fallback_root: bool = True) -> list[dict]:
    p1 = os.path.join(ANOM_DIR, filename)
    p2 = os.path.join(WORKSPACE, filename)
    path = p1 if os.path.isfile(p1) else (p2 if (fallback_root and os.path.isfile(p2)) else None)
    if path is None:
        return []
    df = pd.read_csv(path)
    return df.fillna("").to_dict(orient="records")


def copy_figures() -> list[str]:
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    copied = []
    candidates = [
        (FIG_ARCH_DIR, "fluxo_trabalho_e_decisoes_arquitetura.png"),
        (FIG_ARCH_DIR, "arquitetura_painel_comparativo_7_modelos.png"),
        (FIG_ARCH_DIR, "arquitetura_AE_CNN1D.png"),
        (FIG_ARCH_DIR, "arquitetura_AE_LSTM1D.png"),
        (FIG_ARCH_DIR, "arquitetura_AE_Transformer1D.png"),
        (FIG_ARCH_DIR, "arquitetura_AE_CNN2D_GAF.png"),
        (FIG_ARCH_DIR, "arquitetura_Hybrid_CNN1D_CNN2D.png"),
        (FIG_ARCH_DIR, "arquitetura_Hybrid_LSTM1D_CNN2D.png"),
        (FIG_ARCH_DIR, "arquitetura_Hybrid_Transformer_CNN2D.png"),
        (FIG_ARCH_DIR, "arquitetura_Hibridos_Multimodais.png"),
        (FIG_ANOM_DIR, "matrizes_confusao_ano_acima_2_5_15min.png"),
        (FIG_ANOM_DIR, "comparacao_11_abordagens_rotulos.png"),
        (FIG_ANOM_DIR, "histogramas_erro_reconstrucao_zona_operador.png"),
        (FIG_ANOM_DIR, "comparacao_histograma_corte_unico_vs_operador.png"),
        (FIG_ANOM_DIR, "curvas_perda_loss_7_arquiteturas.png"),
        (FIG_ANOM_DIR, "curvas_perda_loss_por_bomba.png"),
        (ART_DIR, "comparacao_11_abordagens_rotulos.png"),
        (ART_DIR, "matrizes_confusao_ano_acima_2_5_15min.png"),
        (ART_DIR, "heatmap_rmse_bombas_vs_modelos.png"),
        (ART_DIR, "comparacao_arquiteturas_por_janela.png"),
    ]
    for src_dir, fname in candidates:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(IMG_OUT_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            if fname not in copied:
                copied.append(fname)
    return copied


def build_site():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)

    ranking_7 = load_csv_records("ranking_global_deteccao_ano_acima_2_5.csv")
    detalhe_28 = load_csv_records("resumo_deteccao_ano_acima_2_5_por_bomba_janela.csv")
    abordagens_11 = load_csv_records("resumo_global_11_abordagens_rotulos.csv")
    hist_rank = load_csv_records("resumo_histograma_erro_zona_operador_arquiteturas.csv")
    hist_det = load_csv_records("avaliacao_histograma_erro_zona_operador_detalhado.csv")
    janelas_info = load_csv_records("resumo_janelas_anomalias_ano_acima_2_5.csv")

    copied_imgs = copy_figures()
    updated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    payload = {
        "updated_at": updated_at,
        "ranking_7_arquiteturas": ranking_7,
        "detalhe_bomba_janela": detalhe_28,
        "abordagens_11_rotulos": abordagens_11,
        "histograma_operador_ranking": hist_rank,
        "histograma_operador_detalhado": hist_det,
        "distribuicao_janelas": janelas_info,
        "figures": copied_imgs,
    }

    with open(os.path.join(DATA_OUT_DIR, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Gera tabelas HTML estáticas para carregamento instantâneo e indexação
    def rows_ranking_7_html():
        out = []
        for idx, r in enumerate(ranking_7, 1):
            out.append(
                f"<tr>"
                f"<td><strong>#{idx}</strong></td>"
                f"<td><code>{r.get('architecture','')}</code></td>"
                f"<td><strong>{float(r.get('roc_auc_hybrid_mean',0)):.4f}</strong></td>"
                f"<td><strong>{float(r.get('pr_auc_hybrid_mean',0)):.4f}</strong></td>"
                f"<td><strong>{float(r.get('best_f1_hybrid_mean',0)):.4f}</strong></td>"
                f"<td>{float(r.get('precision_hybrid_mean',0)):.4f}</td>"
                f"<td>{float(r.get('recall_hybrid_mean',0)):.4f}</td>"
                f"<td>{float(r.get('accuracy_hybrid_mean',0))*100:.2f}%</td>"
                f"<td>{float(r.get('false_alarm_rate_pct_mean',0)):.2f}%</td>"
                f"<td>{float(r.get('roc_auc_unsup_mean',0)):.4f}</td>"
                f"<td><strong>{float(r.get('unsup_score_ratio_mean',0)):.2f}×</strong></td>"
                f"</tr>"
            )
        return "\n".join(out)

    def rows_11_approaches_html():
        out = []
        for r in abordagens_11:
            out.append(
                f"<tr>"
                f"<td><strong>{r.get('approach_label','')}</strong></td>"
                f"<td><span class='badge'>{r.get('mode','')}</span></td>"
                f"<td><strong>{float(r.get('roc_auc_mean',0)):.4f}</strong></td>"
                f"<td>{float(r.get('pr_auc_mean',0)):.4f}</td>"
                f"<td><strong>{float(r.get('best_f1_mean',0)):.4f}</strong></td>"
                f"<td>{float(r.get('precision_mean',0)):.4f}</td>"
                f"<td>{float(r.get('recall_mean',0)):.4f}</td>"
                f"<td>{float(r.get('false_alarm_rate_pct_mean',0)):.2f}%</td>"
                f"<td>{float(r.get('prob_mae_pct_mean',0)):.2f}%</td>"
                f"<td><strong>{float(r.get('spearman_corr_mean',0)):.4f}</strong></td>"
                f"</tr>"
            )
        return "\n".join(out)

    def rows_hist_operator_html():
        if not hist_rank:
            return "<tr><td colspan='9'>Execução do experimento de histograma em andamento na GPU...</td></tr>"
        out = []
        for idx, r in enumerate(hist_rank, 1):
            out.append(
                f"<tr>"
                f"<td><strong>#{idx}</strong></td>"
                f"<td><code>{r.get('architecture','')}</code></td>"
                f"<td>{float(r.get('roc_auc_unsup_mean',0)):.4f}</td>"
                f"<td>{float(r.get('single_f1_mean',0)):.4f}</td>"
                f"<td>{float(r.get('auto_only_f1_mean',0)):.4f}</td>"
                f"<td><strong style='color:#15803d'>{float(r.get('hitl_f1_mean',0)):.4f}</strong></td>"
                f"<td>{float(r.get('hitl_far_pct_mean',0)):.2f}%</td>"
                f"<td><strong>{float(r.get('pct_fp_fn_captured_mean',0)):.1f}%</strong></td>"
                f"<td>{float(r.get('pct_operator_review_mean',0)):.1f}%</td>"
                f"</tr>"
            )
        return "\n".join(out)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Detecção e Prognóstico de Anomalias em Bombas de Polpa Industriais — Portal Científico de Resultados Experimentais</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --primary: #0f172a;
      --accent: #1d4ed8;
      --accent-soft: #dbeafe;
      --text: #1e293b;
      --muted: #475569;
      --border: #e2e8f0;
      --success: #15803d;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.72;
    }}
    header.hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: #ffffff;
      padding: 3.2rem 1.5rem 2.6rem;
      border-bottom: 4px solid #3b82f6;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 0 1.25rem; }}
    .hero h1 {{ font-size: 2.05rem; font-weight: 800; margin-bottom: 0.75rem; line-height: 1.25; }}
    .hero p {{ color: #cbd5e1; font-size: 1.05rem; max-width: 960px; }}
    .meta-bar {{
      margin-top: 1.3rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      font-size: 0.85rem;
    }}
    .chip {{
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.22);
      padding: 0.32rem 0.75rem;
      border-radius: 999px;
      color: #f8fafc;
    }}
    nav.topbar {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(255,255,255,0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }}
    nav.topbar .nav-inner {{
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
      padding: 0.75rem 0;
    }}
    nav.topbar a {{
      text-decoration: none;
      color: var(--primary);
      font-weight: 600;
      font-size: 0.9rem;
      padding: 0.45rem 0.95rem;
      border-radius: 8px;
      white-space: nowrap;
      transition: all 0.15s;
    }}
    nav.topbar a:hover, nav.topbar a.active {{
      background: var(--accent);
      color: #ffffff;
    }}
    section.block {{
      margin: 2.5rem 0;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 2.2rem;
      box-shadow: 0 4px 16px rgba(15,23,42,0.03);
    }}
    .section-tag {{
      display: inline-block;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      margin-bottom: 0.6rem;
    }}
    h2.sec-title {{
      font-size: 1.65rem;
      color: var(--primary);
      margin-bottom: 1.2rem;
      padding-bottom: 0.6rem;
      border-bottom: 2px solid var(--border);
    }}
    .dissertation h3 {{
      font-size: 1.28rem;
      color: #1e3a8a;
      margin: 1.85rem 0 0.75rem;
    }}
    .dissertation h4 {{
      font-size: 1.05rem;
      color: var(--primary);
      margin: 1.25rem 0 0.45rem;
    }}
    .dissertation p {{
      margin-bottom: 1rem;
      text-align: justify;
      color: #1e293b;
    }}
    .dissertation ul, .dissertation ol {{
      margin: 0.6rem 0 1.1rem 1.6rem;
    }}
    .dissertation li {{ margin-bottom: 0.45rem; }}
    .callout {{
      background: #f1f5f9;
      border-left: 4px solid var(--accent);
      padding: 1.1rem 1.3rem;
      border-radius: 0 10px 10px 0;
      margin: 1.2rem 0;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin: 1.25rem 0 1.8rem;
    }}
    .kpi-card {{
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border: 1px solid var(--border);
      border-top: 4px solid var(--accent);
      border-radius: 12px;
      padding: 1.15rem;
    }}
    .kpi-card .kpi-label {{ font-size: 0.82rem; color: var(--muted); font-weight: 600; }}
    .kpi-card .kpi-val {{ font-size: 1.75rem; font-weight: 800; color: var(--primary); margin: 0.25rem 0; }}
    .kpi-card .kpi-sub {{ font-size: 0.8rem; color: var(--success); font-weight: 600; }}
    .table-wrap {{
      overflow-x: auto;
      margin: 1.2rem 0;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th {{
      background: #0f172a;
      color: #ffffff;
      text-align: left;
      padding: 0.75rem 0.85rem;
      white-space: nowrap;
    }}
    td {{
      padding: 0.68rem 0.85rem;
      border-bottom: 1px solid var(--border);
    }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
    tr:hover td {{ background: #eff6ff; }}
    code {{
      background: #e2e8f0;
      padding: 0.15rem 0.4rem;
      border-radius: 5px;
      font-size: 0.84rem;
    }}
    .badge {{
      background: #e0f2fe;
      color: #0369a1;
      padding: 0.18rem 0.55rem;
      border-radius: 99px;
      font-size: 0.76rem;
      font-weight: 700;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 1.5rem;
      margin: 1.5rem 0;
    }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem;
      background: #ffffff;
    }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1.5rem;
      margin-top: 1.3rem;
    }}
    .fig-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: #ffffff;
    }}
    .fig-card img {{
      width: 100%;
      height: auto;
      display: block;
      cursor: zoom-in;
    }}
    .fig-card .fig-cap {{
      padding: 0.8rem 1rem;
      font-size: 0.86rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      background: #f8fafc;
    }}
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem;
      margin-bottom: 1rem;
      align-items: center;
    }}
    .filter-bar select {{
      padding: 0.48rem 0.85rem;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 0.9rem;
      background: #ffffff;
    }}
    footer {{
      text-align: center;
      padding: 2.2rem 1rem;
      color: var(--muted);
      font-size: 0.88rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }}
  </style>
</head>
<body>

<header class="hero">
  <div class="container">
    <h1>Detecção e Prognóstico de Anomalias em Bombas de Polpa Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais</h1>
    <p>Portal Científico de Resultados Experimentais</p>
    <div class="meta-bar">
      <span class="chip">Framework: PyTorch 2.x + CUDA</span>
      <span class="chip">Hardware: NVIDIA RTX 5080 (16 GB VRAM)</span>
      <span class="chip">Dataset: 6 Bombas × 4 Sensores (Historiador PI, 10s)</span>
      <span class="chip">Validação: 5-Fold StratifiedGroupKFold (Zero Data Leakage)</span>
      <span class="chip">Última atualização automática: {updated_at}</span>
    </div>
  </div>
</header>

<nav class="topbar">
  <div class="container nav-inner">
    <a href="#dissertacao">1. Texto Acadêmico (Dissertação)</a>
    <a href="#arquiteturas">2. Arquiteturas PyTorch (torchviz & summary)</a>
    <a href="#resultados">3. Resultados Experimentais (7 Modelos)</a>
    <a href="#rotulos11">4. Prognóstico e 11 Formulações de Rótulos</a>
    <a href="#histograma-operador">5. Histograma do Erro & Triagem do Operador (FP/FN)</a>
    <a href="#tabela-interativa">6. Explorador por Bomba & Janela</a>
  </div>
</nav>

<main class="container">

  <!-- ===================================================================== -->
  <!-- SEÇÃO 1: TEXTO ACADÊMICO EM FORMATO DE DISSERTAÇÃO                    -->
  <!-- ===================================================================== -->
  <section id="dissertacao" class="block dissertation">
    <span class="section-tag">Seção I — Texto Acadêmico</span>
    <h2 class="sec-title">Contextualização e Fundamentação Científica em Formato de Dissertação</h2>

    <h3>Resumo</h3>
    <p>
      Este trabalho investiga a detecção não supervisionada, a triagem assistida por operador (<em>Human-in-the-Loop</em>) e o prognóstico antecipado de anomalias operacionais em um parque de seis bombas centrífugas industriais (Bombas 1305, 1307, 2307, 3307, 4307 e 5307) monitoradas continuamente ao longo de dez meses (janeiro a outubro de 2026) por meio do sistema Historiador PI com taxa de amostragem de 10 segundos ($0,1\\text{{ Hz}}$). Cada unidade de bombeamento é instrumentada com quatro variáveis de processo: corrente elétrica do motor, vazão volumétrica, pressão de sucção e pressão de descarga. Propõe-se e avalia-se sistematicamente um conjunto de sete arquiteturas neurais baseadas em Autoencoders implementadas nativamente em PyTorch e treinadas em GPU NVIDIA GeForce RTX 5080: três modelos temporais unidimensionais (<code>AE_CNN1D</code>, <code>AE_LSTM1D</code> e <code>AE_Transformer1D</code>), um modelo convolucional bidimensional operando sobre imagens de diferença angular de Gram (<code>AE_CNN2D_GAF</code> utilizando exclusivamente matrizes GADF reduzidas por <em>Piecewise Aggregate Approximation</em> para $32\\times 32$) e três arquiteturas híbridas multimodais (<code>Hybrid_CNN1D_CNN2D</code>, <code>Hybrid_LSTM1D_CNN2D</code> e <code>Hybrid_Transformer_CNN2D</code>) que fundem representações temporais 1D e angulares 2D em um espaço latente compartilhado $\\mathbf{{z}} \\in \\mathbb{{R}}^{{32}}$.
    </p>

    <h3>Capítulo 1 — Introdução e Contextualização do Problema Industrial</h3>
    <p>
      Bombas centrífugas de grande porte constituem ativos críticos em plantas de processamento contínuo. Falhas súbitas por cavitação, sobrecarga elétrica, obstrução de linha ou desgaste de impelidores acarretam paradas não programadas de alto custo e riscos de segurança operacional. Embora sistemas supervisórios industriais (como o OSIsoft PI Historian) armazenem milhões de registros multivariados ao longo do ano, a grande maioria dos dados corresponde a regimes normais de operação, enquanto eventos de falha ou instabilidade severa são raros, transitórios e frequentemente carecem de rotulagem fina instante a instante.
    </p>
    <p>
      Nesse cenário, o paradigma de <strong>Autoencoders de Reconstrução</strong> destaca-se por permitir que a rede neural aprenda a variedade (<em>manifold</em>) de operação normal exclusivamente a partir de janelas saudáveis. Quando submetido a uma janela temporal contendo distúrbios hidrodinâmicos ou elétricos, o modelo falha em reconstruir fielmente os padrões fora da distribuição normal, produzindo desvios residuais elevados que servem tanto para o diagnóstico automático quanto para a triagem de casos limítrofes (Falsos Positivos e Falsos Negativos) junto ao operador da planta.
    </p>

    <h3>Capítulo 2 — Fundamentação Teórica: Representações 1D, Transformada GADF 2D e Compressão PAA</h3>
    <h4>2.1 Janelamento Multiescala e Normalização</h4>
    <p>
      A série temporal contínua $\\mathbf{{X}} \\in \\mathbb{{R}}^{{C \\times T}}$ ($C=4$ sensores por bomba individual ou $C=24$ canais na concatenação dinâmica em memória do <code>Dataset_Completo</code>) é segmentada em janelas deslizantes de comprimento $L$ com sobreposição (<em>overlap</em>) de $50\\%$, contemplando quatro escalas operacionais: 15 minutos ($L=90$), 30 minutos ($L=180$), 1 hora ($L=360$) e 2 horas ($L=720$). Previamente à transformação angular, cada canal é reescalonado para o intervalo $[-1, 1]$ utilizando os limites calibrados exclusivamente nos folds de treinamento:
    </p>
    $$\\tilde{{x}}_t = \\text{{clip}}\\left(2 \\cdot \\frac{{x_t - x_{{\\min}}}}{{x_{{\\max}} - x_{{\\min}}}} - 1,\\ -1,\\ 1\\right)$$

    <h4>2.2 Uso Exclusivo do Gramian Angular Difference Field (GADF) e Redução PAA</h4>
    <p>
      Para converter sinais unidimensionais em representações matriciais bidimensionais capazes de expor correlações temporais par-a-par para redes convolucionais 2D, aplica-se inicialmente o <em>Piecewise Aggregate Approximation</em> (PAA), reduzindo qualquer janela de comprimento $L \\in \\{{90, 180, 360, 720\\}}$ para $M = 32$ segmentos temporais médios. Em seguida, projeta-se cada valor normalizado em coordenadas polares $\\phi_i = \\arccos(\\tilde{{x}}_i)$ e calcula-se exclusivamente a matriz <strong>GADF</strong> (<em>Gramian Angular Difference Field</em>):
    </p>
    $$\\text{{GADF}}_{{i,j}} = \\sin(\\phi_i - \\phi_j) = \\tilde{{x}}_j \\sqrt{{1 - \\tilde{{x}}_i^2}} - \\tilde{{x}}_i \\sqrt{{1 - \\tilde{{x}}_j^2}} \\in \\mathbb{{R}}^{{32 \\times 32}}$$
    <div class="callout">
      <strong>Justificativa Teórica e Computacional do PAA ($32\\times 32$) + GADF Exclusivo:</strong>
      <ul>
        <li><strong>Superioridade Física do GADF sobre o GASF:</strong> Enquanto o GASF ($\\cos(\\phi_i + \\phi_j)$) enfatiza níveis estáticos de carga, o GADF ($\\sin(\\phi_i - \\phi_j)$) captura diretamente derivadas e transientes abruptos entre instantes distintos — exatamente a assinatura física de picos de corrente e instabilidades hidráulicas.</li>
        <li><strong>Viabilidade de Armazenamento e VRAM:</strong> Sem o PAA, as 2.574.624 imagens GADF cresceriam quadraticamente ($L \\times L$, chegando a $720 \\times 720$ na escala de 2h), exigindo <strong>310,74 GiB</strong> brutos em <code>float16</code> (ou <strong>121,1 GiB</strong> mesmo comprimidos em <code>.npz</code>). Com o PAA em $32\\times 32$, a remoção do GASF redundante e a compactação <code>np.savez_compressed</code>, todo o conjunto de 24 arquivos ocupa apenas <strong>2,89 GiB</strong> em disco (redução superior a <strong>97%</strong>), mantendo resolução uniforme para a CNN 2D.</li>
      </ul>
    </div>

    <h3>Capítulo 3 — Metodologia Experimental, Validação Cruzada e Prevenção de Data Leakage</h3>
    <h4>3.1 Critério Operacional de Anomalia Anual ($\\text{{StdDev}} \\ge 2,5\\text{{ A}}$)</h4>
    <p>
      A referência de verdade terrestre (<em>ground truth</em>) do operador baseia-se no desvio padrão amostral da corrente elétrica calculado em blocos sucessivos de 5 minutos ao longo de todo o ano. Blocos cujo desvio padrão atinge ou supera $2,5\\text{{ A}}$ sinalizam excursões severas de corrente associadas a distúrbios operacionais, sendo mapeados para as janelas deslizantes correspondentes.
    </p>
    <h4>3.2 Validação Cruzada Estratificada por Rótulo Agrupada por Blocos com Exclusão de Fronteira</h4>
    <p>
      Como janelas consecutivas possuem $50\\%$ de sobreposição temporal, uma divisão aleatória convencional causaria vazamento de dados (<em>data leakage</em>), pois metade de uma janela de teste estaria presente em uma janela adjacente de treino. Para eliminar rigorosamente esse efeito:
    </p>
    <ol>
      <li>Utiliza-se o <code>StratifiedGroupKFold(n_splits=5)</code> estratificando <strong>estritamente pelo rótulo binário de anomalia</strong> ($y \\in \\{{0, 1\\}}$) e agrupando por blocos temporais contíguos (<code>block_id</code>), garantindo que todos os 5 folds possuam a mesma proporção de anomalias sem fragmentar episódios contíguos.</li>
      <li>Aplica-se uma máscara dinâmica de exclusão de fronteira (<code>clean_mask</code>): toda janela $w$ cujo fold difere do fold da janela imediatamente anterior ($w-1$) ou posterior ($w+1$) é descartada (~0,85% das janelas), assegurando independência temporal absoluta entre treino e validação.</li>
    </ol>

    <h3>Capítulo 4 — Decisão Pós-Reconstrução, Prognóstico Futuro e Triagem pelo Histograma com Operador</h3>
    <h4>4.1 Do Erro de Reconstrução à Decisão Operacional</h4>
    <p>
      Dado o sinal reconstruído $\\hat{{\\mathbf{{X}}}}_{{1D}}$ e a matriz reconstruída $\\hat{{\\mathbf{{X}}}}_{{\\text{{GADF}}}}$, extraem-se os resíduos quadráticos médios ($\\text{{MSE}}_{{1D}}$ e $\\text{{MSE}}_{{\\text{{GADF}}}}$) bem como estatísticas residuais por canal físico. No modo puramente não supervisionado, o erro de cada janela é padronizado contra a média $\\mu_{{\\text{{norm}}}}$ e o desvio padrão $\\sigma_{{\\text{{norm}}}}$ medidos exclusivamente nas janelas normais de treinamento:
    </p>
    $$E(x) = \\frac{{1}}{{K}} \\sum_{{k=1}}^{{K}} \\frac{{|\\text{{Res}}_k(x) - \\mu_{{k,\\text{{norm}}}}|}}{{\\sigma_{{k,\\text{{norm}}}} + \\epsilon}}$$

    <h4>4.2 Separação por Histograma do Erro e Zona Intermediária do Operador (Human-in-the-Loop)</h4>
    <p>
      Ao construir o histograma empírico de $E(x)$ para janelas normais e anômalas, observa-se uma distribuição bimodal com uma região central de sobreposição onde ocorrem os <strong>Falsos Positivos (FP)</strong> e <strong>Falsos Negativos (FN)</strong> de um limiar rígido único $\\tau_{{\\text{{hist}}}}$. Em vez de forçar uma decisão binária cega nessa zona cinzenta, definem-se dois limiares no histograma ($\\tau_{{\\text{{baixo}}}} < \\tau_{{\\text{{hist}}}} < \\tau_{{\\text{{alto}}}}$), dividindo a operação em três zonas:
    </p>
    <ul>
      <li><strong>Zona 1 — Normal Automático ($E(x) < \\tau_{{\\text{{baixo}}}}$):</strong> Decisão automática de operação normal com altíssima confiança.</li>
      <li><strong>Zona 2 — Zona Intermediária de Triagem Humana ($\\tau_{{\\text{{baixo}}}} \\le E(x) \\le \\tau_{{\\text{{alto}}}}$):</strong> Região de incerteza que concentra a vasta maioria dos FP e FN; as janelas desta faixa são encaminhadas para avaliação visual/contextual do operador.</li>
      <li><strong>Zona 3 — Anomalia Confirmada Automática ($E(x) > \\tau_{{\\text{{alto}}}}$):</strong> Disparo automático de alarme com altíssima precisão.</li>
    </ul>

    <h4>4.3 Onze Formulações de Rótulos para Prognóstico e Distribuição Contínua de Probabilidade</h4>
    <p>
      Além da detecção instantânea, avalia-se a capacidade preditiva dos modelos sob 11 formulações distintas: (i) detecção instantânea binária; (ii) previsão geral de horizonte futuro ($H \\in \\{{15, 30, 60\\}}\\text{{ min}}$); (iii) <strong>alerta precoce puro</strong> (avaliado estritamente quando a bomba ainda opera em estado 100% normal no presente, $y_{{\\text{{atual}}}}=0$, mas sofrerá anomalia nos próximos $H$ minutos); e (iv) <strong>distribuição contínua de probabilidade</strong> com kernel gaussiano temporal assimétrico ($\\tau_{{\\text{{antes}}}}, \\tau_{{\\text{{depois}}}}$), atribuindo uma porcentagem gradual de risco antes e depois do pico marcado pelo operador.
    </p>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 2: ARQUITETURAS PYTORCH (TORCHVIZ + TORCHINFO.SUMMARY)          -->
  <!-- ===================================================================== -->
  <section id="arquiteturas" class="block">
    <span class="section-tag">Seção II — Arquiteturas em PyTorch</span>
    <h2 class="sec-title">Grafos Computacionais (torchviz) e Resumo Estrutural (torchinfo.summary)</h2>
    <p style="margin-bottom:1rem;color:var(--muted);">
      Todas as 7 arquiteturas compartilham o mesmo gargalo latente ($Z = 32$) para comparação justa. Clique em qualquer figura para ampliá-la em tela cheia.
    </p>
    <div class="fig-grid">
      <div class="fig-card">
        <img src="assets/img/fluxo_trabalho_e_decisoes_arquitetura.png" alt="Fluxo Metodológico" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 1:</strong> Fluxo de Trabalho End-to-End e Decisões Metodológicas do Pipeline.</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_painel_comparativo_7_modelos.png" alt="Painel 7 Modelos" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 2:</strong> Painel Comparativo das 7 Arquiteturas Avaliadas ($Z = 32$).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_AE_CNN1D.png" alt="AE_CNN1D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 3:</strong> Arquitetura 1 — <code>AE_CNN1D</code> (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_AE_LSTM1D.png" alt="AE_LSTM1D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 4:</strong> Arquitetura 2 — <code>AE_LSTM1D</code> Patch-LSTM Bidirecional (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_AE_Transformer1D.png" alt="AE_Transformer1D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 5:</strong> Arquitetura 3 — <code>AE_Transformer1D</code> Patch-Transformer (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_AE_CNN2D_GAF.png" alt="AE_CNN2D_GAF" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 6:</strong> Arquitetura 4 — <code>AE_CNN2D_GAF</code> Exclusivo GADF $32\\times 32$ (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_Hybrid_CNN1D_CNN2D.png" alt="Hybrid_CNN1D_CNN2D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 7:</strong> Arquitetura 5 — <code>Hybrid_CNN1D_CNN2D</code> Multimodal (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_Hybrid_LSTM1D_CNN2D.png" alt="Hybrid_LSTM1D_CNN2D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 8:</strong> Arquitetura 6 — <code>Hybrid_LSTM1D_CNN2D</code> Multimodal (torchviz + torchinfo.summary).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/arquitetura_Hybrid_Transformer_CNN2D.png" alt="Hybrid_Transformer_CNN2D" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 9:</strong> Arquitetura 7 — <code>Hybrid_Transformer_CNN2D</code> Multimodal (torchviz + torchinfo.summary).</div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 3: RESULTADOS EXPERIMENTAIS GLOBAIS (7 ARQUITETURAS)            -->
  <!-- ===================================================================== -->
  <section id="resultados" class="block">
    <span class="section-tag">Seção III — Benchmark em GPU</span>
    <h2 class="sec-title">Resultados Globais das 7 Arquiteturas (100% Dataset Anual, StdDev ≥ 2,5 A)</h2>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Melhor F1-Score & PR-AUC</div>
        <div class="kpi-val">0.8736 / 0.9366</div>
        <div class="kpi-sub">Hybrid_Transformer_CNN2D</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Maior ROC-AUC Global</div>
        <div class="kpi-val">0.9808</div>
        <div class="kpi-sub">AE_LSTM1D (Acurácia 94,94%)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Razão de Erro Não Supervisionada</div>
        <div class="kpi-val">13,22×</div>
        <div class="kpi-sub">Hybrid_CNN1D_CNN2D (até 30,3× na B1307)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Redução de Volume GADF c/ PAA</div>
        <div class="kpi-val">2,89 GiB</div>
        <div class="kpi-sub">vs. 310,74 GiB sem PAA (−99,1%)</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <h4 style="margin-bottom:0.8rem;color:var(--primary);">Desempenho por Arquitetura (ROC-AUC, PR-AUC e F1-Score)</h4>
        <canvas id="chartArch7" height="220"></canvas>
      </div>
      <div class="chart-card">
        <h4 style="margin-bottom:0.8rem;color:var(--primary);">Razão de Score de Anomalia Não Supervisionado (Anômalo / Normal)</h4>
        <canvas id="chartUnsupRatio" height="220"></canvas>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Arquitetura</th>
            <th>ROC-AUC (Híbrido)</th>
            <th>PR-AUC (Híbrido)</th>
            <th>Melhor F1</th>
            <th>Precisão</th>
            <th>Recall</th>
            <th>Acurácia</th>
            <th>Falsos Alarmes</th>
            <th>ROC-AUC (Não Sup.)</th>
            <th>Razão Score (Não Sup.)</th>
          </tr>
        </thead>
        <tbody>
          {rows_ranking_7_html()}
        </tbody>
      </table>
    </div>

    <div class="fig-grid">
      <div class="fig-card">
        <img src="assets/img/curvas_perda_loss_7_arquiteturas.png" alt="Curvas de Perda Loss 7 Arquiteturas" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 10:</strong> Curvas de Perda (Loss de Reconstrução MSE) no Treinamento vs. Validação (±1 DP nos 5 Folds) para as 7 Arquiteturas em GPU.</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/curvas_perda_loss_por_bomba.png" alt="Curvas de Perda Loss por Bomba" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 11:</strong> Convergência da Perda (Loss de Treino vs. Validação) individualizada para cada Bomba e para o Dataset Completo.</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/matrizes_confusao_ano_acima_2_5_15min.png" alt="Matrizes de Confusão" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 12:</strong> Matrizes de Confusão Out-of-Fold (5 Folds) em 100% do Dataset Anual.</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/comparacao_arquiteturas_por_janela.png" alt="Comparação por Janela" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 13:</strong> Desempenho comparativo das arquiteturas através das 4 escalas de janela temporal.</div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 4: 11 FORMULAÇÕES DE RÓTULOS (FUTURO, ALERTA PRECOCE, SOFT)     -->
  <!-- ===================================================================== -->
  <section id="rotulos11" class="block">
    <span class="section-tag">Seção IV — Prognóstico & Soft Labels</span>
    <h2 class="sec-title">Análise Comparativa das 11 Formulações de Rótulos (Futuro, Alerta Precoce e Distribuição Contínua)</h2>
    <p style="margin-bottom:1rem;color:var(--muted);">
      Comparação executada em GPU avaliando detecção instantânea, previsão de horizonte futuro (15, 30 e 60 min), alerta precoce puro (somente quando a bomba opera normalmente no presente) e distribuição contínua de probabilidade com kernel gaussiano assimétrico.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Abordagem de Rótulo</th>
            <th>Modo</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>F1-Score</th>
            <th>Precisão</th>
            <th>Recall</th>
            <th>Falsos Alarmes</th>
            <th>MAE Prob. (%)</th>
            <th>Corr. Spearman</th>
          </tr>
        </thead>
        <tbody>
          {rows_11_approaches_html()}
        </tbody>
      </table>
    </div>

    <div class="fig-grid">
      <div class="fig-card" style="grid-column: 1 / -1;">
        <img src="assets/img/comparacao_11_abordagens_rotulos.png" alt="Comparação 11 Abordagens" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 12:</strong> Painel Comparativo Global das 11 Formulações de Rótulos executadas na GPU RTX 5080.</div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 5: HISTOGRAMA DO ERRO DE RECONSTRUÇÃO + TRIAGEM DO OPERADOR     -->
  <!-- ===================================================================== -->
  <section id="histograma-operador" class="block">
    <span class="section-tag">Seção V — Human-in-the-Loop (Histograma do Erro)</span>
    <h2 class="sec-title">Separação por Histograma do Erro de Reconstrução com Avaliação do Operador para Dados Intermediários (FP e FN)</h2>
    <p style="margin-bottom:1rem;color:var(--muted);">
      O histograma do erro de reconstrução não supervisionado define três zonas: <strong>Zona 1 ($E &lt; \\tau_{{\\text{{baixo}}}}$: Normal Automático)</strong>, <strong>Zona 2 ($[\\tau_{{\\text{{baixo}}}}, \\tau_{{\\text{{alto}}}}$]: Zona Intermediária onde se concentram os FP e FN, revisada pelo Operador)</strong> e <strong>Zona 3 ($E &gt; \\tau_{{\\text{{alto}}}}$: Anomalia Automática)</strong>.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Arquitetura</th>
            <th>ROC-AUC (Não Sup.)</th>
            <th>F1 Limiar Único (Sem Operador)</th>
            <th>F1 Só Decisões Automáticas</th>
            <th>F1 Final c/ Operador (Zona FP/FN)</th>
            <th>Falsos Alarmes Finais</th>
            <th>% dos FP+FN Capturados na Zona</th>
            <th>% de Janelas Revisadas pelo Operador</th>
          </tr>
        </thead>
        <tbody>
          {rows_hist_operator_html()}
        </tbody>
      </table>
    </div>

    <div class="fig-grid">
      <div class="fig-card">
        <img src="assets/img/histogramas_erro_reconstrucao_zona_operador.png" alt="Histogramas do Erro de Reconstrução" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 13:</strong> Histogramas do Erro de Reconstrução por Bomba mostrando a Zona Verde (Normal), Zona Amarela (Revisão do Operador para FP/FN) e Zona Vermelha (Anomalia).</div>
      </div>
      <div class="fig-card">
        <img src="assets/img/comparacao_histograma_corte_unico_vs_operador.png" alt="Comparação Corte Único vs Operador" onclick="window.open(this.src)" />
        <div class="fig-cap"><strong>Figura 14:</strong> Ganho de F1-Score e taxa de captura de erros (FP+FN) por arquitetura com a triagem do operador.</div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 6: EXPLORADOR INTERATIVO POR BOMBA E JANELA TEMPORAL            -->
  <!-- ===================================================================== -->
  <section id="tabela-interativa" class="block">
    <span class="section-tag">Seção VI — Explorador Detalhado</span>
    <h2 class="sec-title">Resultados Detalhados por Bomba, Escala de Janela e Arquitetura</h2>

    <div class="filter-bar">
      <label><strong>Filtrar Bomba/Dataset:</strong>
        <select id="selDataset" onchange="renderDetailTable()">
          <option value="ALL">Todas as Bombas + Dataset Completo</option>
        </select>
      </label>
      <label><strong>Filtrar Janela:</strong>
        <select id="selWindow" onchange="renderDetailTable()">
          <option value="ALL">Todas (15min, 30min, 1h, 2h)</option>
          <option value="15min">15min (L=90)</option>
          <option value="30min">30min (L=180)</option>
          <option value="1h">1h (L=360)</option>
          <option value="2h">2h (L=720)</option>
        </select>
      </label>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset / Bomba</th>
            <th>Janela</th>
            <th>Arquitetura</th>
            <th>Normais</th>
            <th>Anomalias</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>F1-Score</th>
            <th>Precisão</th>
            <th>Recall</th>
            <th>Falsos Alarmes</th>
            <th>Razão Não Sup.</th>
          </tr>
        </thead>
        <tbody id="detailTableBody"></tbody>
      </table>
    </div>
  </section>

</main>

<footer>
  Projeto de Pesquisa — Detecção de Anomalias em Bombas Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais em PyTorch.<br/>
  Site gerado automaticamente por <code>gerar_site_github.py</code> e integrado via GitHub Actions ao GitHub Pages.
</footer>

<script>
  const SITE_DATA = {json.dumps(payload, ensure_ascii=False)};

  function initCharts() {{
    const r7 = SITE_DATA.ranking_7_arquiteturas || [];
    if (r7.length > 0) {{
      const labels = r7.map(r => r.architecture);
      new Chart(document.getElementById('chartArch7'), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [
            {{ label: 'ROC-AUC', data: r7.map(r => Number(r.roc_auc_hybrid_mean).toFixed(4)), backgroundColor: '#1d4ed8' }},
            {{ label: 'PR-AUC', data: r7.map(r => Number(r.pr_auc_hybrid_mean).toFixed(4)), backgroundColor: '#0284c7' }},
            {{ label: 'F1-Score', data: r7.map(r => Number(r.best_f1_hybrid_mean).toFixed(4)), backgroundColor: '#16a34a' }}
          ]
        }},
        options: {{
          responsive: true,
          scales: {{ y: {{ min: 0.80, max: 1.0 }} }}
        }}
      }});

      new Chart(document.getElementById('chartUnsupRatio'), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [
            {{ label: 'Razão Score Erro (Anômalo / Normal)', data: r7.map(r => Number(r.unsup_score_ratio_mean).toFixed(2)), backgroundColor: '#d97706' }}
          ]
        }},
        options: {{
          responsive: true
        }}
      }});
    }}
  }}

  function initFilters() {{
    const det = SITE_DATA.detalhe_bomba_janela || [];
    const dsSet = [...new Set(det.map(r => r.dataset))];
    const sel = document.getElementById('selDataset');
    dsSet.forEach(d => {{
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d;
      sel.appendChild(opt);
    }});
    renderDetailTable();
  }}

  function renderDetailTable() {{
    const det = SITE_DATA.detalhe_bomba_janela || [];
    const fDs = document.getElementById('selDataset').value;
    const fWin = document.getElementById('selWindow').value;
    const filtered = det.filter(r => (fDs === 'ALL' || r.dataset === fDs) && (fWin === 'ALL' || r.window === fWin));
    const tbody = document.getElementById('detailTableBody');
    tbody.innerHTML = filtered.slice(0, 140).map(r => `
      <tr>
        <td><strong>${{r.dataset}}</strong></td>
        <td>${{r.window}}</td>
        <td><code>${{r.architecture}}</code></td>
        <td>${{r.n_normal_evaluated}}</td>
        <td>${{r.n_anomaly_evaluated}}</td>
        <td><strong>${{Number(r.oof_roc_auc_hybrid).toFixed(4)}}</strong></td>
        <td>${{Number(r.oof_pr_auc_hybrid).toFixed(4)}}</td>
        <td><strong>${{Number(r.oof_best_f1_hybrid).toFixed(4)}}</strong></td>
        <td>${{Number(r.oof_precision_at_best_f1).toFixed(4)}}</td>
        <td>${{Number(r.oof_recall_at_best_f1).toFixed(4)}}</td>
        <td>${{Number(r.oof_false_alarm_rate_pct).toFixed(2)}}%</td>
        <td>${{Number(r.unsup_anomaly_score_ratio).toFixed(2)}}×</td>
      </tr>
    `).join('');
  }}

  window.addEventListener('DOMContentLoaded', () => {{
    initCharts();
    initFilters();
  }});
</script>
</body>
</html>
"""

    index_path = os.path.join(DOCS_DIR, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Cria arquivo .nojekyll para o GitHub Pages servir estaticamente sem interferência
    with open(os.path.join(DOCS_DIR, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")

    print(f"[OK] Site GitHub Pages gerado com sucesso em: {index_path}")
    print(f"[OK] Total de figuras sincronizadas em docs/assets/img/: {len(copied_imgs)}")


if __name__ == "__main__":
    build_site()
