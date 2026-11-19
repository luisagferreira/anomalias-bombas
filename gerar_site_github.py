"""
Gerador Automático do Site Científico Bilíngue (Português / Inglês) para GitHub Pages (`docs/`).

Este script lê dinamicamente os CSVs e figuras mais recentes da pasta
`Dataset_completo_bombas/Resultados_Modelos_1D_2D/Deteccao_Anomalias_Operador/`
e `Dataset_completo_bombas/Figuras_Arquiteturas/`, construindo um portal web
completo (`docs/index.html`, `docs/data/resultados.json`) com alternância instantânea
de idioma (PT-BR / EN) contendo:

1. Seção de Texto Acadêmico em Formato de Dissertação (Capítulos 1 a 4),
   contextualizando todo o problema industrial, fundamentação matemática,
   transformada GADF + PAA, validação cruzada sem data leakage, as 7 arquiteturas
   em PyTorch (torchviz + torchinfo.summary), as 11 formulações de rótulos e a
   triagem por histograma do erro de reconstrução com revisão do operador (FP/FN).
2. Seção Interativa de Resultados Experimentais (tabelas dinâmicas, curvas de
   perda/Loss de treino e validação, gráficos Chart.js, filtros por bomba/janela e
   galeria de figuras em alta resolução).
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
    for src_dir in (FIG_ARCH_DIR, FIG_ANOM_DIR, ART_DIR):
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            if not fname.lower().endswith(".png"):
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(IMG_OUT_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                if fname not in copied:
                    copied.append(fname)
    return sorted(copied)


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
    /* Controle de Idioma Bilíngue (PT / EN) */
    body.lang-pt [data-lang="en"] {{ display: none !important; }}
    body.lang-en [data-lang="pt"] {{ display: none !important; }}

    header.hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: #ffffff;
      padding: 2.8rem 1.5rem 2.4rem;
      border-bottom: 4px solid #3b82f6;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 0 1.25rem; }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1.2rem;
      flex-wrap: wrap;
    }}
    .hero h1 {{ font-size: 1.95rem; font-weight: 800; margin-bottom: 0.55rem; line-height: 1.25; max-width: 980px; }}
    .hero p {{ color: #cbd5e1; font-size: 1.1rem; font-weight: 500; }}
    .lang-switch {{
      display: inline-flex;
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 999px;
      padding: 0.22rem;
      gap: 0.25rem;
      flex-shrink: 0;
    }}
    .lang-btn {{
      border: none;
      background: transparent;
      color: #e2e8f0;
      font-weight: 700;
      font-size: 0.84rem;
      padding: 0.42rem 0.9rem;
      border-radius: 999px;
      cursor: pointer;
      transition: all 0.18s;
    }}
    .lang-btn.active {{
      background: #3b82f6;
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.45);
    }}
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
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      overflow-x: auto;
      padding: 0.65rem 0;
    }}
    .nav-links {{
      display: flex;
      gap: 0.4rem;
      overflow-x: auto;
    }}
    nav.topbar a {{
      text-decoration: none;
      color: var(--primary);
      font-weight: 600;
      font-size: 0.88rem;
      padding: 0.45rem 0.85rem;
      border-radius: 8px;
      white-space: nowrap;
      transition: all 0.15s;
    }}
    nav.topbar a:hover {{
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
    .viewer-box {{
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #ffffff;
      padding: 1.35rem;
      margin-top: 1.6rem;
      box-shadow: 0 2px 10px rgba(15,23,42,0.03);
    }}
    .viewer-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.8rem;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
    }}
    .viewer-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }}
    .v-tab, .v-nav-btn {{
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: var(--primary);
      font-weight: 600;
      font-size: 0.84rem;
      padding: 0.42rem 0.85rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .v-tab:hover, .v-nav-btn:hover {{
      background: #e2e8f0;
    }}
    .v-tab.active {{
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }}
    .viewer-stage {{
      text-align: center;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }}
    .viewer-stage img {{
      width: 100%;
      max-width: 1020px;
      height: auto;
      display: block;
      margin: 0 auto;
      cursor: zoom-in;
    }}
    .viewer-caption {{
      padding: 0.85rem 1.1rem;
      font-size: 0.92rem;
      color: var(--muted);
      background: #f8fafc;
      border-top: 1px solid var(--border);
      text-align: left;
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
<body class="lang-pt">

<header class="hero">
  <div class="container">
    <div class="hero-top">
      <div>
        <h1 data-lang="pt">Detecção e Prognóstico de Anomalias em Bombas de Polpa Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais</h1>
        <h1 data-lang="en">Anomaly Detection and Prognostics in Industrial Slurry Pumps with 1D, 2D (GADF), and Multimodal Hybrid Autoencoders</h1>
        <p data-lang="pt">Portal Científico de Resultados Experimentais</p>
        <p data-lang="en">Scientific Portal of Experimental Results</p>
      </div>
      <div class="lang-switch" role="group" aria-label="Language Selector">
        <button type="button" class="lang-btn active" id="btnLangPt" onclick="setLanguage('pt')">🇧🇷 Português</button>
        <button type="button" class="lang-btn" id="btnLangEn" onclick="setLanguage('en')">🇨🇦 English</button>
      </div>
    </div>
    <div class="meta-bar">
      <span class="chip">Framework: PyTorch 2.x + CUDA</span>
      <span class="chip">Hardware: NVIDIA RTX 5080 (16 GB VRAM)</span>
      <span class="chip" data-lang="pt">Dataset: 6 Bombas × 4 Sensores (Historiador PI, 10s)</span>
      <span class="chip" data-lang="en">Dataset: 6 Slurry Pumps × 4 Sensors (PI Historian, 10s)</span>
      <span class="chip" data-lang="pt">Validação: 5-Fold StratifiedGroupKFold (Zero Data Leakage)</span>
      <span class="chip" data-lang="en">Validation: 5-Fold StratifiedGroupKFold (Zero Data Leakage)</span>
      <span class="chip" data-lang="pt">Atualizado em: {updated_at}</span>
      <span class="chip" data-lang="en">Last updated: {updated_at}</span>
    </div>
  </div>
</header>

<nav class="topbar">
  <div class="container nav-inner">
    <div class="nav-links">
      <a href="#dissertacao">
        <span data-lang="pt">1. Texto Acadêmico (Dissertação)</span>
        <span data-lang="en">1. Academic Dissertation Text</span>
      </a>
      <a href="#arquiteturas">
        <span data-lang="pt">2. Arquiteturas PyTorch (torchviz &amp; summary)</span>
        <span data-lang="en">2. PyTorch Architectures (torchviz &amp; summary)</span>
      </a>
      <a href="#resultados">
        <span data-lang="pt">3. Resultados Experimentais &amp; Loss</span>
        <span data-lang="en">3. Experimental Results &amp; Loss</span>
      </a>
      <a href="#rotulos11">
        <span data-lang="pt">4. Prognóstico (11 Formulações)</span>
        <span data-lang="en">4. Prognostics (11 Label Formulations)</span>
      </a>
      <a href="#histograma-operador">
        <span data-lang="pt">5. Histograma do Erro &amp; Operador (FP/FN)</span>
        <span data-lang="en">5. Error Histogram &amp; Operator Triage (FP/FN)</span>
      </a>
      <a href="#tabela-interativa">
        <span data-lang="pt">6. Explorador por Bomba &amp; Janela</span>
        <span data-lang="en">6. Pump &amp; Window Explorer</span>
      </a>
    </div>
  </div>
</nav>

<main class="container">

  <!-- ===================================================================== -->
  <!-- SEÇÃO 1: TEXTO ACADÊMICO EM FORMATO DE DISSERTAÇÃO (PT & EN)          -->
  <!-- ===================================================================== -->
  <section id="dissertacao" class="block dissertation">
    <span class="section-tag" data-lang="pt">Seção I — Texto Acadêmico</span>
    <span class="section-tag" data-lang="en">Section I — Academic Dissertation</span>
    <h2 class="sec-title" data-lang="pt">Contextualização e Fundamentação Científica em Formato de Dissertação</h2>
    <h2 class="sec-title" data-lang="en">Scientific Contextualization and Theoretical Framework (Dissertation Format)</h2>

    <!-- VERSÃO EM PORTUGUÊS -->
    <div data-lang="pt">
      <h3>Resumo</h3>
      <p>
        Este trabalho investiga a detecção não supervisionada, a triagem assistida por operador (<em>Human-in-the-Loop</em>) e o prognóstico antecipado de anomalias operacionais em um parque de seis bombas de polpa industriais (Bombas 1305, 1307, 2307, 3307, 4307 e 5307) monitoradas continuamente ao longo de dez meses (janeiro a outubro de 2026) por meio do sistema Historiador PI com taxa de amostragem de 10 segundos ($0,1\\text{{ Hz}}$). Cada unidade de bombeamento é instrumentada com quatro variáveis de processo: corrente elétrica do motor, vazão volumétrica, pressão de sucção e pressão de descarga. Propõe-se e avalia-se sistematicamente um conjunto de sete arquiteturas neurais baseadas em Autoencoders implementadas nativamente em PyTorch e treinadas em GPU NVIDIA GeForce RTX 5080: três modelos temporais unidimensionais (<code>AE_CNN1D</code>, <code>AE_LSTM1D</code> e <code>AE_Transformer1D</code>), um modelo convolucional bidimensional operando sobre imagens de diferença angular de Gram (<code>AE_CNN2D_GAF</code> utilizando exclusivamente matrizes GADF reduzidas por <em>Piecewise Aggregate Approximation</em> para $32\\times 32$) e três arquiteturas híbridas multimodais (<code>Hybrid_CNN1D_CNN2D</code>, <code>Hybrid_LSTM1D_CNN2D</code> e <code>Hybrid_Transformer_CNN2D</code>) que fundem representações temporais 1D e angulares 2D em um espaço latente compartilhado $\\mathbf{{z}} \\in \\mathbb{{R}}^{{32}}$.
      </p>

      <h3>Capítulo 1 — Introdução e Contextualização do Problema Industrial</h3>
      <p>
        Bombas de polpa de grande porte constituem ativos críticos em plantas de beneficiamento mineral e processamento contínuo. Falhas súbitas por cavitação, sobrecarga elétrica, abrasão severa, obstrução de linha ou desgaste de impelidores acarretam paradas não programadas de alto custo e riscos operacionais. Embora sistemas supervisórios industriais (como o OSIsoft PI Historian) armazenem milhões de registros multivariados ao longo do ano, a grande maioria dos dados corresponde a regimes normais de operação, enquanto eventos de falha ou instabilidade severa são raros, transitórios e frequentemente carecem de rotulagem fina instante a instante.
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
          <li><strong>Superioridade Física do GADF sobre o GASF:</strong> Enquanto o GASF ($\\cos(\\phi_i + \\phi_j)$) enfatiza níveis estáticos de carga, o GADF ($\\sin(\\phi_i - \\phi_j)$) captura diretamente derivadas e transientes abruptos entre instantes distintos — exatamente a assinatura física de picos de corrente e instabilidades hidráulicas no bombeamento de polpa.</li>
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
        Como janelas consecutivas possuem $50\\%$ de sobreposição temporal, uma divisão aleatória convencional causaria vazamento de dados (<em>data leakage</em>). Para eliminar rigorosamente esse efeito:
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
        <li><strong>Zona 2 — Zona Intermediária de Triagem Humana ($\\tau_{{\\text{{baixo}}}} \\le E(x) \\le \\tau_{{\\text{{alto}}}}$):</strong> Região de incerteza que concentra ~75% de todos os FP e FN; as janelas desta faixa (~14,9% do total) são encaminhadas para avaliação do operador.</li>
        <li><strong>Zona 3 — Anomalia Confirmada Automática ($E(x) > \\tau_{{\\text{{alto}}}}$):</strong> Disparo automático de alarme com altíssima precisão.</li>
      </ul>

      <h4>4.3 Onze Formulações de Rótulos para Prognóstico e Distribuição Contínua de Probabilidade</h4>
      <p>
        Além da detecção instantânea, avalia-se a capacidade preditiva dos modelos sob 11 formulações distintas: (i) detecção instantânea binária; (ii) previsão geral de horizonte futuro ($H \\in \\{{15, 30, 60\\}}\\text{{ min}}$); (iii) <strong>alerta precoce puro</strong> (avaliado estritamente quando a bomba ainda opera em estado 100% normal no presente, $y_{{\\text{{atual}}}}=0$, mas sofrerá anomalia nos próximos $H$ minutos); e (iv) <strong>distribuição contínua de probabilidade</strong> com kernel gaussiano temporal assimétrico ($\\tau_{{\\text{{antes}}}}, \\tau_{{\\text{{depois}}}}$), atribuindo uma porcentagem gradual de risco antes e depois do pico marcado pelo operador.
      </p>
    </div>

    <!-- ENGLISH VERSION -->
    <div data-lang="en">
      <h3>Abstract</h3>
      <p>
        This research investigates unsupervised anomaly detection, operator-assisted triage (<em>Human-in-the-Loop</em>), and early prognostic forecasting in a fleet of six industrial slurry pumps (Pumps 1305, 1307, 2307, 3307, 4307, and 5307) continuously monitored over ten months (January to October 2026) via the OSIsoft PI Historian system at a 10-second sampling interval ($0.1\\text{{ Hz}}$). Each pumping unit is instrumented with four physical process variables: motor electric current, volumetric flow rate, suction pressure, and discharge pressure. We systematically design and benchmark seven neural Autoencoder architectures natively implemented in PyTorch and trained on an NVIDIA GeForce RTX 5080 GPU: three one-dimensional temporal models (<code>AE_CNN1D</code>, <code>AE_LSTM1D</code>, and <code>AE_Transformer1D</code>), one two-dimensional convolutional model operating on Gramian Angular Difference Field images (<code>AE_CNN2D_GAF</code> using exclusively GADF matrices compressed via <em>Piecewise Aggregate Approximation</em> to $32\\times 32$), and three multimodal hybrid architectures (<code>Hybrid_CNN1D_CNN2D</code>, <code>Hybrid_LSTM1D_CNN2D</code>, and <code>Hybrid_Transformer_CNN2D</code>) that fuse 1D temporal and 2D angular representations into a shared latent bottleneck $\\mathbf{{z}} \\in \\mathbb{{R}}^{{32}}$.
      </p>

      <h3>Chapter 1 — Introduction and Industrial Slurry Pumping Context</h3>
      <p>
        Heavy-duty centrifugal slurry pumps are mission-critical assets in mineral processing and continuous industrial plants. Sudden failures caused by cavitation, electrical overload, severe slurry abrasion, pipeline blockage, or impeller wear lead to costly unplanned downtime and operational hazards. Although industrial historian systems store millions of multivariate samples throughout the year, the overwhelming majority of records reflect nominal steady-state operation, whereas severe instability events are rare, transient, and lack fine-grained timestamp annotations.
      </p>
      <p>
        In this setting, <strong>Reconstruction Autoencoders</strong> excel by learning the normal operating manifold exclusively from healthy windows. When presented with a temporal window containing hydrodynamic or electrical disturbances, the network fails to reconstruct out-of-distribution patterns, yielding high residual errors that drive both automated alarming and human-in-the-loop triage of borderline cases (False Positives and False Negatives).
      </p>

      <h3>Chapter 2 — Theoretical Framework: 1D Signals, 2D GADF Transform, and PAA Compression</h3>
      <h4>2.1 Multi-Scale Sliding Windows and Fold-Strict Normalization</h4>
      <p>
        The continuous multivariate time series $\\mathbf{{X}} \\in \\mathbb{{R}}^{{C \\times T}}$ ($C=4$ sensors per individual pump or $C=24$ channels via dynamic in-memory concatenation for <code>Dataset_Completo</code>) is segmented into sliding windows of length $L$ with $50\\%$ overlap across four time scales: 15 minutes ($L=90$), 30 minutes ($L=180$), 1 hour ($L=360$), and 2 hours ($L=720$). Prior to angular projection, each channel is scaled to $[-1, 1]$ using bounds calibrated strictly on the training folds:
      </p>
      $$\\tilde{{x}}_t = \\text{{clip}}\\left(2 \\cdot \\frac{{x_t - x_{{\\min}}}}{{x_{{\\max}} - x_{{\\min}}}} - 1,\\ -1,\\ 1\\right)$$

      <h4>2.2 Exclusive Gramian Angular Difference Field (GADF) and PAA Reduction</h4>
      <p>
        To map 1D signals into 2D matrices that expose pairwise temporal correlations to 2D CNNs, we first apply <em>Piecewise Aggregate Approximation</em> (PAA), compressing any window of length $L \\in \\{{90, 180, 360, 720\\}}$ into $M = 32$ temporal segments. Each normalized value is then mapped into polar coordinates $\\phi_i = \\arccos(\\tilde{{x}}_i)$ to compute exclusively the <strong>GADF</strong> (<em>Gramian Angular Difference Field</em>) matrix:
      </p>
      $$\\text{{GADF}}_{{i,j}} = \\sin(\\phi_i - \\phi_j) = \\tilde{{x}}_j \\sqrt{{1 - \\tilde{{x}}_i^2}} - \\tilde{{x}}_i \\sqrt{{1 - \\tilde{{x}}_j^2}} \\in \\mathbb{{R}}^{{32 \\times 32}}$$
      <div class="callout">
        <strong>Physical and Computational Justification for PAA ($32\\times 32$) + Exclusive GADF:</strong>
        <ul>
          <li><strong>Physical Superiority of GADF over GASF:</strong> Whereas GASF ($\\cos(\\phi_i + \\phi_j)$) reflects static load magnitude, GADF ($\\sin(\\phi_i - \\phi_j)$) directly captures temporal derivatives and abrupt phase transitions across time steps—the exact signature of current spikes and hydraulic instabilities in slurry pumping.</li>
          <li><strong>Storage and GPU VRAM Feasibility:</strong> Without PAA, the 2,574,624 GADF matrices would grow quadratically ($L \\times L$, reaching $720 \\times 720$ at the 2h scale), requiring <strong>310.74 GiB</strong> of raw <code>float16</code> memory (or <strong>121.1 GiB</strong> even in compressed <code>.npz</code>). By combining $32\\times 32$ PAA, removing redundant GASF tensors, and applying lossless <code>np.savez_compressed</code>, all 24 dataset files occupy only <strong>2.89 GiB</strong> on disk (over <strong>97% reduction</strong>) while preserving a uniform input resolution for 2D convolutions.</li>
        </ul>
      </div>

      <h3>Chapter 3 — Experimental Methodology, Cross-Validation, and Zero Data Leakage</h3>
      <h4>3.1 Annual Operator Anomaly Criterion ($\\text{{StdDev}} \\ge 2.5\\text{{ A}}$)</h4>
      <p>
        The operator ground-truth reference is defined by the sample standard deviation of motor electric current computed over consecutive 5-minute blocks across the entire year. Blocks with $\\text{{StdDev}} \\ge 2.5\\text{{ A}}$ indicate severe current oscillations and operational disturbances, which are mapped onto the corresponding sliding windows.
      </p>
      <h4>3.2 Label-Stratified Group K-Fold with Boundary Window Exclusion</h4>
      <p>
        Because consecutive sliding windows share a $50\\%$ overlap, random splitting would induce temporal data leakage. To guarantee zero leakage:
      </p>
      <ol>
        <li>We employ <code>StratifiedGroupKFold(n_splits=5)</code> stratified <strong>strictly by the binary anomaly label</strong> ($y \\in \\{{0, 1\\}}$) and grouped by contiguous 24-hour temporal blocks (<code>block_id</code>), ensuring balanced anomaly proportions across all 5 folds without splitting contiguous events.</li>
        <li>A dynamic boundary exclusion mask (<code>clean_mask</code>) discards any window $w$ whose assigned fold differs from its immediate neighbors ($w-1$ or $w+1$) (~0.85% of windows), enforcing complete temporal independence between training and validation sets.</li>
      </ol>

      <h3>Chapter 4 — Post-Reconstruction Decision, Future Prognostics, and Histogram Triage with Operator</h3>
      <h4>4.1 From Reconstruction Error to Operational Decision</h4>
      <p>
        Given the reconstructed signal $\\hat{{\\mathbf{{X}}}}_{{1D}}$ and reconstructed matrix $\\hat{{\\mathbf{{X}}}}_{{\\text{{GADF}}}}$, the pipeline extracts mean squared residuals ($\\text{{MSE}}_{{1D}}$ and $\\text{{MSE}}_{{\\text{{GADF}}}}$) along with per-channel physical residual descriptors. In purely unsupervised mode, each window's residual vector is standardized against the mean $\\mu_{{\\text{{norm}}}}$ and standard deviation $\\sigma_{{\\text{{norm}}}}$ calibrated strictly on normal training windows:
      </p>
      $$E(x) = \\frac{{1}}{{K}} \\sum_{{k=1}}^{{K}} \\frac{{|\\text{{Res}}_k(x) - \\mu_{{k,\\text{{norm}}}}|}}{{\\sigma_{{k,\\text{{norm}}}} + \\epsilon}}$$

      <h4>4.2 Reconstruction Error Histogram Separation &amp; Human-in-the-Loop Operator Zone</h4>
      <p>
        The empirical histogram of $E(x)$ across normal and anomalous windows exhibits a bimodal distribution with a central overlap region where <strong>False Positives (FP)</strong> and <strong>False Negatives (FN)</strong> of a single hard threshold $\\tau_{{\\text{{hist}}}}$ occur. Rather than enforcing a blind binary cut inside this grey zone, two histogram thresholds ($\\tau_{{\\text{{low}}}} < \\tau_{{\\text{{hist}}}} < \\tau_{{\\text{{high}}}}$) partition operation into three zones:
      </p>
      <ul>
        <li><strong>Zone 1 — Automated Normal ($E(x) < \\tau_{{\\text{{low}}}}$):</strong> High-confidence automated normal classification.</li>
        <li><strong>Zone 2 — Intermediate Operator Triage Zone ($\\tau_{{\\text{{low}}}} \\le E(x) \\le \\tau_{{\\text{{high}}}}$):</strong> Uncertainty region capturing ~75% of all FPs and FNs (~14.9% of windows), routed to the human operator for inspection.</li>
        <li><strong>Zone 3 — Automated Confirmed Anomaly ($E(x) > \\tau_{{\\text{{high}}}}$):</strong> High-precision automated anomaly alarm.</li>
      </ul>

      <h4>4.3 Eleven Label Formulations for Prognostics and Continuous Probability Distributions</h4>
      <p>
        Beyond instantaneous detection, we evaluate predictive capability across 11 label formulations: (i) instantaneous binary detection; (ii) general future horizon forecasting ($H \\in \\{{15, 30, 60\\}}\\text{{ min}}$); (iii) <strong>pure early warning</strong> (evaluated strictly when the pump is currently 100% normal, $y_{{\\text{{current}}}}=0$, but will enter an anomalous regime within the next $H$ minutes); and (iv) <strong>continuous probability distributions</strong> using an asymmetric Gaussian temporal kernel ($\\tau_{{\\text{{before}}}}, \\tau_{{\\text{{after}}}}$) that assigns a smooth risk percentage around operator-marked events.
      </p>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 2: ARQUITETURAS PYTORCH (TORCHVIZ + TORCHINFO.SUMMARY)          -->
  <!-- ===================================================================== -->
  <section id="arquiteturas" class="block">
    <span class="section-tag" data-lang="pt">Seção II — Arquiteturas em PyTorch</span>
    <span class="section-tag" data-lang="en">Section II — PyTorch Architectures</span>
    <h2 class="sec-title" data-lang="pt">Grafos Computacionais (torchviz) e Resumo Estrutural (torchinfo.summary)</h2>
    <h2 class="sec-title" data-lang="en">Computational Graphs (torchviz) and Structural Summary (torchinfo.summary)</h2>
    <p style="margin-bottom:1rem;color:var(--muted);" data-lang="pt">
      Todas as 7 arquiteturas compartilham o mesmo gargalo latente ($Z = 32$) para comparação justa. Clique em qualquer figura para ampliá-la em tela cheia.
    </p>
    <p style="margin-bottom:1rem;color:var(--muted);" data-lang="en">
      All 7 architectures share the exact same latent bottleneck ($Z = 32$) for fair comparison. Click any figure to open it in full resolution.
    </p>
    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURAS 1 A 9 (ARQUITETURAS PYTORCH) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figuras 1 a 9 (Visualização Individual): Fluxo Metodológico e Arquiteturas PyTorch (torchviz + torchinfo.summary)</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figures 1 to 9 (Image-by-Image Viewer): Methodological Workflow &amp; PyTorch Architectures (torchviz + torchinfo.summary)</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('arch', -1)">◀ Anterior / Prev</button>
          <span id="archCounter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 9</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('arch', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="archTabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="archImg" src="assets/img/fluxo_trabalho_e_decisoes_arquitetura.png" alt="Arquitetura PyTorch Individual" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="archCap"></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 3: RESULTADOS EXPERIMENTAIS GLOBAIS (7 ARQUITETURAS & LOSS)     -->
  <!-- ===================================================================== -->
  <section id="resultados" class="block">
    <span class="section-tag" data-lang="pt">Seção III — Benchmark em GPU &amp; Curvas de Loss</span>
    <span class="section-tag" data-lang="en">Section III — GPU Benchmark &amp; Training Loss Curves</span>
    <h2 class="sec-title" data-lang="pt">Resultados Globais das 7 Arquiteturas (100% Dataset Anual, StdDev ≥ 2,5 A)</h2>
    <h2 class="sec-title" data-lang="en">Global Results Across the 7 Architectures (100% Annual Dataset, StdDev ≥ 2.5 A)</h2>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label" data-lang="pt">Melhor F1-Score &amp; PR-AUC</div>
        <div class="kpi-label" data-lang="en">Best F1-Score &amp; PR-AUC</div>
        <div class="kpi-val">0.8736 / 0.9366</div>
        <div class="kpi-sub">Hybrid_Transformer_CNN2D</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label" data-lang="pt">Maior ROC-AUC Global</div>
        <div class="kpi-label" data-lang="en">Highest Overall ROC-AUC</div>
        <div class="kpi-val">0.9808</div>
        <div class="kpi-sub">AE_LSTM1D (Acc: 94.94%)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label" data-lang="pt">F1 Final c/ Triagem do Operador</div>
        <div class="kpi-label" data-lang="en">Final F1 with Operator Triage</div>
        <div class="kpi-val">0.9789</div>
        <div class="kpi-sub" data-lang="pt">FAR reduzido para 1,11% (AE_LSTM1D)</div>
        <div class="kpi-sub" data-lang="en">FAR reduced to 1.11% (AE_LSTM1D)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label" data-lang="pt">Redução de Volume GADF c/ PAA</div>
        <div class="kpi-label" data-lang="en">GADF Storage Reduction via PAA</div>
        <div class="kpi-val">2,89 GiB</div>
        <div class="kpi-sub" data-lang="pt">vs. 310,74 GiB sem PAA (−99,1%)</div>
        <div class="kpi-sub" data-lang="en">vs. 310.74 GiB without PAA (−99.1%)</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <h4 style="margin-bottom:0.8rem;color:var(--primary);" data-lang="pt">Desempenho por Arquitetura (ROC-AUC, PR-AUC e F1-Score)</h4>
        <h4 style="margin-bottom:0.8rem;color:var(--primary);" data-lang="en">Performance by Architecture (ROC-AUC, PR-AUC, and F1-Score)</h4>
        <canvas id="chartArch7" height="220"></canvas>
      </div>
      <div class="chart-card">
        <h4 style="margin-bottom:0.8rem;color:var(--primary);" data-lang="pt">Razão de Score de Anomalia Não Supervisionado (Anômalo / Normal)</h4>
        <h4 style="margin-bottom:0.8rem;color:var(--primary);" data-lang="en">Unsupervised Anomaly Score Ratio (Anomalous / Normal)</h4>
        <canvas id="chartUnsupRatio" height="220"></canvas>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Architecture</th>
            <th>ROC-AUC (Hybrid)</th>
            <th>PR-AUC (Hybrid)</th>
            <th>Best F1</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Accuracy</th>
            <th>False Alarm Rate</th>
            <th>ROC-AUC (Unsup.)</th>
            <th>Score Ratio (Unsup.)</th>
          </tr>
        </thead>
        <tbody>
          {rows_ranking_7_html()}
        </tbody>
      </table>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 10 (CURVAS DE LOSS POR ARQUITETURA) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 10 (Visualização Individual por Arquitetura): Curvas de Perda (Loss MSE) no Treino vs. Validação</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 10 (Image-by-Image Architecture Viewer): Training vs. Validation Reconstruction Loss Curves (MSE)</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig10', -1)">◀ Anterior / Prev</button>
          <span id="fig10Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 8</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig10', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig10Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig10Img" src="assets/img/fig10_loss_AE_CNN1D.png" alt="Figura 10 Loss por Arquitetura" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig10Cap"></div>
      </div>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 11 (CURVAS DE LOSS POR BOMBA) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 11 (Visualização Individual por Bomba): Convergência da Perda (Loss) no Treino e Validação</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 11 (Image-by-Image Pump Viewer): Training &amp; Validation Loss Convergence per Slurry Pump</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig11', -1)">◀ Anterior / Prev</button>
          <span id="fig11Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 7</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig11', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig11Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig11Img" src="assets/img/fig11_loss_Bomba_1305.png" alt="Figura 11 Loss por Bomba" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig11Cap"></div>
      </div>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 12 (MATRIZES DE CONFUSÃO POR BOMBA) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 12 (Visualização Individual por Bomba): Matrizes de Confusão Out-of-Fold (5 Folds)</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 12 (Image-by-Image Pump Viewer): Out-of-Fold Confusion Matrices (5 Folds)</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig12', -1)">◀ Anterior / Prev</button>
          <span id="fig12Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 7</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig12', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig12Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig12Img" src="assets/img/fig12_cm_Bomba_1305.png" alt="Figura 12 Matriz de Confusão" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig12Cap"></div>
      </div>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 13 (DESEMPENHO POR ESCALA DE JANELA) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 13 (Visualização Individual por Métrica): Comparação nas 4 Escalas de Janela Temporal (15m, 30m, 1h, 2h)</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 13 (Single-Chart Metric Viewer): Comparison Across the 4 Sliding Window Time Scales (15m, 30m, 1h, 2h)</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig13', -1)">◀ Anterior / Prev</button>
          <span id="fig13Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 4</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig13', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig13Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig13Img" src="assets/img/fig13a_janela_roc_auc.png" alt="Figura 13 Comparação por Janela" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig13Cap"></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 4: 11 FORMULAÇÕES DE RÓTULOS (FUTURO, ALERTA PRECOCE, SOFT)     -->
  <!-- ===================================================================== -->
  <section id="rotulos11" class="block">
    <span class="section-tag" data-lang="pt">Seção IV — Prognóstico &amp; Soft Labels</span>
    <span class="section-tag" data-lang="en">Section IV — Prognostics &amp; Soft Labels</span>
    <h2 class="sec-title" data-lang="pt">Análise Comparativa das 11 Formulações de Rótulos (Futuro, Alerta Precoce e Distribuição Contínua)</h2>
    <h2 class="sec-title" data-lang="en">Comparative Analysis of the 11 Label Formulations (Future Horizon, Pure Early Warning, and Soft Kernel)</h2>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Label Formulation</th>
            <th>Mode</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>F1-Score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>False Alarm Rate</th>
            <th>Prob. MAE (%)</th>
            <th>Spearman Corr.</th>
          </tr>
        </thead>
        <tbody>
          {rows_11_approaches_html()}
        </tbody>
      </table>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 14 (11 FORMULAÇÕES DE RÓTULOS) -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 14 (Visualização Individual por Métrica): Comparação Global das 11 Formulações de Rótulos</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 14 (Single-Chart Metric Viewer): Global Comparison of the 11 Label Formulations</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig14', -1)">◀ Anterior / Prev</button>
          <span id="fig14Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 4</span>
          <button type="button" class="v-nav-btn" onclick="stepViewer('fig14', 1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig14Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig14Img" src="assets/img/fig14a_rotulos_roc_pr.png" alt="Figura 14 11 Abordagens" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig14Cap"></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 5: HISTOGRAMA DO ERRO DE RECONSTRUÇÃO + TRIAGEM DO OPERADOR     -->
  <!-- ===================================================================== -->
  <section id="histograma-operador" class="block">
    <span class="section-tag" data-lang="pt">Seção V — Human-in-the-Loop (Histograma do Erro)</span>
    <span class="section-tag" data-lang="en">Section V — Human-in-the-Loop (Reconstruction Error Histogram)</span>
    <h2 class="sec-title" data-lang="pt">Separação por Histograma do Erro de Reconstrução com Avaliação do Operador para Dados Intermediários (FP e FN)</h2>
    <h2 class="sec-title" data-lang="en">Reconstruction Error Histogram Separation with Operator Assessment for Intermediate Data (FP &amp; FN)</h2>
    <p style="margin-bottom:1rem;color:var(--muted);" data-lang="pt">
      O histograma do erro de reconstrução define três zonas: <strong>Zona 1 ($E &lt; \\tau_{{\\text{{baixo}}}}$: Normal Automático)</strong>, <strong>Zona 2 ($[\\tau_{{\\text{{baixo}}}}, \\tau_{{\\text{{alto}}}}$]: Zona Intermediária onde se concentram os FP e FN, revisada pelo Operador)</strong> e <strong>Zona 3 ($E &gt; \\tau_{{\\text{{alto}}}}$: Anomalia Automática)</strong>.
    </p>
    <p style="margin-bottom:1rem;color:var(--muted);" data-lang="en">
      The reconstruction error histogram defines three decision zones: <strong>Zone 1 ($E &lt; \\tau_{{\\text{{low}}}}$: Automated Normal)</strong>, <strong>Zone 2 ($[\\tau_{{\\text{{low}}}}, \\tau_{{\\text{{high}}}}$]: Intermediate overlap region concentrating FPs and FNs, routed to the Operator)</strong>, and <strong>Zone 3 ($E &gt; \\tau_{{\\text{{high}}}}$: Automated Anomaly)</strong>.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Architecture</th>
            <th>ROC-AUC (Unsup.)</th>
            <th>Single-Threshold F1</th>
            <th>Auto-Only F1 (Zones 1 &amp; 3)</th>
            <th>Final HITL F1 (with Operator)</th>
            <th>Final False Alarm Rate</th>
            <th>% of FP+FN Captured in Zone 2</th>
            <th>% Windows Reviewed by Operator</th>
          </tr>
        </thead>
        <tbody>
          {rows_hist_operator_html()}
        </tbody>
      </table>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 15 -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 15 (Visualização Individual por Bomba): Histograma do Erro de Reconstrução e Zonas de Decisão</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 15 (Image-by-Image Pump Viewer): Reconstruction Error Histogram &amp; Decision Zones</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepFig15(-1)">◀ Anterior / Prev</button>
          <span id="fig15Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 7</span>
          <button type="button" class="v-nav-btn" onclick="stepFig15(1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig15Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig15Img" src="assets/img/fig15_hist_Bomba_1305.png" alt="Figura 15 Histograma Individual" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig15Cap"></div>
      </div>
    </div>

    <!-- VISUALIZADOR IMAGEM POR IMAGEM — FIGURA 16 -->
    <div class="viewer-box">
      <div class="viewer-header">
        <div>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="pt">Figura 16 (Visualização Individual por Gráfico): Ganho de F1-Score e Taxa de Captura (FP+FN) com Operador</strong>
          <strong style="font-size:1.05rem;color:var(--primary);" data-lang="en">Figure 16 (Single-Chart Viewer): F1-Score Gain &amp; Error Capture Rate (FP+FN) with Operator Triage</strong>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button type="button" class="v-nav-btn" onclick="stepFig16(-1)">◀ Anterior / Prev</button>
          <span id="fig16Counter" style="font-weight:700;font-size:0.88rem;color:var(--muted);">1 / 2</span>
          <button type="button" class="v-nav-btn" onclick="stepFig16(1)">Próxima / Next ▶</button>
        </div>
      </div>
      <div class="viewer-tabs" id="fig16Tabs" style="margin-bottom:1rem;"></div>
      <div class="viewer-stage">
        <img id="fig16Img" src="assets/img/fig16a_ganho_f1_operador.png" alt="Figura 16 Gráfico Individual" onclick="window.open(this.src)" />
        <div class="viewer-caption" id="fig16Cap"></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- SEÇÃO 6: EXPLORADOR INTERATIVO POR BOMBA E JANELA TEMPORAL            -->
  <!-- ===================================================================== -->
  <section id="tabela-interativa" class="block">
    <span class="section-tag" data-lang="pt">Seção VI — Explorador Detalhado</span>
    <span class="section-tag" data-lang="en">Section VI — Interactive Explorer</span>
    <h2 class="sec-title" data-lang="pt">Resultados Detalhados por Bomba, Escala de Janela e Arquitetura</h2>
    <h2 class="sec-title" data-lang="en">Detailed Results by Slurry Pump, Window Scale, and Architecture</h2>

    <div class="filter-bar">
      <label><strong data-lang="pt">Filtrar Bomba/Dataset:</strong><strong data-lang="en">Filter Pump/Dataset:</strong>
        <select id="selDataset" onchange="renderDetailTable()">
          <option value="ALL">ALL (6 Pumps + Dataset_Completo)</option>
        </select>
      </label>
      <label><strong data-lang="pt">Filtrar Janela:</strong><strong data-lang="en">Filter Window:</strong>
        <select id="selWindow" onchange="renderDetailTable()">
          <option value="ALL">ALL (15min, 30min, 1h, 2h)</option>
          <option value="15min">15min (L=90)</option>
          <option value="30min">30min (L=180)</option>
          <option value="1h">1h (L=360)</option>
          <option value="2h">2h (L=720)</option>
        </select>
      </label>
      <label><strong data-lang="pt">Filtrar Arquitetura:</strong><strong data-lang="en">Filter Architecture:</strong>
        <select id="selArch" onchange="renderDetailTable()">
          <option value="ALL">ALL (7 Architectures)</option>
        </select>
      </label>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset / Pump</th>
            <th>Window</th>
            <th>Architecture</th>
            <th>Normal Win.</th>
            <th>Anomaly Win.</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>F1-Score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>False Alarms</th>
            <th>Unsup. Ratio</th>
          </tr>
        </thead>
        <tbody id="detailTableBody"></tbody>
      </table>
    </div>
  </section>

</main>

<footer>
  <span data-lang="pt">Detecção e Prognóstico de Anomalias em Bombas de Polpa Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais — Portal Científico de Resultados Experimentais.</span>
  <span data-lang="en">Anomaly Detection and Prognostics in Industrial Slurry Pumps with 1D, 2D (GADF), and Multimodal Hybrid Autoencoders — Scientific Portal of Experimental Results.</span>
</footer>

<script>
  const SITE_DATA = {json.dumps(payload, ensure_ascii=False)};

  function setLanguage(lang) {{
    document.body.classList.remove('lang-pt', 'lang-en');
    document.body.classList.add('lang-' + lang);
    document.getElementById('btnLangPt').classList.toggle('active', lang === 'pt');
    document.getElementById('btnLangEn').classList.toggle('active', lang === 'en');
    try {{ localStorage.setItem('site_lang_pref', lang); }} catch (e) {{}}
    if (window.MathJax && window.MathJax.typesetPromise) {{
      window.MathJax.typesetPromise();
    }}
  }}

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
            {{ label: 'Score Ratio (Anomalous / Normal)', data: r7.map(r => Number(r.unsup_score_ratio_mean).toFixed(2)), backgroundColor: '#d97706' }}
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

    const archSet = [...new Set(det.map(r => r.architecture))];
    const selArch = document.getElementById('selArch');
    archSet.forEach(a => {{
      const opt = document.createElement('option');
      opt.value = a;
      opt.textContent = a;
      selArch.appendChild(opt);
    }});

    renderDetailTable();
  }}

  function renderDetailTable() {{
    const det = SITE_DATA.detalhe_bomba_janela || [];
    const fDs = document.getElementById('selDataset').value;
    const fWin = document.getElementById('selWindow').value;
    const fArch = document.getElementById('selArch').value;
    const filtered = det.filter(r =>
      (fDs === 'ALL' || r.dataset === fDs) &&
      (fWin === 'ALL' || r.window === fWin) &&
      (fArch === 'ALL' || r.architecture === fArch)
    );
    const tbody = document.getElementById('detailTableBody');
    tbody.innerHTML = filtered.map(r => `
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

  const VIEWERS = {{
    arch: {{
      idx: 0,
      slides: [
        {{ label: '1. Fluxo do Pipeline', src: 'assets/img/fluxo_trabalho_e_decisoes_arquitetura.png', capPt: '<strong>Figura 1 (1/9):</strong> Fluxo de Trabalho End-to-End e Decisões Metodológicas do Pipeline.', capEn: '<strong>Figure 1 (1/9):</strong> End-to-End Workflow and Methodological Pipeline Decisions.' }},
        {{ label: '2. Painel 7 Modelos', src: 'assets/img/arquitetura_painel_comparativo_7_modelos.png', capPt: '<strong>Figura 2 (2/9):</strong> Painel Comparativo das 7 Arquiteturas Avaliadas ($Z = 32$).', capEn: '<strong>Figure 2 (2/9):</strong> Comparative Panel of the 7 Evaluated Architectures ($Z = 32$).' }},
        {{ label: '3. AE_CNN1D', src: 'assets/img/arquitetura_AE_CNN1D.png', capPt: '<strong>Figura 3 (3/9) — AE_CNN1D:</strong> Grafo Computacional Autograd (torchviz) e Resumo Estrutural (torchinfo.summary).', capEn: '<strong>Figure 3 (3/9) — AE_CNN1D:</strong> Autograd Computational Graph (torchviz) and Structural Summary (torchinfo.summary).' }},
        {{ label: '4. AE_LSTM1D', src: 'assets/img/arquitetura_AE_LSTM1D.png', capPt: '<strong>Figura 4 (4/9) — AE_LSTM1D:</strong> Patch-LSTM Bidirecional 1D (torchviz + torchinfo.summary).', capEn: '<strong>Figure 4 (4/9) — AE_LSTM1D:</strong> Bidirectional 1D Patch-LSTM (torchviz + torchinfo.summary).' }},
        {{ label: '5. AE_Transformer1D', src: 'assets/img/arquitetura_AE_Transformer1D.png', capPt: '<strong>Figura 5 (5/9) — AE_Transformer1D:</strong> Patch-Transformer 1D com Multi-Head Self-Attention (torchviz + torchinfo.summary).', capEn: '<strong>Figure 5 (5/9) — AE_Transformer1D:</strong> 1D Patch-Transformer with Multi-Head Self-Attention (torchviz + torchinfo.summary).' }},
        {{ label: '6. AE_CNN2D_GAF', src: 'assets/img/arquitetura_AE_CNN2D_GAF.png', capPt: '<strong>Figura 6 (6/9) — AE_CNN2D_GAF:</strong> Autoencoder Convolucional 2D operando exclusivamente sobre GADF 32×32 (torchviz + torchinfo.summary).', capEn: '<strong>Figure 6 (6/9) — AE_CNN2D_GAF:</strong> 2D Convolutional Autoencoder operating exclusively on 32×32 GADF (torchviz + torchinfo.summary).' }},
        {{ label: '7. Hybrid_CNN1D_CNN2D', src: 'assets/img/arquitetura_Hybrid_CNN1D_CNN2D.png', capPt: '<strong>Figura 7 (7/9) — Hybrid_CNN1D_CNN2D:</strong> Fusão Multimodal CNN-1D + CNN-2D GADF (torchviz + torchinfo.summary).', capEn: '<strong>Figure 7 (7/9) — Hybrid_CNN1D_CNN2D:</strong> Multimodal Fusion CNN-1D + CNN-2D GADF (torchviz + torchinfo.summary).' }},
        {{ label: '8. Hybrid_LSTM1D_CNN2D', src: 'assets/img/arquitetura_Hybrid_LSTM1D_CNN2D.png', capPt: '<strong>Figura 8 (8/9) — Hybrid_LSTM1D_CNN2D:</strong> Fusão Multimodal LSTM-1D + CNN-2D GADF (torchviz + torchinfo.summary).', capEn: '<strong>Figure 8 (8/9) — Hybrid_LSTM1D_CNN2D:</strong> Multimodal Fusion LSTM-1D + CNN-2D GADF (torchviz + torchinfo.summary).' }},
        {{ label: '9. Hybrid_Transformer_CNN2D', src: 'assets/img/arquitetura_Hybrid_Transformer_CNN2D.png', capPt: '<strong>Figura 9 (9/9) — Hybrid_Transformer_CNN2D:</strong> Fusão Multimodal Transformer-1D + CNN-2D GADF (torchviz + torchinfo.summary).', capEn: '<strong>Figure 9 (9/9) — Hybrid_Transformer_CNN2D:</strong> Multimodal Fusion Transformer-1D + CNN-2D GADF (torchviz + torchinfo.summary).' }}
      ]
    }},
    fig10: {{
      idx: 0,
      slides: [
        {{ label: '1. AE_CNN1D', src: 'assets/img/fig10_loss_AE_CNN1D.png', capPt: '<strong>Figura 10 (1/8) — AE_CNN1D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (1/8) — AE_CNN1D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '2. AE_LSTM1D', src: 'assets/img/fig10_loss_AE_LSTM1D.png', capPt: '<strong>Figura 10 (2/8) — AE_LSTM1D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (2/8) — AE_LSTM1D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '3. AE_Transformer1D', src: 'assets/img/fig10_loss_AE_Transformer1D.png', capPt: '<strong>Figura 10 (3/8) — AE_Transformer1D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (3/8) — AE_Transformer1D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '4. AE_CNN2D_GAF', src: 'assets/img/fig10_loss_AE_CNN2D_GAF.png', capPt: '<strong>Figura 10 (4/8) — AE_CNN2D_GAF:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (4/8) — AE_CNN2D_GAF:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '5. Hybrid_CNN1D_CNN2D', src: 'assets/img/fig10_loss_Hybrid_CNN1D_CNN2D.png', capPt: '<strong>Figura 10 (5/8) — Hybrid_CNN1D_CNN2D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (5/8) — Hybrid_CNN1D_CNN2D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '6. Hybrid_LSTM1D_CNN2D', src: 'assets/img/fig10_loss_Hybrid_LSTM1D_CNN2D.png', capPt: '<strong>Figura 10 (6/8) — Hybrid_LSTM1D_CNN2D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (6/8) — Hybrid_LSTM1D_CNN2D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '7. Hybrid_Transformer_CNN2D', src: 'assets/img/fig10_loss_Hybrid_Transformer_CNN2D.png', capPt: '<strong>Figura 10 (7/8) — Hybrid_Transformer_CNN2D:</strong> Curva de Perda (Loss MSE) de Treino vs. Validação (±1 DP nos 5 Folds).', capEn: '<strong>Figure 10 (7/8) — Hybrid_Transformer_CNN2D:</strong> Training vs. Validation Reconstruction Loss Curve (MSE ±1 SD across 5 Folds).' }},
        {{ label: '8. Comparativo 7 Modelos', src: 'assets/img/fig10_loss_Comparativo_7_Modelos.png', capPt: '<strong>Figura 10 (8/8) — Comparativo Direto:</strong> Evolução da Loss de Validação Out-of-Fold entre todas as 7 arquiteturas.', capEn: '<strong>Figure 10 (8/8) — Direct Comparison:</strong> Out-of-Fold Validation Loss evolution across all 7 architectures.' }}
      ]
    }},
    fig11: {{
      idx: 0,
      slides: [
        {{ label: 'Bomba_1305', src: 'assets/img/fig11_loss_Bomba_1305.png', capPt: '<strong>Figura 11 (1/7) — Bomba_1305:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (1/7) — Bomba_1305:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Bomba_1307', src: 'assets/img/fig11_loss_Bomba_1307.png', capPt: '<strong>Figura 11 (2/7) — Bomba_1307:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (2/7) — Bomba_1307:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Bomba_2307', src: 'assets/img/fig11_loss_Bomba_2307.png', capPt: '<strong>Figura 11 (3/7) — Bomba_2307:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (3/7) — Bomba_2307:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Bomba_3307', src: 'assets/img/fig11_loss_Bomba_3307.png', capPt: '<strong>Figura 11 (4/7) — Bomba_3307:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (4/7) — Bomba_3307:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Bomba_4307', src: 'assets/img/fig11_loss_Bomba_4307.png', capPt: '<strong>Figura 11 (5/7) — Bomba_4307:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (5/7) — Bomba_4307:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Bomba_5307', src: 'assets/img/fig11_loss_Bomba_5307.png', capPt: '<strong>Figura 11 (6/7) — Bomba_5307:</strong> Convergência da Loss de Treino e Validação nos 5 Folds.', capEn: '<strong>Figure 11 (6/7) — Bomba_5307:</strong> Training & Validation Loss Convergence across 5 Folds.' }},
        {{ label: 'Dataset_Completo', src: 'assets/img/fig11_loss_Dataset_Completo.png', capPt: '<strong>Figura 11 (7/7) — Dataset_Completo:</strong> Convergência da Loss de Treino e Validação para as 6 bombas concatenadas.', capEn: '<strong>Figure 11 (7/7) — Dataset_Completo:</strong> Training & Validation Loss Convergence for all 6 concatenated slurry pumps.' }}
      ]
    }},
    fig12: {{
      idx: 0,
      slides: [
        {{ label: 'Bomba_1305', src: 'assets/img/fig12_cm_Bomba_1305.png', capPt: '<strong>Figura 12 (1/7) — Bomba_1305:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (1/7) — Bomba_1305:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Bomba_1307', src: 'assets/img/fig12_cm_Bomba_1307.png', capPt: '<strong>Figura 12 (2/7) — Bomba_1307:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (2/7) — Bomba_1307:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Bomba_2307', src: 'assets/img/fig12_cm_Bomba_2307.png', capPt: '<strong>Figura 12 (3/7) — Bomba_2307:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (3/7) — Bomba_2307:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Bomba_3307', src: 'assets/img/fig12_cm_Bomba_3307.png', capPt: '<strong>Figura 12 (4/7) — Bomba_3307:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (4/7) — Bomba_3307:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Bomba_4307', src: 'assets/img/fig12_cm_Bomba_4307.png', capPt: '<strong>Figura 12 (5/7) — Bomba_4307:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (5/7) — Bomba_4307:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Bomba_5307', src: 'assets/img/fig12_cm_Bomba_5307.png', capPt: '<strong>Figura 12 (6/7) — Bomba_5307:</strong> Matriz de Confusão Out-of-Fold (5 Folds) em 100% do dataset anual.', capEn: '<strong>Figure 12 (6/7) — Bomba_5307:</strong> Out-of-Fold Confusion Matrix (5 Folds) on 100% of the annual dataset.' }},
        {{ label: 'Dataset_Completo', src: 'assets/img/fig12_cm_Dataset_Completo.png', capPt: '<strong>Figura 12 (7/7) — Dataset_Completo:</strong> Matriz de Confusão Out-of-Fold (5 Folds) nas 6 bombas concatenadas.', capEn: '<strong>Figure 12 (7/7) — Dataset_Completo:</strong> Out-of-Fold Confusion Matrix (5 Folds) across all 6 concatenated pumps.' }}
      ]
    }},
    fig13: {{
      idx: 0,
      slides: [
        {{ label: '13A: ROC-AUC por Janela', src: 'assets/img/fig13a_janela_roc_auc.png', capPt: '<strong>Figura 13A (1/4):</strong> Comparação de ROC-AUC das 7 arquiteturas nas escalas de 15min, 30min, 1h e 2h.', capEn: '<strong>Figure 13A (1/4):</strong> ROC-AUC comparison across the 7 architectures for 15min, 30min, 1h, and 2h windows.' }},
        {{ label: '13B: PR-AUC por Janela', src: 'assets/img/fig13b_janela_pr_auc.png', capPt: '<strong>Figura 13B (2/4):</strong> Comparação de PR-AUC das 7 arquiteturas nas escalas de 15min, 30min, 1h e 2h.', capEn: '<strong>Figure 13B (2/4):</strong> PR-AUC comparison across the 7 architectures for 15min, 30min, 1h, and 2h windows.' }},
        {{ label: '13C: F1-Score por Janela', src: 'assets/img/fig13c_janela_f1_score.png', capPt: '<strong>Figura 13C (3/4):</strong> Comparação de F1-Score das 7 arquiteturas nas escalas de 15min, 30min, 1h e 2h.', capEn: '<strong>Figure 13C (3/4):</strong> F1-Score comparison across the 7 architectures for 15min, 30min, 1h, and 2h windows.' }},
        {{ label: '13D: Falsos Alarmes (%)', src: 'assets/img/fig13d_janela_falsos_alarmes.png', capPt: '<strong>Figura 13D (4/4):</strong> Taxa de Falsos Alarmes (%) das 7 arquiteturas nas escalas de 15min, 30min, 1h e 2h.', capEn: '<strong>Figure 13D (4/4):</strong> False Alarm Rate (%) across the 7 architectures for 15min, 30min, 1h, and 2h windows.' }}
      ]
    }},
    fig14: {{
      idx: 0,
      slides: [
        {{ label: '14A: ROC-AUC & PR-AUC', src: 'assets/img/fig14a_rotulos_roc_pr.png', capPt: '<strong>Figura 14A (1/4):</strong> ROC-AUC e PR-AUC nas 11 Formulações de Rótulos (Instantâneo, Futuro, Alerta Precoce e Kernel Gaussiano).', capEn: '<strong>Figure 14A (1/4):</strong> ROC-AUC and PR-AUC across the 11 Label Formulations (Instantaneous, Future, Early Warning, and Gaussian Kernel).' }},
        {{ label: '14B: F1, Precisão & Recall', src: 'assets/img/fig14b_rotulos_f1_prec_rec.png', capPt: '<strong>Figura 14B (2/4):</strong> F1-Score, Precisão e Recall nas 11 Formulações de Rótulos.', capEn: '<strong>Figure 14B (2/4):</strong> F1-Score, Precision, and Recall across the 11 Label Formulations.' }},
        {{ label: '14C: Correlação de Spearman', src: 'assets/img/fig14c_rotulos_spearman.png', capPt: '<strong>Figura 14C (3/4):</strong> Correlação de Spearman (ρ) entre a probabilidade predita pelo modelo e a evolução real do risco.', capEn: '<strong>Figure 14C (3/4):</strong> Spearman Rank Correlation (ρ) between predicted probability and true risk evolution.' }},
        {{ label: '14D: Falsos Alarmes & MAE (%)', src: 'assets/img/fig14d_rotulos_far_mae.png', capPt: '<strong>Figura 14D (4/4):</strong> Taxa de Falsos Alarmes (%) e Erro Médio Absoluto da Probabilidade (%) nas 11 Formulações.', capEn: '<strong>Figure 14D (4/4):</strong> False Alarm Rate (%) and Probability Mean Absolute Error (%) across the 11 Formulations.' }}
      ]
    }},
    fig15: {{
      idx: 0,
      slides: [
        {{ label: 'Bomba_1305', src: 'assets/img/fig15_hist_Bomba_1305.png', capPt: '<strong>Figura 15 (1/7) — Bomba_1305:</strong> Histograma do Erro de Reconstrução mostrando Zona Verde (Normal Automático), Zona Amarela (Revisão do Operador: 15,8% das janelas, capturando 73,2% dos FP+FN) e Zona Vermelha (Anomalia Automática). F1 salta de 0.8697 para 0.9647.', capEn: '<strong>Figure 15 (1/7) — Bomba_1305:</strong> Reconstruction Error Histogram showing Green Zone (Auto Normal), Yellow Zone (Operator Review: 15.8% of windows, capturing 73.2% of FP+FN), and Red Zone (Auto Anomaly). F1 jumps from 0.8697 to 0.9647.' }},
        {{ label: 'Bomba_1307', src: 'assets/img/fig15_hist_Bomba_1307.png', capPt: '<strong>Figura 15 (2/7) — Bomba_1307:</strong> Histograma do Erro de Reconstrução (AE_Transformer1D). Zona do Operador: 14,0% das janelas, capturando 79,5% dos FP+FN. F1 salta de 0.9524 para 0.9901.', capEn: '<strong>Figure 15 (2/7) — Bomba_1307:</strong> Reconstruction Error Histogram (AE_Transformer1D). Operator Zone: 14.0% of windows, capturing 79.5% of FP+FN. F1 jumps from 0.9524 to 0.9901.' }},
        {{ label: 'Bomba_2307', src: 'assets/img/fig15_hist_Bomba_2307.png', capPt: '<strong>Figura 15 (3/7) — Bomba_2307:</strong> Histograma do Erro de Reconstrução (AE_Transformer1D). Zona do Operador: 13,5% das janelas, capturando 79,0% dos FP+FN. F1 salta de 0.9517 para 0.9898.', capEn: '<strong>Figure 15 (3/7) — Bomba_2307:</strong> Reconstruction Error Histogram (AE_Transformer1D). Operator Zone: 13.5% of windows, capturing 79.0% of FP+FN. F1 jumps from 0.9517 to 0.9898.' }},
        {{ label: 'Bomba_3307', src: 'assets/img/fig15_hist_Bomba_3307.png', capPt: '<strong>Figura 15 (4/7) — Bomba_3307:</strong> Histograma do Erro de Reconstrução (AE_LSTM1D). Zona do Operador: apenas 9,3% das janelas, capturando 68,0% dos FP+FN. F1 salta de 0.9588 para 0.9868.', capEn: '<strong>Figure 15 (4/7) — Bomba_3307:</strong> Reconstruction Error Histogram (AE_LSTM1D). Operator Zone: only 9.3% of windows, capturing 68.0% of FP+FN. F1 jumps from 0.9588 to 0.9868.' }},
        {{ label: 'Bomba_4307', src: 'assets/img/fig15_hist_Bomba_4307.png', capPt: '<strong>Figura 15 (5/7) — Bomba_4307:</strong> Histograma do Erro de Reconstrução (AE_LSTM1D). Zona do Operador: 17,3% das janelas, capturando 78,7% dos FP+FN. F1 salta de 0.9217 para 0.9831.', capEn: '<strong>Figure 15 (5/7) — Bomba_4307:</strong> Reconstruction Error Histogram (AE_LSTM1D). Operator Zone: 17.3% of windows, capturing 78.7% of FP+FN. F1 jumps from 0.9217 to 0.9831.' }},
        {{ label: 'Bomba_5307', src: 'assets/img/fig15_hist_Bomba_5307.png', capPt: '<strong>Figura 15 (6/7) — Bomba_5307:</strong> Histograma do Erro de Reconstrução (AE_LSTM1D). Zona do Operador: 13,9% das janelas, capturando 74,2% dos FP+FN. F1 salta de 0.8912 para 0.9719.', capEn: '<strong>Figure 15 (6/7) — Bomba_5307:</strong> Reconstruction Error Histogram (AE_LSTM1D). Operator Zone: 13.9% of windows, capturing 74.2% of FP+FN. F1 jumps from 0.8912 to 0.9719.' }},
        {{ label: 'Dataset_Completo', src: 'assets/img/fig15_hist_Dataset_Completo.png', capPt: '<strong>Figura 15 (7/7) — Dataset_Completo (6 Bombas Concatenadas):</strong> Histograma do Erro de Reconstrução (AE_LSTM1D). Zona do Operador: 18,8% das janelas, capturando 75,5% dos FP+FN. F1 salta de 0.9204 para 0.9804.', capEn: '<strong>Figure 15 (7/7) — Dataset_Completo (All 6 Pumps Concatenated):</strong> Reconstruction Error Histogram (AE_LSTM1D). Operator Zone: 18.8% of windows, capturing 75.5% of FP+FN. F1 jumps from 0.9204 to 0.9804.' }}
      ]
    }},
    fig16: {{
      idx: 0,
      slides: [
        {{ label: '16A: Ganho de F1-Score / F1 Gain', src: 'assets/img/fig16a_ganho_f1_operador.png', capPt: '<strong>Figura 16A (1/2) — Ganho de F1-Score por Arquitetura:</strong> Comparação direta entre Limiar Único no Histograma (cinza), Só Decisões Automáticas nas Zonas 1 e 3 (azul) e Sistema Final com Revisão do Operador na Zona 2 (verde).', capEn: '<strong>Figure 16A (1/2) — F1-Score Gain by Architecture:</strong> Direct comparison between Single Histogram Threshold (grey), Auto-Only Decisions in Zones 1 & 3 (blue), and Final System with Operator Review in Zone 2 (green).' }},
        {{ label: '16B: Captura de FP+FN vs. Esforço / Error Capture vs. Effort', src: 'assets/img/fig16b_captura_fp_fn_operador.png', capPt: '<strong>Figura 16B (2/2) — Eficiência da Triagem do Operador:</strong> Porcentagem de erros (FP+FN) capturados na Zona Intermediária (~74,7% no AE_LSTM1D) versus porcentagem total de janelas enviadas para revisão humana (~14,9%).', capEn: '<strong>Figure 16B (2/2) — Operator Triage Efficiency:</strong> Percentage of total errors (FP+FN) captured inside the Intermediate Zone (~74.7% for AE_LSTM1D) versus percentage of windows routed for human review (~14.9%).' }}
      ]
    }}
  }};

  function renderViewer(key) {{
    const v = VIEWERS[key];
    const s = v.slides[v.idx];
    document.getElementById(key + 'Img').src = s.src;
    document.getElementById(key + 'Counter').textContent = (v.idx + 1) + ' / ' + v.slides.length;
    document.getElementById(key + 'Cap').innerHTML = `<span data-lang="pt">${{s.capPt}}</span><span data-lang="en">${{s.capEn}}</span>`;
    const tabs = document.getElementById(key + 'Tabs');
    tabs.innerHTML = v.slides.map((item, i) =>
      `<button type="button" class="v-tab ${{i === v.idx ? 'active' : ''}}" onclick="setViewer('${{key}}', ${{i}})">${{item.label}}</button>`
    ).join('');
  }}
  function setViewer(key, i) {{ VIEWERS[key].idx = i; renderViewer(key); }}
  function stepViewer(key, delta) {{
    const v = VIEWERS[key];
    v.idx = (v.idx + delta + v.slides.length) % v.slides.length;
    renderViewer(key);
  }}
  function stepFig15(d) {{ stepViewer('fig15', d); }}
  function stepFig16(d) {{ stepViewer('fig16', d); }}

  window.addEventListener('DOMContentLoaded', () => {{
    Object.keys(VIEWERS).forEach(k => renderViewer(k));
    try {{
      const savedLang = localStorage.getItem('site_lang_pref');
      if (savedLang === 'en' || savedLang === 'pt') {{
        setLanguage(savedLang);
      }}
    }} catch (e) {{}}
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

    with open(os.path.join(DOCS_DIR, ".nojekyll"), "w", encoding="utf-8") as f:
        f.write("")

    print(f"[OK] Site Bilíngue (PT/EN) gerado com sucesso em: {index_path}")
    print(f"[OK] Total de figuras sincronizadas em docs/assets/img/: {len(copied_imgs)}")


if __name__ == "__main__":
    build_site()
