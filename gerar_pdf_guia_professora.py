"""
Converte o Guia Didático da Professora (Guia_Didatico_Professora_Projeto_Bombas.md)
em um arquivo PDF vetorial de alta resolução (Guia_Didatico_Professora_Projeto_Bombas.pdf),
incluindo todas as tabelas, explicações de tensores (Entrada -> Gargalo -> Saída),
diagramas de arquitetura e matrizes de confusão.
"""

import os
import re
import subprocess

WORKSPACE = os.path.abspath(".")
MD_PATH = os.path.join(WORKSPACE, "Guia_Didatico_Professora_Projeto_Bombas.md")
HTML_PATH = os.path.join(WORKSPACE, "_temp_guia_professora.html")
PDF_PATH = os.path.join(WORKSPACE, "Guia_Didatico_Professora_Projeto_Bombas.pdf")
FIG_ARCH_DIR = os.path.join(WORKSPACE, "Dataset_completo_bombas", "Figuras_Arquiteturas")
FIG_ANOM_DIR = os.path.join(
    WORKSPACE,
    "Dataset_completo_bombas",
    "Resultados_Modelos_1D_2D",
    "Deteccao_Anomalias_Operador",
    "Figuras_Anomalias_Operador",
)


def format_inline(text: str) -> str:
    # Code spans
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    # Inline math cleanup for clean HTML display
    text = text.replace(r"\approx", "≈").replace(r"\ge", "≥").replace(r"\le", "≤")
    text = text.replace(r"\times", "×").replace(r"\cdot", "·").replace(r"\rightarrow", "→")
    text = text.replace(r"\Rightarrow", "⇒").replace(r"\in", "∈").replace(r"\mathbb{R}", "ℝ")
    text = text.replace(r"\sigma", "σ").replace(r"\phi", "φ").replace(r"\pi", "π")
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]+)\}", r"<strong>\1</strong>", text)
    text = re.sub(r"\$([^$]+)\$", r"<span class='math'>\1</span>", text)
    return text


def md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    in_ul = False
    in_ol = False
    in_blockquote = False

    def close_blocks():
        nonlocal in_table, in_ul, in_ol, in_blockquote
        if in_table:
            html_lines.append("</tbody></table>")
            in_table = False
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False
        if in_blockquote:
            html_lines.append("</blockquote>")
            in_blockquote = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            close_blocks()
            i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            close_blocks()
            eq = format_inline(stripped[2:-2].strip())
            html_lines.append(f"<div class='equation'>{eq}</div>")
            i += 1
            continue

        if stripped == "---":
            close_blocks()
            html_lines.append("<hr/>")
            i += 1
            continue

        # Headers
        if stripped.startswith("# "):
            close_blocks()
            html_lines.append(f"<h1>{format_inline(stripped[2:])}</h1>")
            i += 1
            continue
        if stripped.startswith("## "):
            close_blocks()
            title_txt = stripped[3:]
            html_lines.append(f"<h2>{format_inline(title_txt)}</h2>")
            # Injeta diagramas nos módulos correspondentes
            if "Módulo 2" in title_txt:
                f1 = os.path.join(FIG_ARCH_DIR, "fluxo_trabalho_e_decisoes_arquitetura.png").replace("\\", "/")
                html_lines.append(f"<div class='fig-box'><img src='file:///{f1}'/><div class='cap'>Figura 1 — Fluxo de Trabalho Completo e Decisões Metodológicas do Pipeline.</div></div>")
            elif "Módulo 3" in title_txt:
                for idx_f, (fn, cap) in enumerate([
                    ("arquitetura_painel_comparativo_7_modelos.png", "Painel Comparativo das 7 Arquiteturas Avaliadas (Z = 32)"),
                    ("arquitetura_AE_CNN1D.png", "Arquitetura 1: AE_CNN1D (torchviz + torchinfo.summary)"),
                    ("arquitetura_AE_LSTM1D.png", "Arquitetura 2: AE_LSTM1D (torchviz + torchinfo.summary)"),
                    ("arquitetura_AE_Transformer1D.png", "Arquitetura 3: AE_Transformer1D (torchviz + torchinfo.summary)"),
                    ("arquitetura_AE_CNN2D_GAF.png", "Arquitetura 4: AE_CNN2D_GAF (Somente GADF — torchviz + torchinfo.summary)"),
                    ("arquitetura_Hybrid_CNN1D_CNN2D.png", "Arquitetura 5: Hybrid_CNN1D_CNN2D (torchviz + torchinfo.summary)"),
                    ("arquitetura_Hybrid_LSTM1D_CNN2D.png", "Arquitetura 6: Hybrid_LSTM1D_CNN2D (torchviz + torchinfo.summary)"),
                    ("arquitetura_Hybrid_Transformer_CNN2D.png", "Arquitetura 7: Hybrid_Transformer_CNN2D (torchviz + torchinfo.summary)"),
                ], start=2):
                    fp = os.path.join(FIG_ARCH_DIR, fn).replace("\\", "/")
                    html_lines.append(f"<div class='fig-box'><img src='file:///{fp}'/><div class='cap'>Figura {idx_f} — {cap}.</div></div>")
            elif "Módulo 6" in title_txt:
                cm_p = os.path.join(FIG_ANOM_DIR, "matrizes_confusao_ano_acima_2_5_15min.png").replace("\\", "/")
                if os.path.exists(cm_p):
                    html_lines.append(f"<div class='fig-box'><img src='file:///{cm_p}'/><div class='cap'>Figura 10 — Matrizes de Confusão Out-of-Fold (5 Folds) em 100% do Dataset Anual (StdDev ≥ 2,5 A).</div></div>")
            elif "Módulo 9" in title_txt:
                c11_p = os.path.join(FIG_ANOM_DIR, "comparacao_11_abordagens_rotulos.png").replace("\\", "/")
                if os.path.exists(c11_p):
                    html_lines.append(f"<div class='fig-box'><img src='file:///{c11_p}'/><div class='cap'>Figura 11 — Comparação Global das 11 Formulações de Rótulo (Estado Atual, Futuro, Alerta Precoce e Distribuição de Probabilidade).</div></div>")
            i += 1
            continue
        if stripped.startswith("### "):
            close_blocks()
            html_lines.append(f"<h3>{format_inline(stripped[4:])}</h3>")
            i += 1
            continue
        if stripped.startswith("#### "):
            close_blocks()
            html_lines.append(f"<h4>{format_inline(stripped[5:])}</h4>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            if not in_blockquote:
                close_blocks()
                html_lines.append("<blockquote>")
                in_blockquote = True
            html_lines.append(f"<p>{format_inline(stripped.lstrip('> ').strip())}</p>")
            i += 1
            continue

        # Table
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if i + 1 < len(lines) and re.match(r"^\|[\s:\-\|]+\|$", lines[i + 1].strip()):
                close_blocks()
                html_lines.append("<table><thead><tr>")
                for c in cells:
                    html_lines.append(f"<th>{format_inline(c)}</th>")
                html_lines.append("</tr></thead><tbody>")
                in_table = True
                i += 2
                continue
            elif in_table:
                html_lines.append("<tr>")
                for c in cells:
                    html_lines.append(f"<td>{format_inline(c)}</td>")
                html_lines.append("</tr>")
                i += 1
                continue

        # Lists
        if stripped.startswith("* ") or stripped.startswith("- "):
            if not in_ul:
                close_blocks()
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{format_inline(stripped[2:])}</li>")
            i += 1
            continue

        m_ol = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m_ol:
            if not in_ol:
                close_blocks()
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{format_inline(m_ol.group(1))}</li>")
            i += 1
            continue

        close_blocks()
        html_lines.append(f"<p>{format_inline(stripped)}</p>")
        i += 1

    close_blocks()
    return "\n".join(html_lines)


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    body_html = md_to_html(md_content)

    full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<title>Guia: Detecção de Anomalias em Bombas Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais</title>
<style>
  @page {{
    size: A4 landscape;
    margin: 12mm 14mm 12mm 14mm;
  }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.42;
    color: #0F172A;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 16pt;
    color: #0F172A;
    border-bottom: 2.5px solid #1E40AF;
    padding-bottom: 6px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 13pt;
    color: #1E3A8A;
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    padding: 5px 10px;
    margin-top: 16px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 11pt;
    color: #0F172A;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 10pt;
    color: #1D4ED8;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  p, li {{
    margin: 3px 0;
  }}
  ul, ol {{
    margin: 4px 0 8px 18px;
    padding: 0;
  }}
  blockquote {{
    background: #FEF3C7;
    border-left: 4px solid #D97706;
    margin: 8px 0;
    padding: 6px 12px;
    border-radius: 4px;
  }}
  code {{
    font-family: Consolas, monospace;
    background: #F1F5F9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.8pt;
    color: #0F172A;
  }}
  .equation {{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    padding: 6px 12px;
    margin: 6px 0;
    text-align: center;
    font-weight: bold;
    border-radius: 4px;
  }}
  .math {{
    font-style: italic;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 8.2pt;
    page-break-inside: avoid;
  }}
  th {{
    background: #1E293B;
    color: #FFFFFF;
    padding: 5px 6px;
    border: 1px solid #334155;
    text-align: center;
  }}
  td {{
    padding: 4px 6px;
    border: 1px solid #CBD5E1;
    text-align: center;
  }}
  tr:nth-child(even) td {{
    background: #F8FAFC;
  }}
  .fig-box {{
    text-align: center;
    margin: 10px 0;
    page-break-inside: avoid;
  }}
  .fig-box img {{
    max-width: 96%;
    max-height: 145mm;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
  }}
  .cap {{
    font-size: 8.2pt;
    color: #475569;
    font-style: italic;
    margin-top: 3px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #CBD5E1;
    margin: 12px 0;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = next((b for b in browsers if os.path.exists(b)), None)
    if not browser_exe:
        raise RuntimeError("Nenhum navegador Chromium encontrado para renderizar o PDF.")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        f"file:///{HTML_PATH.replace(os.sep, '/')}",
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(HTML_PATH):
        os.remove(HTML_PATH)
    print(f"PDF gerado com sucesso: {PDF_PATH} ({os.path.getsize(PDF_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
