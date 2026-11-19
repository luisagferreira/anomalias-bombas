import os
import numpy as np
import pandas as pd

BASE_DIR = "Dataset_completo_bombas"
PUMPS = ["1305", "1307", "2307", "3307", "4307", "5307"]

def main():
    wide_dict = {}
    long_records_2_5 = []
    summary_rows = []

    for pid in PUMPS:
        csv_path = os.path.join(BASE_DIR, "Bombas_por_ID", f"Bomba_{pid}.csv")
        df = pd.read_csv(csv_path)
        curr_col = [c for c in df.columns if "YAR" in c][0]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df[curr_col] = pd.to_numeric(df[curr_col], errors="coerce").astype(np.float64)

        s_yar = df.set_index("Timestamp")[curr_col]
        # Desvio padrão amostral (ddof=1) em blocos fixos de 5 minutos (30 amostras de 10s)
        s_std = s_yar.resample("5min", closed="left", label="left").std(ddof=1).fillna(0.0)

        wide_dict[f"BP_{pid}_StdDev_5min"] = s_std

        n_total = len(s_std)
        n_normal = int((s_std < 2.5).sum())
        n_anom_total = int((s_std >= 2.5).sum())
        n_25_40 = int(((s_std >= 2.5) & (s_std < 4.0)).sum())
        n_40_80 = int(((s_std >= 4.0) & (s_std <= 8.0)).sum())
        n_gt_80 = int((s_std > 8.0).sum())

        s_anom = s_std[s_std >= 2.5]
        summary_rows.append({
            "Bomba": f"BP {pid}",
            "Nome_Bomba": pid,
            "Total_Blocos_5min_Ano": n_total,
            "Blocos_Normais_(<2.5)": n_normal,
            "Total_Anomalias_(>=2.5)": n_anom_total,
            "Taxa_Anomalia_Anual_(>=2.5)_pct": round(100.0 * n_anom_total / n_total, 3),
            "Faixa_[2.5_a_4.0)": n_25_40,
            "Faixa_[4.0_a_8.0]": n_40_80,
            "Faixa_(>8.0_Partida/Parada)": n_gt_80,
            "Media_StdDev_Anomalias_(>=2.5)": round(float(s_anom.mean()), 4) if len(s_anom) else 0.0,
            "Maximo_StdDev_Ano": round(float(s_std.max()), 4),
        })

        for ts, val in s_anom.items():
            val_f = round(float(val), 4)
            if val_f < 4.0:
                faixa = "2.5 a 4.0 (Oscilacao Moderada)"
            elif val_f <= 8.0:
                faixa = "4.0 a 8.0 (Faixa Planilha Abril)"
            else:
                faixa = "> 8.0 (Transiente Forte / Rampa)"
            long_records_2_5.append({
                "Bomba": f"BP {pid}",
                "Nome_Bomba": pid,
                "Data": ts.strftime("%Y-%m-%d"),
                "Hora": ts.strftime("%H:%M:%S"),
                "Timestamp Estimado": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Valor Estimado (StdDev)": val_f,
                "Faixa_Desvio_Padrao": faixa,
            })

    # 1. Tabela Wide Completa (Todos os 85.825 intervalos de 5 min do ano com as 6 bombas)
    df_wide = pd.DataFrame(wide_dict).round(4)
    df_wide.index.name = "Timestamp_5min"
    df_wide = df_wide.reset_index()
    df_wide["Data"] = df_wide["Timestamp_5min"].dt.strftime("%Y-%m-%d")
    df_wide["Hora"] = df_wide["Timestamp_5min"].dt.strftime("%H:%M:%S")
    df_wide["Timestamp_5min"] = df_wide["Timestamp_5min"].dt.strftime("%Y-%m-%d %H:%M:%S")

    std_cols = [f"BP_{pid}_StdDev_5min" for pid in PUMPS]
    df_wide["Max_StdDev_Entre_6_Bombas"] = df_wide[std_cols].max(axis=1).round(4)
    df_wide["Qtd_Bombas_Acima_2_5"] = (df_wide[std_cols] >= 2.5).sum(axis=1)
    df_wide["Anomalia_Planta_(>=2.5)"] = (df_wide["Qtd_Bombas_Acima_2_5"] > 0).astype(int)

    cols_order = ["Timestamp_5min", "Data", "Hora"] + std_cols + [
        "Max_StdDev_Entre_6_Bombas", "Qtd_Bombas_Acima_2_5", "Anomalia_Planta_(>=2.5)"
    ]
    df_wide = df_wide[cols_order]

    wide_csv = os.path.join(BASE_DIR, "planilha_desvio_padrao_5min_ano_todo_completa.csv")
    df_wide.to_csv(wide_csv, index=False)

    # 2. Tabela Long Filtrada (>= 2.5) no formato do Edcarlos
    df_long_2_5 = pd.DataFrame(long_records_2_5)
    df_long_2_5 = df_long_2_5.sort_values(["Timestamp Estimado", "Nome_Bomba"]).reset_index(drop=True)
    long_csv = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_acima_2_5.csv")
    df_long_2_5.to_csv(long_csv, index=False)

    # 3. Resumo por Bomba
    df_summary = pd.DataFrame(summary_rows)
    out_dir = os.path.join(BASE_DIR, "Resultados_Modelos_1D_2D", "Deteccao_Anomalias_Operador")
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "resumo_anual_acima_2_5_por_bomba.csv")
    df_summary.to_csv(summary_csv, index=False)

    # 4. Arquivo Excel (.xlsx) com todas as abas para filtro livre
    xlsx_path = os.path.join(BASE_DIR, "relatorio_anomalias_ano_completo_acima_2_5.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_long_2_5.to_excel(writer, sheet_name="Anomalias >= 2.5 (Ano)", index=False)
        df_wide.to_excel(writer, sheet_name="Serie 5min Ano Todo (6 Bombas)", index=False)
        df_summary.to_excel(writer, sheet_name="Resumo por Faixa (>=2.5)", index=False)

    print("=== RESUMO ANUAL COM LIMIAR >= 2.5 ===")
    print(df_summary.to_string(index=False))
    print(f"\nTotal de registros anomalos (>= 2.5) no ano: {len(df_long_2_5)}")
    print(f"Blocos de 5min do ano em que pelo menos 1 bomba teve StdDev >= 2.5: {df_wide['Anomalia_Planta_(>=2.5)'].sum()} de {len(df_wide)} ({100.0*df_wide['Anomalia_Planta_(>=2.5)'].mean():.2f}%)")
    print(f"\nArquivos salvos:")
    print(f"  1. {wide_csv}")
    print(f"  2. {long_csv}")
    print(f"  3. {xlsx_path}")

if __name__ == "__main__":
    main()
