from pathlib import Path
import numpy as np
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    """Min-max scaling robusto: ignora NaN, mantém NaN."""
    s = s.astype(float)
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([np.nan] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"

    # pega o snis_sp_XXXX.parquet mais recente
    arquivos = sorted(processed_dir.glob("snis_sp_*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {processed_dir}. Rode o 01_preparacao primeiro.")

    parquet_path = arquivos[-1]
    print("Lendo:", parquet_path)

    df = pd.read_parquet(parquet_path)
    ano = int(df["ano"].max())
    print("Ano base:", ano, " | shape:", df.shape)

    # --- Colunas finais (nível projeto sério)
    cols = [
        "id_municipio", "ano",
        "indice_atendimento_total_agua",
        "indice_atendimento_esgoto_agua",
        "indice_coleta_esgoto",
        "indice_tratamento_esgoto",
        "indice_perda_distribuicao_agua",
        "indice_consumo_agua_per_capita",
        "extensao_rede_agua",
        "extensao_rede_esgoto",
        "quantidade_ligacao_ativa_agua",
        "quantidade_ligacao_ativa_esgoto",
        "investimento_total_municipio",
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas faltando no parquet: {missing}")

    base = df[cols].copy()

    # --- Diagnóstico de nulos do “base”
    null_pct = (base.isna().mean() * 100).sort_values(ascending=False)
    print("\nNulos (%) no conjunto base:")
    print(null_pct)

    # --- Features para índice (quanto maior, melhor) / (quanto menor, melhor)
    # "bom alto"
    good_high = {
        "indice_atendimento_total_agua": 0.20,
        "indice_atendimento_esgoto_agua": 0.20,
        "indice_coleta_esgoto": 0.20,
        "indice_tratamento_esgoto": 0.20,
    }

    # "bom baixo" (vamos inverter após normalizar)
    good_low = {
        "indice_perda_distribuicao_agua": 0.10,  # menor perda é melhor
    }

    # opcional (não entra no índice por enquanto; entra na priorização)
    invest_col = "investimento_total_municipio"

    # --- normalização
    norm = pd.DataFrame(index=base.index)

    for col in good_high:
        norm[col] = minmax(base[col])

    for col in good_low:
        norm[col] = 1 - minmax(base[col])  # inverte: menor -> maior score

    # --- índice ponderado (re-normaliza pesos para ignorar NaN por linha)
    weights = {**good_high, **good_low}

    weight_sum = pd.Series(0.0, index=base.index)
    score_sum = pd.Series(0.0, index=base.index)

    for col, w in weights.items():
        valid = norm[col].notna()
        score_sum[valid] += norm.loc[valid, col] * w
        weight_sum[valid] += w

    base["indice_saneamento_0_100"] = np.where(weight_sum > 0, (score_sum / weight_sum) * 100, np.nan)

    # --- Ranking (maior índice = melhor saneamento; para prioridade usamos o inverso)
    base["ranking_saneamento"] = base["indice_saneamento_0_100"].rank(ascending=False, method="dense")

    # --- Prioridade (quanto menor o índice, maior a prioridade)
    base["prioridade_intervencao"] = base["indice_saneamento_0_100"].rank(ascending=True, method="dense")

    # --- Prioridade ajustada por investimento (se investimento baixo e índice baixo -> atenção)
    # regra simples: normaliza investimento e cria um score de "risco" (baixo índice + baixo investimento)
    inv_norm = minmax(base[invest_col].fillna(0))
    base["investimento_norm_0_1"] = inv_norm

    # risco: (1 - indice_norm) * (1 - inv_norm)
    idx_norm = base["indice_saneamento_0_100"] / 100
    base["risco_saneamento_baixo_investimento"] = (1 - idx_norm) * (1 - inv_norm)

    # --- salvar outputs
    out_dir = processed_dir
    out_csv = out_dir / f"snis_sp_{ano}_indice_ranking.csv"
    out_parquet = out_dir / f"snis_sp_{ano}_indice_ranking.parquet"

    base.to_csv(out_csv, index=False, encoding="utf-8")
    base.to_parquet(out_parquet, index=False)

    print("\nSalvo:")
    print("-", out_csv)
    print("-", out_parquet)

    # --- Top 15 piores (para virar insight no README)
    worst = base.sort_values("indice_saneamento_0_100").head(15)[
        ["id_municipio", "indice_saneamento_0_100", "prioridade_intervencao", invest_col, "risco_saneamento_baixo_investimento"]
    ]
    print("\nTop 15 piores (menor índice):")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
