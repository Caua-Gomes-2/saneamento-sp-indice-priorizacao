from pathlib import Path
import pandas as pd


def main():
    # --- paths
    base_dir = Path(__file__).resolve().parents[1]  # src -> projeto
    data_path = base_dir / "data" / "raw" / "br_mdr_snis_municipio_agua_esgoto.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {data_path}")

    # --- load
    df = pd.read_csv(data_path, sep=",", low_memory=False)

    # --- filter SP
    df_sp = df[df["sigla_uf"] == "SP"].copy()
    print("SP total (todas as linhas/anos):", df_sp.shape)

    # --- last year
    ultimo_ano = int(df_sp["ano"].max())
    print("Último ano SP:", ultimo_ano)

    df_sp_ultimo_ano = df_sp[df_sp["ano"] == ultimo_ano].copy()
    print("SP último ano:", df_sp_ultimo_ano.shape)
    print("Anos únicos:", df_sp_ultimo_ano["ano"].unique())

    # --- save parquet (fast)
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"snis_sp_{ultimo_ano}.parquet"
    df_sp_ultimo_ano.to_parquet(out_path, index=False)
    print("Salvo em:", out_path)

    # --- null diagnostics (top 25)
    null_pct = (df_sp_ultimo_ano.isna().mean() * 100).sort_values(ascending=False)
    print("\nTop 25 colunas com mais nulos (%):")
    print(null_pct.head(25))

    # --- duplicates check
    dups = df_sp_ultimo_ano.duplicated(subset=["id_municipio"]).sum()
    print("\nDuplicados por id_municipio:", dups)


if __name__ == "__main__":
    main()
