# Análise Visual – Saneamento Básico em SP (2022)
#
# Notebook voltado à interpretação dos resultados do índice (0–100)
# de priorização, com foco em tomada de decisão.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_parquet("../data/processed/snis_sp_2022_indice_ranking.parquet")
df.head()

plt.figure(figsize=(8,5))
sns.histplot(df["indice_saneamento_0_100"], bins=20, kde=True)
plt.title("Distribuição do Índice de Saneamento – SP (2022)")
plt.xlabel("Índice (0–100)")
plt.ylabel("Municípios")
plt.show()

top10 = df.sort_values("indice_saneamento_0_100").head(10)

plt.figure(figsize=(8,5))
sns.barplot(
    data=top10,
    x="indice_saneamento_0_100",
    y="id_municipio",
    orient="h"
)
plt.title("Top 10 Municípios com Pior Índice de Saneamento – SP (2022)")
plt.xlabel("Índice de Saneamento")
plt.ylabel("Município")
plt.show()

df_plot = df[df["investimento_total_municipio"] > 0]

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df_plot,
    x="investimento_total_municipio",
    y="indice_saneamento_0_100"
)
plt.xscale("log")
plt.title("Investimento Municipal × Índice de Saneamento (escala log)")
plt.xlabel("Investimento Total do Município")
plt.ylabel("Índice de Saneamento")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["risco_saneamento_baixo_investimento"], bins=20)
plt.title("Distribuição do Risco de Baixo Saneamento com Baixo Investimento")
plt.xlabel("Risco (0–1)")
plt.ylabel("Municípios")
plt.show()

# Conclusão
#
# A análise visual indica que a maioria dos municípios apresenta
# desempenho intermediário ou alto em saneamento.
#
# Entretanto, existe um grupo reduzido com índices significativamente
# baixos, concentrando maior risco associado à ausência de investimento.
#
# Além disso, não há relação linear direta entre investimento e desempenho,
# indicando que eficiência na aplicação dos recursos é fator crítico.
