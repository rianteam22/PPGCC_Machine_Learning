"""Funções de gráfico e tabela compartilhadas pelos notebooks do projeto.

Convenções (iguais em todos os notebooks):
- um só tom de azul para categorias nominais; cinza para a cauda "Outros" e para referências;
- toda figura vem acompanhada da tabela com os mesmos números;
- decimais com vírgula nos rótulos ("17,5K", "5,7%").
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display

from traducao import rotulo, AUSENTE_PT

AZUL = "#2a78d6"
AZUL_CLARO = "#9cc3ee"
CINZA_OUTROS = "#c3c2b7"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
GRADE = "#e6e5e2"


def configurar_estilo():
    """Aplica o estilo padrão do projeto ao matplotlib e ao pandas."""
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 160)
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": GRADE,
        "axes.linewidth": 1,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": TEXTO_SEC,
        "xtick.color": TEXTO_SEC,
        "ytick.color": TEXTO_SEC,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXTO,
        "legend.frameon": False,
    })


def mostrar(arquivo=None):
    """Mostra a figura atual e, se `arquivo` for informado, salva-a em PNG antes (150 dpi)."""
    if arquivo:
        plt.savefig(arquivo, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def formatar_int(n):
    return f"{int(n):,}".replace(",", ".")


def formatar_pct(p, casas=1):
    if 0 < p < 0.05:
        return "<0,1%"
    return f"{p:.{casas}f}%".replace(".", ",")


def formatar_num(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def n_distintos(serie):
    """Número de valores distintos, sem contar o marcador de ausente."""
    return serie[serie != AUSENTE_PT].nunique()


def classes_ordenadas(serie):
    """Classes da mais frequente para a menos frequente, sem o marcador de ausente."""
    return [c for c in serie.value_counts().index if c != AUSENTE_PT]


def eixo_compacto(ax, eixo="x"):
    fmt = FuncFormatter(lambda v, _: (f"{v / 1000:g}K" if v >= 1000 else f"{v:g}").replace(".", ","))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)


def eixo_percentual(ax, eixo="x"):
    fmt = FuncFormatter(lambda v, _: f"{v:g}%".replace(".", ","))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Distribuições univariadas
# ---------------------------------------------------------------------------
def tabela_frequencia(serie, ordem=None):
    """Contagem e percentual de cada classe, por frequência ou na ordem informada.

    Classes presentes nos dados mas ausentes de `ordem` (ex.: 'Não informado') vão para o fim,
    para que nenhuma linha seja descartada e os percentuais continuem somando 100%."""
    contagem = serie.value_counts(dropna=False)
    if ordem is not None:
        presentes = [c for c in ordem if c in contagem.index]
        restantes = [c for c in contagem.index if c not in presentes]
        contagem = contagem.reindex(presentes + restantes)
    tabela = pd.DataFrame({
        "Contagem": contagem,
        "Percentual (%)": (contagem / contagem.sum() * 100).round(2),
    })
    tabela.index.name = "Classe"
    return tabela


def grafico_barras(contagem, titulo, max_classes=20, rotular=True, xlabel="Contagem", total=None,
                   arquivo=None):
    """Barras horizontais em um só tom. Acima de `max_classes`, a cauda vira 'Outros' (só no gráfico).

    `total` é o denominador dos percentuais nos rótulos (padrão: soma das barras)."""
    contagem = contagem.copy()
    cores = [AZUL] * len(contagem)
    if max_classes is not None and len(contagem) > max_classes:
        principais = contagem.nlargest(max_classes).index
        cauda = contagem.drop(principais)
        contagem = pd.concat([
            contagem[contagem.index.isin(principais)],
            pd.Series({f"Outros ({len(cauda)} classes)": cauda.sum()}),
        ])
        cores = [AZUL] * max_classes + [CINZA_OUTROS]

    total = total if total is not None else contagem.sum()
    posicoes = np.arange(len(contagem))[::-1]
    altura = max(2.4, 0.42 * len(contagem) + 1.2)

    fig, ax = plt.subplots(figsize=(9, altura))
    ax.barh(posicoes, contagem.values, color=cores, height=0.62)
    ax.set_yticks(posicoes, [str(c) for c in contagem.index])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if rotular:
        for pos, valor in zip(posicoes, contagem.values):
            ax.text(valor + contagem.max() * 0.012, pos,
                    f"{formatar_int(valor)}  ({formatar_pct(valor / total * 100)})",
                    va="center", fontsize=8.5, color=TEXTO_SEC)
    ax.set_xlim(0, contagem.max() * 1.24)
    eixo_compacto(ax, "x")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    mostrar(arquivo)


def distribuicao_categorica(df, coluna, ordem=None, max_classes=20, titulo=None):
    """Gráfico de barras + tabela (contagem e %) da distribuição das classes de uma coluna."""
    tabela = tabela_frequencia(df[coluna], ordem=ordem)
    titulo = titulo or f"{rotulo(coluna)}: distribuição das classes (n = {formatar_int(len(df))})"
    grafico_barras(tabela["Contagem"], titulo, max_classes=max_classes)
    display(tabela)
    return tabela


def distribuicao_numerica(df, coluna, bins=20, titulo=None, somente_positivos=False):
    """Histograma + resumo estatístico + tabela de contagem por faixa de uma coluna numérica.

    A tabela usa exatamente as barras do histograma: faixas [a, b), fechadas à esquerda
    e abertas à direita; apenas a última faixa inclui o limite superior."""
    serie = df[coluna]
    if somente_positivos:
        serie = serie[serie > 0]

    nome = rotulo(coluna) + (" (apenas valores > 0)" if somente_positivos else "")
    resumo = serie.describe().rename({
        "count": "n", "mean": "média", "std": "desvio padrão", "min": "mínimo",
        "25%": "Q1 (25%)", "50%": "mediana", "75%": "Q3 (75%)", "max": "máximo",
    }).to_frame(nome).T

    contagens, bordas = np.histogram(serie, bins=bins)
    faixas = [f"[{formatar_int(a)}, {formatar_int(b)})" for a, b in zip(bordas[:-1], bordas[1:])]
    faixas[-1] = faixas[-1][:-1] + "]"
    tabela = pd.DataFrame({
        "Contagem": contagens,
        "Percentual (%)": (contagens / contagens.sum() * 100).round(2),
    }, index=pd.Index(faixas, name="Faixa"))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(bordas[:-1], contagens, width=np.diff(bordas), align="edge",
           color=AZUL, edgecolor="white", linewidth=1.5)
    ax.grid(axis="x", visible=False)
    ax.set_title(titulo or f"{rotulo(coluna)}: distribuição (n = {formatar_int(len(serie))})")
    ax.set_xlabel(rotulo(coluna))
    ax.set_ylabel("Contagem")
    eixo_compacto(ax, "x")
    eixo_compacto(ax, "y")
    plt.tight_layout()
    plt.show()

    display(resumo)
    display(tabela)
    return tabela


# ---------------------------------------------------------------------------
# Análise bivariada: atributo × renda
# ---------------------------------------------------------------------------
def tabela_taxa(df, coluna, alvo="renda", positivo="> 50 mil", ordem=None):
    """Para cada classe de `coluna`: total, quantos são `positivo` e a taxa (%)."""
    cruzada = pd.crosstab(df[coluna], df[alvo])
    if positivo not in cruzada.columns:
        cruzada[positivo] = 0
    tabela = pd.DataFrame({
        "Total": cruzada.sum(axis=1),
        positivo: cruzada[positivo],
    })
    tabela["Taxa (%)"] = (tabela[positivo] / tabela["Total"] * 100).round(2)
    if ordem is not None:
        presentes = [c for c in ordem if c in tabela.index]
        restantes = [c for c in tabela.index if c not in presentes]
        tabela = tabela.reindex(presentes + restantes)
    else:
        tabela = tabela.sort_values("Taxa (%)", ascending=False)
    tabela.index.name = rotulo(coluna) if coluna in df.columns else coluna
    return tabela


def grafico_taxa(tabela, titulo, referencia=None, xlabel="Pessoas com renda > 50 mil (%)",
                 max_classes=25):
    """Barras horizontais da taxa de positivos por classe, com rótulo 'taxa (n total)'.

    `referencia` traça a taxa global como linha pontilhada cinza."""
    tabela = tabela.copy()
    cores = [AZUL] * len(tabela)
    if max_classes is not None and len(tabela) > max_classes:
        tabela = tabela.iloc[:max_classes]
        cores = cores[:max_classes]
    taxa = tabela["Taxa (%)"].values
    posicoes = np.arange(len(tabela))[::-1]
    altura = max(2.4, 0.42 * len(tabela) + 1.2)

    fig, ax = plt.subplots(figsize=(9, altura))
    ax.barh(posicoes, taxa, color=cores, height=0.62)
    ax.set_yticks(posicoes, [str(c) for c in tabela.index])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for pos, t, n in zip(posicoes, taxa, tabela["Total"].values):
        ax.text(t + 1.0, pos, f"{formatar_pct(t)}  (n = {formatar_int(n)})",
                va="center", fontsize=8.5, color=TEXTO_SEC)
    if referencia is not None:
        ax.axvline(referencia, color=TEXTO_SEC, linestyle=":", linewidth=1)
        ax.text(referencia, posicoes.max() + 0.75, f"média geral {formatar_pct(referencia)}",
                ha="center", va="bottom", fontsize=8, color=TEXTO_SEC)
    ax.set_xlim(0, max(100, taxa.max() * 1.3 if len(taxa) else 100) if taxa.max() > 75 else taxa.max() * 1.45 + 5)
    eixo_percentual(ax, "x")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def renda_por_categoria(df, coluna, ordem=None, alvo="renda", positivo="> 50 mil", titulo=None,
                        max_classes=25):
    """Gráfico da taxa de renda > 50 mil por classe + tabela (total, positivos, taxa)."""
    tabela = tabela_taxa(df, coluna, alvo=alvo, positivo=positivo, ordem=ordem)
    referencia = (df[alvo] == positivo).mean() * 100
    titulo = titulo or f"{rotulo(coluna)}: proporção de renda > 50 mil por classe"
    grafico_taxa(tabela, titulo, referencia=referencia, max_classes=max_classes)
    display(tabela)
    return tabela


def renda_por_faixa(df, coluna, bins, alvo="renda", positivo="> 50 mil", titulo=None):
    """Taxa de renda > 50 mil por faixa de uma variável numérica (faixas [a, b), última fechada).

    Mostra barras verticais na escala da variável + tabela (total, positivos, taxa)."""
    bordas = np.asarray(list(bins), dtype=float)
    serie = df[coluna]
    indices = np.clip(np.searchsorted(bordas, serie, side="right") - 1, 0, len(bordas) - 2)
    indices = pd.Series(indices, index=df.index)
    dentro = (serie >= bordas[0]) & (serie <= bordas[-1])
    faixas = [f"[{formatar_int(a)}, {formatar_int(b)})" for a, b in zip(bordas[:-1], bordas[1:])]
    faixas[-1] = faixas[-1][:-1] + "]"
    positivos = (df[alvo] == positivo)
    total = indices[dentro].value_counts().reindex(range(len(faixas)), fill_value=0)
    pos = indices[dentro & positivos].value_counts().reindex(range(len(faixas)), fill_value=0)
    tabela = pd.DataFrame({"Total": total.values, positivo: pos.values}, index=pd.Index(faixas, name="Faixa"))
    tabela["Taxa (%)"] = (tabela[positivo] / tabela["Total"].replace(0, np.nan) * 100).round(2)

    referencia = positivos.mean() * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(bordas[:-1], tabela["Taxa (%)"].fillna(0).values, width=np.diff(bordas), align="edge",
           color=AZUL, edgecolor="white", linewidth=1.5)
    ax.axhline(referencia, color=TEXTO_SEC, linestyle=":", linewidth=1)
    ax.text(bordas[-1], referencia, f" média geral {formatar_pct(referencia)}", va="center", ha="left",
            fontsize=8, color=TEXTO_SEC)
    ax.grid(axis="x", visible=False)
    ax.set_title(titulo or f"{rotulo(coluna)}: proporção de renda > 50 mil por faixa")
    ax.set_xlabel(rotulo(coluna))
    ax.set_ylabel("Renda > 50 mil (%)")
    eixo_compacto(ax, "x")
    eixo_percentual(ax, "y")
    plt.tight_layout()
    plt.show()
    display(tabela)
    return tabela


def media_por_grupo(df, coluna, grupo="renda", ordem=None, titulo=None):
    """Média, mediana e desvio padrão de `coluna` por classe de `grupo` + gráfico de barras das médias."""
    tabela = df.groupby(grupo, observed=True)[coluna].agg(["count", "mean", "median", "std"])
    tabela.columns = ["n", "Média", "Mediana", "Desvio padrão"]
    if ordem is not None:
        tabela = tabela.reindex([c for c in ordem if c in tabela.index])
    tabela = tabela.round(2)
    tabela.index.name = rotulo(grupo)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(tabela))
    ax.bar(x, tabela["Média"].values, color=AZUL, width=0.55)
    ax.set_xticks(x, [str(c) for c in tabela.index])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for xi, m in zip(x, tabela["Média"].values):
        ax.text(xi, m + tabela["Média"].max() * 0.015, formatar_num(m, 1), ha="center", va="bottom",
                fontsize=9, color=TEXTO_SEC)
    ax.set_ylim(0, tabela["Média"].max() * 1.18)
    ax.set_title(titulo or f"{rotulo(coluna)}: média por {rotulo(grupo).lower()}")
    ax.set_ylabel(f"{rotulo(coluna)} (média)")
    plt.tight_layout()
    plt.show()
    display(tabela)
    return tabela


# ---------------------------------------------------------------------------
# Resultados de modelos
# ---------------------------------------------------------------------------
def grafico_metricas(tabela, metrica, erro=None, titulo=None, destaque=None, xlabel=None,
                     xmin=None, referencia=None, rotulo_referencia=None, arquivo=None):
    """Barras horizontais de uma métrica (em %) por linha de `tabela` (índice = nome do modelo).

    `erro` é o nome da coluna com o desvio padrão (barra de erro). `destaque` é o índice da linha
    pintada em azul; as demais ficam em azul-claro. `referencia` traça uma linha vertical."""
    tabela = tabela.sort_values(metrica, ascending=True)
    valores = tabela[metrica].values
    posicoes = np.arange(len(tabela))
    cores = [AZUL if (destaque is None or i == destaque) else AZUL_CLARO for i in tabela.index]
    altura = max(2.4, 0.42 * len(tabela) + 1.2)

    fig, ax = plt.subplots(figsize=(9, altura))
    ax.barh(posicoes, valores, color=cores, height=0.62,
            xerr=tabela[erro].values if erro else None,
            error_kw={"ecolor": TEXTO_SEC, "capsize": 2, "elinewidth": 1})
    ax.set_yticks(posicoes, [str(c) for c in tabela.index])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    desloc = (tabela[erro].values if erro else np.zeros(len(tabela)))
    for pos, v, e in zip(posicoes, valores, desloc):
        ax.text(v + e + 0.15, pos, formatar_pct(v, 2), va="center", fontsize=8.5, color=TEXTO_SEC)
    if referencia is not None:
        ax.axvline(referencia, color=TEXTO_SEC, linestyle=":", linewidth=1)
        ax.text(referencia, posicoes.max() + 0.7, rotulo_referencia or formatar_pct(referencia, 2),
                ha="center", va="bottom", fontsize=8, color=TEXTO_SEC)
    xmin = xmin if xmin is not None else max(0, np.floor(valores.min() - 3))
    ax.set_xlim(xmin, valores.max() + (desloc.max() if len(desloc) else 0) + 3.5)
    eixo_percentual(ax, "x")
    ax.set_title(titulo or metrica)
    ax.set_xlabel(xlabel or metrica)
    plt.tight_layout()
    mostrar(arquivo)
