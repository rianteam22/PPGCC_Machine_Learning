"""
Tradução dos termos do dataset *Adult Census Income* (UCI / Kaggle) para português.

O dataset foi extraído do Censo dos EUA de 1994 (Current Population Survey).
As traduções abaixo levam em conta esse contexto: por exemplo, ``relationship``
é a relação da pessoa com o responsável pelo domicílio (não um "relacionamento"),
``education.num`` é um ranking ordinal da escolaridade (não "anos de estudo") e
``fnlwgt`` é o peso amostral do censo.

Uso básico
----------
>>> import pandas as pd
>>> from traducao import traduzir, traduzir_dataframe, dicionario_dados
>>> traduzir("workclass")                            # 'classe_trabalho'
>>> traduzir("Self-emp-inc", coluna="workclass")     # 'Autônomo (incorporado)'
>>> traduzir("Prof-specialty")                       # 'Profissional especializado'
>>> df = traduzir_dataframe(pd.read_csv("adult.csv"))

``traduzir`` não depende de pandas; apenas ``traduzir_dataframe`` e
``dicionario_dados`` precisam dele.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Marcador de valor ausente
# ---------------------------------------------------------------------------
AUSENTE = "?"                 # como aparece no CSV original
AUSENTE_PT = "Não informado"  # tradução padrão do marcador

# ---------------------------------------------------------------------------
# Colunas: nome original -> identificador em português (sem acento, snake_case,
# seguro para usar em código) e rótulo legível (para títulos e gráficos).
# ---------------------------------------------------------------------------
COLUNAS: dict[str, str] = {
    "age": "idade",
    "workclass": "classe_trabalho",
    "fnlwgt": "peso_amostral",
    "education": "escolaridade",
    "education.num": "escolaridade_num",
    "marital.status": "estado_civil",
    "occupation": "ocupacao",
    "relationship": "relacao_familiar",
    "race": "raca",
    "sex": "sexo",
    "capital.gain": "ganho_capital",
    "capital.loss": "perda_capital",
    "hours.per.week": "horas_semana",
    "native.country": "pais_origem",
    "income": "renda",
}

ROTULOS: dict[str, str] = {
    "age": "Idade",
    "workclass": "Classe de trabalho",
    "fnlwgt": "Peso amostral",
    "education": "Escolaridade",
    "education.num": "Escolaridade (nível numérico)",
    "marital.status": "Estado civil",
    "occupation": "Ocupação",
    "relationship": "Relação familiar",
    "race": "Raça",
    "sex": "Sexo",
    "capital.gain": "Ganho de capital",
    "capital.loss": "Perda de capital",
    "hours.per.week": "Horas por semana",
    "native.country": "País de origem",
    "income": "Renda",
}

TIPOS: dict[str, str] = {
    "age": "numérica",
    "workclass": "categórica",
    "fnlwgt": "numérica",
    "education": "categórica (ordinal)",
    "education.num": "numérica (ordinal)",
    "marital.status": "categórica",
    "occupation": "categórica",
    "relationship": "categórica",
    "race": "categórica",
    "sex": "categórica (binária)",
    "capital.gain": "numérica",
    "capital.loss": "numérica",
    "hours.per.week": "numérica",
    "native.country": "categórica",
    "income": "alvo (binária)",
}

DESCRICOES: dict[str, str] = {
    "age": "Idade da pessoa, em anos.",
    "workclass": "Tipo de vínculo/empregador: setor privado, governo (federal, estadual, local), autônomo etc.",
    "fnlwgt": "Peso amostral (final weight) atribuído pelo Census Bureau: quantas pessoas da população a linha representa. Não descreve a pessoa.",
    "education": "Maior nível de escolaridade concluído.",
    "education.num": "Codificação ordinal da escolaridade (1 = pré-escola ... 16 = doutorado). Não representa anos de estudo.",
    "marital.status": "Estado civil.",
    "occupation": "Grupo ocupacional (categorias amplas do censo).",
    "relationship": "Relação da pessoa com o responsável pelo domicílio (marido, esposa, filho(a) etc.).",
    "race": "Raça declarada (categorias do censo dos EUA).",
    "sex": "Sexo.",
    "capital.gain": "Ganhos de capital no ano, em dólares.",
    "capital.loss": "Perdas de capital no ano, em dólares.",
    "hours.per.week": "Horas trabalhadas por semana.",
    "native.country": "País de origem (nascimento).",
    "income": "Variável alvo: renda anual acima de US$ 50 mil ou até US$ 50 mil.",
}

# ---------------------------------------------------------------------------
# Valores categóricos: coluna original -> {valor original: tradução}
# ---------------------------------------------------------------------------
VALORES: dict[str, dict[str, str]] = {
    "workclass": {
        "Private": "Setor privado",
        "Self-emp-not-inc": "Autônomo (não incorporado)",
        "Self-emp-inc": "Autônomo (incorporado)",
        "Federal-gov": "Governo federal",
        "State-gov": "Governo estadual",
        "Local-gov": "Governo local",
        "Without-pay": "Sem remuneração",
        "Never-worked": "Nunca trabalhou",
    },
    "education": {
        "Preschool": "Pré-escola",
        "1st-4th": "1ª a 4ª série",
        "5th-6th": "5ª a 6ª série",
        "7th-8th": "7ª a 8ª série",
        "9th": "9ª série",
        "10th": "10ª série",
        "11th": "11ª série",
        "12th": "12ª série",
        "HS-grad": "Ensino médio completo",
        "Some-college": "Superior incompleto",
        "Assoc-voc": "Grau associado (vocacional)",
        "Assoc-acdm": "Grau associado (acadêmico)",
        "Bachelors": "Bacharelado",
        "Masters": "Mestrado",
        "Prof-school": "Escola profissional",
        "Doctorate": "Doutorado",
    },
    "marital.status": {
        "Married-civ-spouse": "Casado(a) – cônjuge civil",
        "Married-AF-spouse": "Casado(a) – cônjuge nas Forças Armadas",
        "Married-spouse-absent": "Casado(a) – cônjuge ausente",
        "Never-married": "Solteiro(a)",
        "Divorced": "Divorciado(a)",
        "Separated": "Separado(a)",
        "Widowed": "Viúvo(a)",
    },
    "occupation": {
        "Tech-support": "Suporte técnico",
        "Craft-repair": "Ofícios e reparos",
        "Other-service": "Outros serviços",
        "Sales": "Vendas",
        "Exec-managerial": "Executivo/gerencial",
        "Prof-specialty": "Profissional especializado",
        "Handlers-cleaners": "Carga e limpeza",
        "Machine-op-inspct": "Operação de máquinas/inspeção",
        "Adm-clerical": "Administrativo",
        "Farming-fishing": "Agricultura e pesca",
        "Transport-moving": "Transporte e movimentação",
        "Priv-house-serv": "Serviço doméstico",
        "Protective-serv": "Serviços de proteção",
        "Armed-Forces": "Forças Armadas",
    },
    "relationship": {
        "Husband": "Marido",
        "Wife": "Esposa",
        "Own-child": "Filho(a)",
        "Other-relative": "Outro parente",
        "Not-in-family": "Sem vínculo familiar",
        "Unmarried": "Não casado(a)",
    },
    "race": {
        "White": "Branca",
        "Black": "Negra",
        "Asian-Pac-Islander": "Asiática/Ilhas do Pacífico",
        "Amer-Indian-Eskimo": "Indígena americana/Esquimó",
        "Other": "Outra",
    },
    "sex": {
        "Male": "Masculino",
        "Female": "Feminino",
    },
    "native.country": {
        "United-States": "Estados Unidos",
        "Mexico": "México",
        "Philippines": "Filipinas",
        "Germany": "Alemanha",
        "Canada": "Canadá",
        "Puerto-Rico": "Porto Rico",
        "El-Salvador": "El Salvador",
        "India": "Índia",
        "Cuba": "Cuba",
        "England": "Inglaterra",
        "Jamaica": "Jamaica",
        "South": "Sul (não especificado)",   # provavelmente Coreia do Sul
        "China": "China",
        "Italy": "Itália",
        "Dominican-Republic": "República Dominicana",
        "Vietnam": "Vietnã",
        "Guatemala": "Guatemala",
        "Japan": "Japão",
        "Poland": "Polônia",
        "Columbia": "Colômbia",             # grafia incorreta no original
        "Taiwan": "Taiwan",
        "Haiti": "Haiti",
        "Iran": "Irã",
        "Portugal": "Portugal",
        "Nicaragua": "Nicarágua",
        "Peru": "Peru",
        "Greece": "Grécia",
        "France": "França",
        "Ecuador": "Equador",
        "Ireland": "Irlanda",
        "Hong": "Hong Kong",                # truncado no original
        "Trinadad&Tobago": "Trinidad e Tobago",
        "Cambodia": "Camboja",
        "Laos": "Laos",
        "Thailand": "Tailândia",
        "Yugoslavia": "Iugoslávia",
        "Outlying-US(Guam-USVI-etc)": "Territórios dos EUA (Guam, Ilhas Virgens etc.)",
        "Honduras": "Honduras",
        "Hungary": "Hungria",
        "Scotland": "Escócia",
        "Holand-Netherlands": "Holanda",
    },
    "income": {
        "<=50K": "≤ 50 mil",
        ">50K": "> 50 mil",
    },
}

# Ordem natural das classes em colunas ordinais (nomes originais).
ORDEM_VALORES: dict[str, list[str]] = {
    "education": [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
        "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
        "Prof-school", "Doctorate",
    ],
    "income": ["<=50K", ">50K"],
}

# ---------------------------------------------------------------------------
# Índices inversos (montados uma vez)
# ---------------------------------------------------------------------------
_COLUNAS_INV: dict[str, str] = {pt: en for en, pt in COLUNAS.items()}
_VALORES_INV: dict[str, dict[str, str]] = {
    col: {pt: en for en, pt in mapa.items()} for col, mapa in VALORES.items()
}


def _normalizar(termo: Any) -> str:
    """Remove espaços nas pontas e o ponto final usado no arquivo de teste do UCI ('<=50K.')."""
    return str(termo).strip().rstrip(".")


def coluna_original(nome: str) -> str | None:
    """Devolve o nome original (inglês) da coluna, aceitando o nome em inglês ou em português."""
    nome = _normalizar(nome)
    if nome in COLUNAS:
        return nome
    # aceita também a variante com hífen do UCI (education-num, marital-status...)
    com_ponto = nome.replace("-", ".")
    if com_ponto in COLUNAS:
        return com_ponto
    return _COLUNAS_INV.get(nome)


def traduzir_coluna(nome: str) -> str:
    """Traduz o nome de uma coluna. Nomes desconhecidos voltam inalterados."""
    original = coluna_original(nome)
    return COLUNAS[original] if original else nome


def rotulo(nome: str) -> str:
    """Rótulo legível da coluna (ex.: 'marital.status' ou 'estado_civil' -> 'Estado civil')."""
    original = coluna_original(nome)
    return ROTULOS[original] if original else str(nome)


def traduzir_valor(coluna: str, valor: Any, ausente: str | None = AUSENTE_PT) -> Any:
    """
    Traduz um valor categórico da coluna indicada.

    - ``coluna`` pode estar em inglês ou português.
    - Valores numéricos ou desconhecidos voltam inalterados.
    - O marcador ``?`` vira ``ausente`` (padrão: 'Não informado'); use ``ausente=None`` para mantê-lo.
    """
    if not isinstance(valor, str):
        return valor
    texto = _normalizar(valor)
    if texto == AUSENTE:
        return ausente if ausente is not None else valor
    original = coluna_original(coluna)
    if original is None or original not in VALORES:
        return valor
    return VALORES[original].get(texto, valor)


def traduzir(termo: Any, coluna: str | None = None) -> Any:
    """
    Traduz um termo do dataset (nome de coluna ou valor categórico).

    - Com ``coluna``: procura o termo entre os valores daquela coluna.
    - Sem ``coluna``: procura primeiro entre os nomes de coluna, depois entre os
      valores de todas as colunas.
    - Termos desconhecidos voltam inalterados.
    """
    if coluna is not None:
        return traduzir_valor(coluna, termo)
    if not isinstance(termo, str):
        return termo
    texto = _normalizar(termo)
    if texto == AUSENTE:
        return AUSENTE_PT
    original = coluna_original(texto)
    if original is not None:
        return COLUNAS[original]
    for mapa in VALORES.values():
        if texto in mapa:
            return mapa[texto]
    return termo


def valor_original(coluna: str, valor_pt: str) -> str | None:
    """Caminho inverso: tradução em português -> valor original em inglês."""
    original = coluna_original(coluna)
    if original is None:
        return None
    if valor_pt == AUSENTE_PT:
        return AUSENTE
    return _VALORES_INV.get(original, {}).get(valor_pt)


def ordem_classes(coluna: str, traduzido: bool = True) -> list[str] | None:
    """Ordem natural das classes de uma coluna ordinal (``education``, ``income``), ou None."""
    original = coluna_original(coluna)
    if original is None or original not in ORDEM_VALORES:
        return None
    ordem = ORDEM_VALORES[original]
    return [VALORES[original][v] for v in ordem] if traduzido else list(ordem)


def traduzir_dataframe(
    df,
    colunas: bool = True,
    valores: bool = True,
    ausente: str | None = AUSENTE_PT,
    copiar: bool = True,
):
    """
    Traduz um DataFrame do Adult (nomes de colunas e/ou valores categóricos).

    Parâmetros
    ----------
    colunas : renomeia as colunas para os identificadores em português.
    valores : traduz os valores das colunas categóricas (espaços nas pontas são removidos).
    ausente : com o que substituir o marcador '?'. Padrão 'Não informado';
              ``None`` mantém o '?'; ``float('nan')`` converte para NaN.
    copiar  : trabalha em uma cópia (padrão) ou altera o DataFrame recebido.
    """
    import pandas as pd  # importado aqui para que o módulo funcione sem pandas

    if copiar:
        df = df.copy()

    if valores:
        for col in df.columns:
            original = coluna_original(col)
            if original is None or original not in VALORES:
                continue
            if not pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
                continue
            serie = df[col].astype("string").str.strip().str.rstrip(".")
            mapa: dict[str, Any] = dict(VALORES[original])
            if ausente is not None:
                mapa[AUSENTE] = ausente
            else:
                mapa[AUSENTE] = AUSENTE
            df[col] = serie.map(lambda v: mapa.get(v, v) if isinstance(v, str) else v)
            if ausente is not None and isinstance(ausente, float):  # NaN pedido
                df[col] = df[col].replace({ausente: pd.NA})

    if colunas:
        df = df.rename(columns={c: traduzir_coluna(c) for c in df.columns})

    return df


def dicionario_dados():
    """Tabela (DataFrame) com coluna original, identificador em português, rótulo, tipo e descrição."""
    import pandas as pd

    linhas = [
        {
            "Coluna original": en,
            "Coluna (pt)": COLUNAS[en],
            "Rótulo": ROTULOS[en],
            "Tipo": TIPOS[en],
            "Nº de classes": len(VALORES[en]) if en in VALORES else None,
            "Descrição": DESCRICOES[en],
        }
        for en in COLUNAS
    ]
    tabela = pd.DataFrame(linhas)
    tabela["Nº de classes"] = tabela["Nº de classes"].astype("Int64")
    return tabela


__all__ = [
    "AUSENTE", "AUSENTE_PT", "COLUNAS", "ROTULOS", "TIPOS", "DESCRICOES", "VALORES",
    "ORDEM_VALORES", "coluna_original", "traduzir_coluna", "rotulo", "traduzir_valor",
    "traduzir", "valor_original", "ordem_classes", "traduzir_dataframe", "dicionario_dados",
]
