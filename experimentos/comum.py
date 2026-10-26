"""Protocolo comum dos experimentos com a árvore de decisão (dados, partição, validação cruzada,
métricas e persistência dos resultados).

O protocolo reproduz o do notebook 03 (treinamento da árvore de decisão):
- remoção das 24 linhas duplicadas;
- remoção das linhas com valor ausente ("?") -> 30.139 linhas;
- alvo binário: 1 = renda "> 50 mil", 0 = "≤ 50 mil";
- partição estratificada 70 % treino / 30 % teste, random_state = 42;
- validação cruzada estratificada com 5 folds (shuffle, random_state = 42) só no treino.

O único classificador dos experimentos é o DecisionTreeClassifier. As métricas são as mesmas já
aplicadas no notebook 03: acurácia, acurácia balanceada, precisão, revocação e F1 da classe
positiva (> 50 mil) e ROC-AUC.

Além da validação cruzada padrão do scikit-learn (`avaliar_cv`), o módulo oferece a validação
cruzada com folds explícitos e máscara de pontuação (`avaliar_cv_folds`). Ela é necessária para a
variante que acrescenta ao treino as linhas com valor ausente: os folds das linhas completas são
os mesmos do protocolo e só as linhas completas de cada fold de validação são pontuadas, de modo
que a acurácia de validação fica comparável entre todas as variantes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from traducao import traduzir_dataframe, ordem_classes  # noqa: E402,F401

SEED = 42
TESTE = 0.30
FOLDS = 5
CAMINHO_DADOS = RAIZ / "adult.csv"
DIRETORIO_RESULTADOS = RAIZ / "resultados"

POSITIVO = "> 50 mil"
AUSENTE = "Não informado"
COLUNAS_COM_AUSENTE = ["classe_trabalho", "ocupacao", "pais_origem"]
NUMERICAS = ["idade", "escolaridade_num", "ganho_capital", "perda_capital", "horas_semana"]
CATEGORICAS = ["classe_trabalho", "estado_civil", "ocupacao", "relacao_familiar", "raca", "sexo",
               "pais_origem"]

# Árvore do notebook 03 original (referência de todos os experimentos)
PARAMETROS_ARVORE_ANTERIOR = {"max_depth": 7, "min_samples_leaf": 10, "min_samples_split": 20}

METRICAS = {
    "acuracia": "accuracy",
    "acuracia_bal": "balanced_accuracy",
    "precisao": "precision",
    "revocacao": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_dados(ausentes: str = "remover", remover_duplicatas: bool = True) -> pd.DataFrame:
    """Lê adult.csv traduzido.

    ausentes = "remover"  -> descarta linhas com "?" (protocolo do notebook 03);
    ausentes = "categoria" -> mantém as linhas e o valor "Não informado" vira uma categoria.
    """
    bruto = pd.read_csv(CAMINHO_DADOS)
    if ausentes == "remover":
        df = traduzir_dataframe(bruto, ausente=np.nan)
    elif ausentes == "categoria":
        df = traduzir_dataframe(bruto)
    else:
        raise ValueError("ausentes deve ser 'remover' ou 'categoria'")
    if remover_duplicatas:
        df = df.drop_duplicates()
    if ausentes == "remover":
        df = df.dropna()
    return df.reset_index(drop=True)


def separar_xy(df: pd.DataFrame, peso_amostral: bool = False, escolaridade: str = "num"):
    """Separa atributos e alvo (1 = > 50 mil).

    escolaridade = "num" (só escolaridade_num), "texto" (só a categórica) ou "ambas".
    """
    y = (df["renda"] == POSITIVO).astype(int).rename("renda")
    remover = ["renda"]
    if not peso_amostral:
        remover.append("peso_amostral")
    if escolaridade == "num":
        remover.append("escolaridade")
    elif escolaridade == "texto":
        remover.append("escolaridade_num")
    elif escolaridade != "ambas":
        raise ValueError("escolaridade deve ser 'num', 'texto' ou 'ambas'")
    X = df.drop(columns=remover)
    return X, y


def dividir(X, y, seed: int = SEED):
    """Partição estratificada 70/30 com a mesma semente do notebook 03."""
    return train_test_split(X, y, test_size=TESTE, stratify=y, random_state=seed)


def validacao_cruzada(seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)


def folds_protocolo(y, seed: int = SEED) -> list:
    """Folds (posições) da validação cruzada do protocolo para um vetor y."""
    return [(tr, va) for tr, va in validacao_cruzada(seed).split(np.zeros(len(y)), y)]


def colunas_por_tipo(X: pd.DataFrame):
    categoricas = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str", "category")]
    numericas = [c for c in X.columns if c not in categoricas]
    return numericas, categoricas


def conjunto_estendido(seed: int = SEED):
    """Treino acrescido das linhas com valor ausente ("Não informado" como categoria) e o MESMO
    conjunto de teste do protocolo (que só contém linhas completas).

    Devolve X_tr, y_tr, X_te, y_te, mascara_completa (True nas linhas completas do treino, que
    são as primeiras e estão na mesma ordem do treino do protocolo) e os folds estendidos.
    As linhas incompletas são distribuídas entre os folds por estratificação e entram apenas
    como treino (elas nunca são pontuadas)."""
    df = carregar_dados("remover")
    X, y = separar_xy(df)
    X_tr, X_te, y_tr, y_te = dividir(X, y, seed)
    df_cat = carregar_dados("categoria")
    completas = ~(df_cat[COLUNAS_COM_AUSENTE] == AUSENTE).any(axis=1)
    pos_completas = np.flatnonzero(completas.values)
    assert len(pos_completas) == len(df), (len(pos_completas), len(df))
    pos_treino = pos_completas[X_tr.index.values]
    pos_teste = pos_completas[X_te.index.values]
    pos_incompletas = np.flatnonzero(~completas.values)
    Xc, yc = separar_xy(df_cat)
    X_tr_ext = pd.concat([Xc.iloc[pos_treino], Xc.iloc[pos_incompletas]]).reset_index(drop=True)
    y_tr_ext = pd.concat([yc.iloc[pos_treino], yc.iloc[pos_incompletas]]).reset_index(drop=True)
    X_te_ext = Xc.iloc[pos_teste].reset_index(drop=True)
    y_te_ext = yc.iloc[pos_teste].reset_index(drop=True)
    assert np.array_equal(y_te_ext.values, y_te.values)
    assert np.array_equal(X_te_ext["idade"].values, X_te["idade"].values)
    assert np.array_equal(y_tr_ext.values[:len(pos_treino)], y_tr.values)
    mascara = np.r_[np.ones(len(pos_treino), dtype=bool), np.zeros(len(pos_incompletas), dtype=bool)]
    # folds das linhas completas = folds do protocolo (dependem só de n e de y); as incompletas
    # são repartidas por estratificação com a mesma semente e deslocadas para o fim
    folds_c = folds_protocolo(y_tr.values, seed)
    folds_i = folds_protocolo(y_tr_ext.values[len(pos_treino):], seed)
    desloc = len(pos_treino)
    folds = [(np.r_[tc, ti + desloc], np.r_[vc, vi + desloc]) for (tc, vc), (ti, vi) in zip(folds_c, folds_i)]
    return X_tr_ext, y_tr_ext, X_te_ext, y_te_ext, mascara, folds


# ---------------------------------------------------------------------------
# Pré-processamento e modelo (dentro do pipeline, ajustado só no treino)
# ---------------------------------------------------------------------------
def preprocessador(X: pd.DataFrame, codificacao: str = "onehot") -> ColumnTransformer:
    """Codifica as categóricas por one-hot ou ordinal; as numéricas passam sem transformação
    (a árvore de decisão não depende da escala)."""
    numericas, categoricas = colunas_por_tipo(X)
    if codificacao == "onehot":
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif codificacao == "ordinal":
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError("codificacao deve ser 'onehot' ou 'ordinal'")
    ct = ColumnTransformer([("num", "passthrough", numericas), ("cat", cat, categoricas)],
                           verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def montar(modelo, X: pd.DataFrame, codificacao: str = "onehot") -> Pipeline:
    return Pipeline([("prep", preprocessador(X, codificacao=codificacao)), ("modelo", modelo)])


def limpar_parametros(params: dict) -> dict:
    """Converte valores lidos de JSON de volta aos tipos esperados pelo DecisionTreeClassifier."""
    saida = {}
    for k, v in params.items():
        if v in ("None", None):
            saida[k] = None
        elif isinstance(v, float) and k in ("max_depth", "min_samples_leaf", "min_samples_split", "max_leaf_nodes") \
                and float(v).is_integer():
            saida[k] = int(v)
        else:
            saida[k] = v
    return saida


def construir_arvore(params: dict, X: pd.DataFrame, codificacao: str = "onehot") -> Pipeline:
    """Pipeline (codificação + DecisionTreeClassifier) a partir de hiperparâmetros serializados."""
    return montar(DecisionTreeClassifier(random_state=SEED, **limpar_parametros(params)), X, codificacao)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def avaliar_cv(pipeline, X, y, cv=None, n_jobs: int = -1) -> dict:
    """Média e desvio padrão de cada métrica na validação cruzada + tempo médio de ajuste."""
    cv = cv or validacao_cruzada()
    inicio = time.perf_counter()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=METRICAS, n_jobs=n_jobs,
                            return_train_score=True)
    duracao = time.perf_counter() - inicio
    resumo = {}
    for m in METRICAS:
        resumo[m] = float(scores[f"test_{m}"].mean())
        resumo[f"{m}_std"] = float(scores[f"test_{m}"].std())
        resumo[f"{m}_treino"] = float(scores[f"train_{m}"].mean())
    resumo["tempo_ajuste_s"] = float(scores["fit_time"].mean())
    resumo["tempo_total_s"] = float(duracao)
    return resumo


def _metricas(y_true, y_pred, y_score) -> dict:
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "acuracia_bal": float(balanced_accuracy_score(y_true, y_pred)),
        "precisao": float(precision_score(y_true, y_pred, zero_division=0)),
        "revocacao": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def _ajustar_fold(pipeline, X, y, tr, va, mascara):
    pipe = clone(pipeline)
    t0 = time.perf_counter()
    pipe.fit(X.iloc[tr], y.iloc[tr])
    tempo = time.perf_counter() - t0
    va = va[mascara[va]] if mascara is not None else va
    Xv, yv = X.iloc[va], y.iloc[va]
    res = _metricas(yv, pipe.predict(Xv), pipe.predict_proba(Xv)[:, 1])
    tr_p = tr[mascara[tr]] if mascara is not None else tr
    res["acuracia_treino"] = float(accuracy_score(y.iloc[tr_p], pipe.predict(X.iloc[tr_p])))
    res["tempo_ajuste_s"] = tempo
    return res


def avaliar_cv_folds(pipeline, X, y, folds, mascara=None, n_jobs: int = 5) -> dict:
    """Validação cruzada com folds explícitos (posições) e máscara de pontuação.

    Só as linhas de validação com mascara True são pontuadas (mascara None = todas). Devolve o
    mesmo dicionário de `avaliar_cv` (sem as métricas de treino além da acurácia)."""
    inicio = time.perf_counter()
    por_fold = Parallel(n_jobs=n_jobs)(delayed(_ajustar_fold)(pipeline, X, y, tr, va, mascara) for tr, va in folds)
    resumo = {}
    for m in METRICAS:
        valores = np.array([r[m] for r in por_fold])
        resumo[m] = float(valores.mean())
        resumo[f"{m}_std"] = float(valores.std())
    resumo["acuracia_treino"] = float(np.mean([r["acuracia_treino"] for r in por_fold]))
    resumo["tempo_ajuste_s"] = float(np.mean([r["tempo_ajuste_s"] for r in por_fold]))
    resumo["tempo_total_s"] = float(time.perf_counter() - inicio)
    return resumo


def avaliar_teste(pipeline, X_test, y_test) -> dict:
    """Métricas no conjunto de teste para um pipeline já ajustado."""
    return _metricas(y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1])


def tamanho_arvore(pipeline) -> dict:
    arvore = pipeline.named_steps["modelo"]
    return {"profundidade": int(arvore.get_depth()), "n_folhas": int(arvore.get_n_leaves()),
            "n_nos": int(arvore.tree_.node_count)}


def resumo_busca(busca, n: int = 15) -> pd.DataFrame:
    """Tabela com os n melhores candidatos de uma busca (GridSearchCV, RandomizedSearchCV, ...)."""
    res = pd.DataFrame(busca.cv_results_)
    colunas = [c for c in res.columns if c.startswith("param_")]
    chave = f"mean_test_{busca.refit}" if isinstance(busca.refit, str) else "mean_test_score"
    colunas += [c for c in res.columns if c.startswith("mean_test_") or c.startswith("std_test_")]
    colunas += ["mean_fit_time"]
    return res.sort_values(chave, ascending=False)[colunas].head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistência dos resultados
# ---------------------------------------------------------------------------
def _serializavel(obj):
    if isinstance(obj, dict):
        return {str(k): _serializavel(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializavel(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def salvar_resultado(nome: str, dados: dict) -> Path:
    DIRETORIO_RESULTADOS.mkdir(exist_ok=True)
    caminho = DIRETORIO_RESULTADOS / f"{nome}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(_serializavel(dados), f, ensure_ascii=False, indent=2)
    return caminho


def salvar_tabela(nome: str, tabela: pd.DataFrame) -> Path:
    DIRETORIO_RESULTADOS.mkdir(exist_ok=True)
    caminho = DIRETORIO_RESULTADOS / f"{nome}.csv"
    tabela.to_csv(caminho, index=False)
    return caminho


def carregar_resultado(nome: str) -> dict:
    with open(DIRETORIO_RESULTADOS / f"{nome}.json", encoding="utf-8") as f:
        return json.load(f)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
