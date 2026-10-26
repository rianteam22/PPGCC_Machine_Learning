"""Experimento 1 — pré-processamento (árvore de decisão).

Mede o efeito de decisões de preparação dos dados sobre duas árvores de decisão fixas: a árvore
anterior (notebook 03: profundidade 7, 10 registros por folha, 20 para dividir) e uma árvore de
profundidade 10 (10 registros por folha). As variantes são:
- a referência do protocolo;
- manter peso_amostral;
- escolaridade textual no lugar da numérica, ou as duas;
- codificação ordinal das categóricas no lugar de one-hot;
- agrupar pais_origem em "Estados Unidos" / "Outro";
- substituir ganho_capital e perda_capital por capital_liquido (ganho − perda);
- aproveitar no treino as 2.398 linhas com valor ausente ("Não informado" como categoria),
  mantendo o MESMO conjunto de teste do protocolo (que só contém linhas completas).

Todas as variantes usam os mesmos folds de validação cruzada no treino e o mesmo conjunto de teste.
Na última variante, só as linhas completas de cada fold de validação são pontuadas, de modo que a
acurácia de validação é comparável às demais. A acurácia pontuada em todas as linhas de validação
(como no experimento anterior) também é registrada, para documentar a diferença.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum

N_JOBS = int(os.environ.get("N_JOBS", "5"))

ARVORES = {
    "Árvore anterior (prof. 7)": comum.PARAMETROS_ARVORE_ANTERIOR,
    "Árvore (prof. 10)": {"max_depth": 10, "min_samples_leaf": 10},
}


def agrupar_pais(df):
    df = df.copy()
    df["pais_origem"] = np.where(df["pais_origem"] == "Estados Unidos", "Estados Unidos", "Outro")
    return df


def capital_liquido(df):
    df = df.copy()
    df["capital_liquido"] = df["ganho_capital"] - df["perda_capital"]
    return df.drop(columns=["ganho_capital", "perda_capital"])


def avaliar(nome, X_tr, y_tr, X_te, y_te, folds, mascara=None, codificacao="onehot", descricao=""):
    linhas = []
    for nome_arvore, params in ARVORES.items():
        pipe = comum.construir_arvore(params, X_tr, codificacao)
        cv = comum.avaliar_cv_folds(pipe, X_tr, y_tr, folds, mascara, n_jobs=N_JOBS)
        linha = {"configuracao": nome, "descricao": descricao, "modelo": nome_arvore,
                 "n_treino": len(X_tr), "n_treino_pontuado": int(mascara.sum()) if mascara is not None else len(X_tr),
                 "n_teste": len(X_te)}
        if mascara is not None:
            cv_todas = comum.avaliar_cv_folds(pipe, X_tr, y_tr, folds, None, n_jobs=N_JOBS)
            linha["cv_acuracia_todas_as_linhas"] = cv_todas["acuracia"]
        pipe.fit(X_tr, y_tr)
        teste = comum.avaliar_teste(pipe, X_te, y_te)
        linha["n_atributos"] = int(pipe[:-1].transform(X_tr.head(2)).shape[1])
        linha.update({f"cv_{k}": v for k, v in cv.items()})
        linha.update({f"teste_{k}": v for k, v in teste.items()})
        linha.update(comum.tamanho_arvore(pipe))
        linhas.append(linha)
        comum.log(f"{nome:<26} {nome_arvore:<26} CV acc {cv['acuracia']*100:.2f} ± {cv['acuracia_std']*100:.2f} | "
                  f"teste acc {teste['acuracia']*100:.2f}  f1 {teste['f1']*100:.2f}  auc {teste['roc_auc']*100:.2f}")
    return linhas


def main():
    comum.log("Experimento 1 — pré-processamento (árvore de decisão)")
    df = comum.carregar_dados(ausentes="remover")
    resultados = []

    X, y = comum.separar_xy(df, peso_amostral=False, escolaridade="num")
    X_tr, X_te, y_tr, y_te = comum.dividir(X, y)
    idx_tr, idx_te = X_tr.index, X_te.index
    folds = comum.folds_protocolo(y_tr.values)

    def particao(X):
        return X.loc[idx_tr], X.loc[idx_te]

    resultados += avaliar("referencia", X_tr, y_tr, X_te, y_te, folds,
                          descricao="sem peso_amostral, escolaridade_num, one-hot, só linhas completas")

    Xp, _ = comum.separar_xy(df, peso_amostral=True, escolaridade="num")
    a, b = particao(Xp)
    resultados += avaliar("com_peso_amostral", a, y_tr, b, y_te, folds, descricao="mantém peso_amostral (fnlwgt)")

    for esc, desc in [("texto", "só a escolaridade textual (16 categorias)"),
                      ("ambas", "escolaridade textual + escolaridade_num")]:
        Xe, _ = comum.separar_xy(df, peso_amostral=False, escolaridade=esc)
        a, b = particao(Xe)
        resultados += avaliar(f"escolaridade_{esc}", a, y_tr, b, y_te, folds, descricao=desc)

    resultados += avaliar("codificacao_ordinal", X_tr, y_tr, X_te, y_te, folds, codificacao="ordinal",
                          descricao="categóricas como códigos inteiros (OrdinalEncoder)")

    Xg = agrupar_pais(X)
    a, b = particao(Xg)
    resultados += avaliar("pais_agrupado", a, y_tr, b, y_te, folds, descricao="pais_origem = Estados Unidos / Outro")

    Xl = capital_liquido(X)
    a, b = particao(Xl)
    resultados += avaliar("capital_liquido", a, y_tr, b, y_te, folds,
                          descricao="capital_liquido = ganho_capital − perda_capital no lugar das duas colunas")

    Xc_tr, yc_tr, Xc_te, yc_te, mascara, folds_ext = comum.conjunto_estendido()
    resultados += avaliar("ausentes_como_categoria", Xc_tr, yc_tr, Xc_te, yc_te, folds_ext, mascara,
                          descricao=f"treino com +{len(Xc_tr) - len(X_tr)} linhas com 'Não informado'; teste idêntico; "
                                    "validação pontuada só nas linhas completas")

    tabela = pd.DataFrame(resultados)
    comum.salvar_tabela("preprocessamento", tabela)
    comum.salvar_resultado("preprocessamento", {
        "experimento": "preprocessamento", "descricao": __doc__.strip(),
        "arvores": ARVORES, "n_treino_referencia": int(len(X_tr)), "n_teste": int(len(X_te)),
        "resultados": resultados,
    })
    comum.log("Concluído: resultados/preprocessamento.json e .csv")


if __name__ == "__main__":
    main()
