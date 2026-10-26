"""Experimento 2 — estratégias clássicas de busca de hiperparâmetros da árvore de decisão.

Compara, com o mesmo protocolo (validação cruzada de 5 folds no treino, seleção pela acurácia,
demais métricas registradas), três estratégias:
1. curva de validação de max_depth (1 a 30), com os demais hiperparâmetros padrão;
2. busca em grade exaustiva (GridSearchCV, 648 combinações de cinco hiperparâmetros);
3. busca aleatória (RandomizedSearchCV, 300 candidatos) em um espaço mais amplo, que acrescenta
   ccp_alpha, max_features e min_impurity_decrease.

A árvore inicial (profundidade 7, 10 registros por folha, 20 para dividir) é reavaliada como
referência. O melhor candidato da grade pelo F1 também é registrado, para a discussão sobre o
balanceamento de classes.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum
from scipy.stats import loguniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier

N_JOBS = int(os.environ.get("N_JOBS", "8"))
RAPIDO = os.environ.get("RAPIDO") == "1"


def registrar(nome, busca, X_te, y_te, duracao, n_candidatos):
    pipe = busca.best_estimator_
    teste = comum.avaliar_teste(pipe, X_te, y_te)
    res = busca.cv_results_
    i = busca.best_index_
    cv = {m: float(res[f"mean_test_{m}"][i]) for m in comum.METRICAS}
    cv.update({f"{m}_std": float(res[f"std_test_{m}"][i]) for m in comum.METRICAS})
    melhores = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
    comum.log(f"{nome}: CV acc {cv['acuracia']*100:.2f} ± {cv['acuracia_std']*100:.2f} | teste acc "
              f"{teste['acuracia']*100:.2f} f1 {teste['f1']*100:.2f} auc {teste['roc_auc']*100:.2f} | "
              f"{n_candidatos} candidatos, {duracao:.0f} s | {melhores}")
    return {
        "estrategia": nome, "melhores_parametros": melhores, "n_candidatos": int(n_candidatos),
        "n_ajustes": int(n_candidatos * comum.FOLDS), "tempo_s": float(duracao),
        **comum.tamanho_arvore(pipe), "cv": cv, "teste": teste,
    }


def main():
    comum.log("Experimento 2 — estratégias clássicas de busca")
    df = comum.carregar_dados()
    X, y = comum.separar_xy(df)
    X_tr, X_te, y_tr, y_te = comum.dividir(X, y)
    cv = comum.validacao_cruzada()
    base = comum.montar(DecisionTreeClassifier(random_state=comum.SEED), X_tr)
    saida = {"experimento": "arvore", "descricao": __doc__.strip(), "estrategias": []}

    # 0. Árvore inicial, reavaliada sob o protocolo
    inicial = comum.construir_arvore(comum.PARAMETROS_ARVORE_ANTERIOR, X_tr)
    cv_ini = comum.avaliar_cv(inicial, X_tr, y_tr, n_jobs=N_JOBS)
    inicial.fit(X_tr, y_tr)
    saida["arvore_anterior"] = {"parametros": comum.PARAMETROS_ARVORE_ANTERIOR, "cv": cv_ini,
                                "teste": comum.avaliar_teste(inicial, X_te, y_te), **comum.tamanho_arvore(inicial)}
    comum.log(f"árvore inicial: CV acc {cv_ini['acuracia']*100:.2f} | teste acc {saida['arvore_anterior']['teste']['acuracia']*100:.2f}")

    # 1. Curva de validação de max_depth (acurácia e F1)
    profundidades = np.arange(1, 31)
    curva = {"max_depth": profundidades.tolist()}
    for metrica in ["accuracy", "f1"]:
        tr, va = validation_curve(base, X_tr, y_tr, param_name="modelo__max_depth",
                                  param_range=profundidades, cv=cv, scoring=metrica, n_jobs=N_JOBS)
        curva[f"{metrica}_treino"] = tr.mean(axis=1).tolist()
        curva[f"{metrica}_validacao"] = va.mean(axis=1).tolist()
        curva[f"{metrica}_validacao_std"] = va.std(axis=1).tolist()
    saida["curva_validacao"] = curva
    comum.log(f"curva de validação: melhor profundidade por acurácia = "
              f"{int(profundidades[np.argmax(curva['accuracy_validacao'])])} "
              f"({max(curva['accuracy_validacao'])*100:.2f}%), por F1 = {int(profundidades[np.argmax(curva['f1_validacao'])])}")

    # 2. Busca em grade exaustiva
    grade = {
        "modelo__criterion": ["gini", "entropy"],
        "modelo__max_depth": [4, 6, 8, 10, 12, 14, 16, 20, None],
        "modelo__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "modelo__min_samples_split": [2, 10, 50],
        "modelo__class_weight": [None, "balanced"],
    }
    if RAPIDO:
        grade = {k: v[:2] for k, v in grade.items()}
    n_grade = int(np.prod([len(v) for v in grade.values()]))
    t0 = time.perf_counter()
    gs = GridSearchCV(base, grade, cv=cv, scoring=comum.METRICAS, refit="acuracia", n_jobs=N_JOBS)
    gs.fit(X_tr, y_tr)
    saida["estrategias"].append(registrar("grade", gs, X_te, y_te, time.perf_counter() - t0, n_grade))
    comum.salvar_tabela("arvore_grade_top", comum.resumo_busca(gs, 20))
    tabela_grade = pd.DataFrame(gs.cv_results_)
    tabela_grade = tabela_grade[[c for c in tabela_grade.columns if c.startswith("param_") or c in
                                 ("mean_test_acuracia", "std_test_acuracia", "mean_test_f1", "mean_test_roc_auc",
                                  "mean_fit_time")]]
    comum.salvar_tabela("arvore_grade_completa", tabela_grade)

    # 3. Busca aleatória
    distribuicoes = {
        "modelo__criterion": ["gini", "entropy"],
        "modelo__max_depth": randint(3, 31),
        "modelo__min_samples_leaf": randint(1, 101),
        "modelo__min_samples_split": randint(2, 201),
        "modelo__max_features": [None, "sqrt", 0.5, 0.8],
        "modelo__ccp_alpha": loguniform(1e-6, 1e-2),
        "modelo__min_impurity_decrease": loguniform(1e-7, 1e-3),
        "modelo__class_weight": [None, "balanced"],
    }
    n_iter = 300 if not RAPIDO else 8
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(base, distribuicoes, n_iter=n_iter, cv=cv, scoring=comum.METRICAS,
                            refit="acuracia", n_jobs=N_JOBS, random_state=comum.SEED)
    rs.fit(X_tr, y_tr)
    saida["estrategias"].append(registrar("aleatoria", rs, X_te, y_te, time.perf_counter() - t0, n_iter))
    comum.salvar_tabela("arvore_aleatoria_top", comum.resumo_busca(rs, 20))

    # 4. Melhor candidato da grade pelo F1 (para a discussão sobre o balanceamento de classes)
    res = pd.DataFrame(gs.cv_results_)
    i_f1 = int(res["mean_test_f1"].idxmax())
    saida["melhor_por_f1_na_grade"] = {
        "parametros": {k.replace("modelo__", ""): v for k, v in res.loc[i_f1, "params"].items()},
        "cv_acuracia": float(res.loc[i_f1, "mean_test_acuracia"]), "cv_f1": float(res.loc[i_f1, "mean_test_f1"]),
    }

    comum.salvar_resultado("arvore", saida)
    comum.log("Concluído: resultados/arvore.json")


if __name__ == "__main__":
    main()
