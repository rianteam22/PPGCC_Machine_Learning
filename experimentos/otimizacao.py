"""Otimização de hiperparâmetros da árvore de decisão com o Optuna.

O espaço de busca (`espaco`) é compartilhado pelos experimentos 3 (otimização bayesiana × busca
aleatória) e 4 (codificação × linhas com valor ausente), de modo que os resultados dos dois são
comparáveis. Cada tentativa avalia um candidato pela validação cruzada do protocolo
(`comum.avaliar_cv_folds`), com seleção pela acurácia.

Hiperparâmetros do DecisionTreeClassifier cobertos:
- criterion (gini, entropy), splitter (best, random);
- max_depth (3 a 30), min_samples_leaf (1 a 100), min_samples_split (2 a 200);
- max_features (todos, sqrt, log2, 50 %, 80 %);
- ccp_alpha (1e-6 a 1e-2, escala log), min_impurity_decrease (1e-7 a 1e-3, escala log);
- class_weight (nenhum, balanced);
- max_leaf_nodes (sem limite, ou 8 a 1024 em escala log).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum

N_TRIALS = 300
optuna.logging.set_verbosity(optuna.logging.WARNING)


def espaco(trial) -> dict:
    p = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 200, log=True),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.8]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 1e-7, 1e-3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    if trial.suggest_categorical("limitar_folhas", [False, True]):
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 8, 1024, log=True)
    return p


def amostrador(nome: str, semente: int):
    if nome == "tpe":
        return optuna.samplers.TPESampler(seed=semente)
    if nome == "aleatoria":
        return optuna.samplers.RandomSampler(seed=semente)
    raise ValueError("amostrador deve ser 'tpe' ou 'aleatoria'")


def executar(nome_amostrador: str, semente: int, X_tr, y_tr, folds, mascara=None, codificacao: str = "onehot",
             n_trials: int = N_TRIALS, n_jobs: int = 5, rotulo: str = "") -> dict:
    """Roda um estudo do Optuna e devolve o melhor candidato com o histórico das tentativas."""
    historico = []

    def objetivo(trial):
        params = espaco(trial)
        pipe = comum.construir_arvore(params, X_tr, codificacao)
        cv = comum.avaliar_cv_folds(pipe, X_tr, y_tr, folds, mascara, n_jobs=n_jobs)
        historico.append({"tentativa": trial.number, "parametros": params, **cv})
        return cv["acuracia"]

    def registrar(study, trial):
        if (trial.number + 1) % 25 == 0:
            comum.log(f"{rotulo} tentativa {trial.number + 1}/{n_trials}: melhor CV acc "
                      f"{study.best_value * 100:.3f} (tentativa {study.best_trial.number})")

    study = optuna.create_study(direction="maximize", sampler=amostrador(nome_amostrador, semente))
    t0 = time.perf_counter()
    study.optimize(objetivo, n_trials=n_trials, callbacks=[registrar])
    tempo = time.perf_counter() - t0
    valores = np.array([h["acuracia"] for h in historico])
    melhor = historico[study.best_trial.number]
    cv = {k: v for k, v in melhor.items() if k not in ("tentativa", "parametros")}
    return {
        "amostrador": nome_amostrador, "semente": int(semente), "codificacao": codificacao,
        "n_candidatos": int(n_trials), "n_ajustes": int(n_trials * comum.FOLDS), "tempo_s": float(tempo),
        "melhor_tentativa": int(study.best_trial.number), "melhores_parametros": melhor["parametros"],
        "cv": cv, "melhor_ate_a_tentativa": np.maximum.accumulate(valores).tolist(),
        "historico": historico,
    }


def avaliar_melhor(resultado: dict, X_tr, y_tr, X_te, y_te) -> dict:
    """Reajusta o melhor candidato no treino completo e acrescenta as métricas de teste e o
    tamanho da árvore ao resultado."""
    pipe = comum.construir_arvore(resultado["melhores_parametros"], X_tr, resultado["codificacao"])
    pipe.fit(X_tr, y_tr)
    resultado["teste"] = comum.avaliar_teste(pipe, X_te, y_te)
    resultado["tamanho"] = comum.tamanho_arvore(pipe)
    resultado["n_atributos"] = int(pipe[:-1].transform(X_tr.head(2)).shape[1])
    return resultado
