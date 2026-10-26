"""Experimento 3 — otimização bayesiana × busca aleatória para a árvore de decisão.

Compara dois amostradores do Optuna sobre o mesmo espaço de hiperparâmetros da árvore
(`otimizacao.espaco`): o TPE (otimização bayesiana) e o amostrador aleatório. Cada execução
tem 300 tentativas, e cada tentativa avalia um candidato pela validação cruzada do protocolo
(5 folds no treino, seleção pela acurácia). Cada amostrador é executado com três sementes
(0, 1 e 2), para que a comparação não dependa de uma única execução. O melhor candidato de cada
execução é reajustado no treino completo e avaliado no conjunto de teste.

Uso: python exp_arvore_bayesiana.py <tpe|aleatoria> <semente>   ou   python exp_arvore_bayesiana.py juntar
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum
import otimizacao

SEMENTES = [0, 1, 2]
AMOSTRADORES = ["tpe", "aleatoria"]
N_JOBS = int(os.environ.get("N_JOBS", "5"))
RAPIDO = os.environ.get("RAPIDO") == "1"


def executar(nome_amostrador, semente):
    comum.log(f"Experimento 3 — {nome_amostrador}, semente {semente}")
    df = comum.carregar_dados()
    X, y = comum.separar_xy(df)
    X_tr, X_te, y_tr, y_te = comum.dividir(X, y)
    folds = comum.folds_protocolo(y_tr.values)
    res = otimizacao.executar(nome_amostrador, semente, X_tr, y_tr, folds, n_trials=8 if RAPIDO else otimizacao.N_TRIALS,
                              n_jobs=N_JOBS, rotulo=f"{nome_amostrador}/{semente}")
    res = otimizacao.avaliar_melhor(res, X_tr, y_tr, X_te, y_te)
    comum.log(f"{nome_amostrador}/{semente}: CV acc {res['cv']['acuracia']*100:.2f} ± {res['cv']['acuracia_std']*100:.2f} | "
              f"teste acc {res['teste']['acuracia']*100:.2f} f1 {res['teste']['f1']*100:.2f} | "
              f"{res['tempo_s']:.0f} s | tentativa {res['melhor_tentativa']} | {res['melhores_parametros']}")
    comum.salvar_resultado(f"arvore_bayesiana_{nome_amostrador}_{semente}", res)


def juntar():
    execucoes = []
    for a in AMOSTRADORES:
        for s in SEMENTES:
            r = comum.carregar_resultado(f"arvore_bayesiana_{a}_{s}")
            execucoes.append({k: v for k, v in r.items() if k != "historico"})
    resumo = {}
    for a in AMOSTRADORES:
        cvs = np.array([e["cv"]["acuracia"] for e in execucoes if e["amostrador"] == a])
        testes = np.array([e["teste"]["acuracia"] for e in execucoes if e["amostrador"] == a])
        tempos = np.array([e["tempo_s"] for e in execucoes if e["amostrador"] == a])
        resumo[a] = {"cv_media": float(cvs.mean()), "cv_std": float(cvs.std()), "cv_min": float(cvs.min()),
                     "cv_max": float(cvs.max()), "teste_media": float(testes.mean()), "teste_std": float(testes.std()),
                     "tempo_medio_s": float(tempos.mean()), "n_execucoes": int(len(cvs))}
    # melhor acurácia de validação até a tentativa t, média entre sementes, por amostrador
    curvas = {a: np.mean([e["melhor_ate_a_tentativa"] for e in execucoes if e["amostrador"] == a], axis=0).tolist()
              for a in AMOSTRADORES}
    comum.salvar_resultado("arvore_bayesiana", {
        "experimento": "arvore_bayesiana", "descricao": __doc__.strip(), "espaco": otimizacao.__doc__.strip(),
        "n_tentativas": otimizacao.N_TRIALS, "sementes": SEMENTES, "execucoes": execucoes, "resumo": resumo,
        "curva_media": curvas,
    })
    for a in AMOSTRADORES:
        r = resumo[a]
        comum.log(f"{a}: CV {r['cv_media']*100:.2f} ± {r['cv_std']*100:.2f} (min {r['cv_min']*100:.2f}, máx {r['cv_max']*100:.2f}) | "
                  f"teste {r['teste_media']*100:.2f} ± {r['teste_std']*100:.2f} | {r['tempo_medio_s']:.0f} s")
    comum.log("Concluído: resultados/arvore_bayesiana.json")


if __name__ == "__main__":
    if sys.argv[1] == "juntar":
        juntar()
    else:
        executar(sys.argv[1], int(sys.argv[2]))
