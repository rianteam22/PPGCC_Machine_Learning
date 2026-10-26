# Classificação de renda no Adult Census Income

Atividade da disciplina de Aprendizado de Máquina (PPGCC). O objetivo é prever, a partir de 14 atributos sociodemográficos do censo norte-americano de 1994, se a renda anual de uma pessoa excede US$ 50 mil, **maximizando a acurácia** de uma árvore de decisão (`DecisionTreeClassifier`, único classificador da atividade). Os experimentos variam o pré-processamento, o espaço de hiperparâmetros e a estratégia de busca. O relatório está em [`relatorio/relatorio.md`](relatorio/relatorio.md), [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf) e [`relatorio/relatorio.docx`](relatorio/relatorio.docx).

## Estrutura

| Arquivo | Conteúdo |
| --- | --- |
| `adult.csv` | Dataset (Kaggle `uciml/adult-census-income`, 32.561 linhas × 15 colunas) |
| `traducao.py` | Tradução dos nomes de colunas e dos valores categóricos para o português |
| `graficos.py` | Funções de gráfico e tabela compartilhadas pelos notebooks |
| `01_exploracao_inicial.ipynb` | Distribuição de cada atributo, valores ausentes, duplicatas |
| `02_analise_bivariada.ipynb` | Cada atributo cruzado com a renda (proporção de `> 50 mil` por classe/faixa) |
| `03_treinamento_arvore.ipynb` | Árvore de decisão: ablação, balanceamento, busca em grade, avaliação no teste, regras |
| `04_experimentos.ipynb` | Consolidação dos três experimentos, seleção da árvore final e avaliação no teste (McNemar, matriz de confusão, importância por permutação) |
| `experimentos/` | `comum.py` (protocolo: dados, partição, CV, métricas, árvore), `otimizacao.py` (espaço de busca e Optuna) e um script por experimento |
| `resultados/` | Saídas dos experimentos e da árvore final (`*.json`, `*.csv`), versionadas |
| `relatorio/` | Relatório (`relatorio.md`, `relatorio.pdf` gerado por `gerar_pdf.py`, `relatorio.docx` gerado por `gerar_docx.js`), `auditoria.py` (auditoria numérica), `figura_arvore_final.py` (desenho da árvore final), figuras e tabelas |

## Ambiente

```bash
uv venv .venv && uv pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_experimentos.ipynb
.venv/bin/python relatorio/auditoria.py auditar relatorio/relatorio.md
.venv/bin/python relatorio/gerar_pdf.py   # relatorio.md -> relatorio.pdf (requer google-chrome)
(cd relatorio && npm install && node gerar_docx.js)   # relatorio.md -> relatorio.docx (biblioteca docx, Node.js)
.venv/bin/python relatorio/figura_arvore_final.py   # desenha a árvore final (figuras/fig_arvore_final.{pdf,png} e _topo.png)
```

## Protocolo de avaliação

Todos os experimentos usam o mesmo protocolo do notebook 03: remoção das 24 duplicatas e dos registros com valor ausente (30.139 registros), partição estratificada 70/30 com seed 42 (21.097 / 9.042 registros) e validação cruzada estratificada de 5 folds só no treino. A seleção usa a acurácia; acurácia balanceada, precision, recall, F1 (classe `> 50 mil`) e ROC-AUC são registradas lado a lado. Na variante que acrescenta ao treino os 2.398 registros incompletos (`comum.conjunto_estendido`), o conjunto de teste é o mesmo, os folds dos registros completos são os do protocolo e só os registros completos de cada fold de validação são pontuados, de modo que as acurácias de validação são comparáveis entre todas as configurações.

## Experimentos em worktrees

Cada experimento vive em um ramo `exp/<nome>`, criado como worktree em `.worktrees/<nome>` (pasta ignorada pelo git), e grava seus resultados em `resultados/`. Os ramos são integrados em `main` ao final.

```bash
git worktree add .worktrees/arvore exp/arvore
cd .worktrees/arvore && N_JOBS=8 ../../.venv/bin/python experimentos/exp_arvore.py
```

| Ramo | Script | Pergunta |
| --- | --- | --- |
| `exp/arvore-preprocessamento` | `exp_preprocessamento.py` | Peso amostral, escolaridade, codificação, país agrupado, capital líquido e registros incompletos mudam o resultado de duas árvores fixas? |
| `exp/arvore` | `exp_arvore.py` | Curva de validação, busca em grade e busca aleatória encontram uma árvore melhor que a inicial, e a que custo? |
| `exp/arvore-bayesiana` | `exp_arvore_bayesiana.py <tpe\|aleatoria> <seed>` e `juntar` | A otimização bayesiana (Optuna, TPE) supera o amostrador aleatório com o mesmo orçamento (300 tentativas, 3 seeds)? |

`RAPIDO=1` executa cada script com buscas reduzidas para teste. Após a integração em `main`, as worktrees foram removidas; os ramos `exp/*` permanecem e qualquer um pode ser reaberto com `git worktree add .worktrees/<nome> exp/<nome>`.

O ramo `relatorio-completo` guarda uma versão estendida do trabalho, com dois experimentos adicionais (codificação ordinal combinada com os registros incompletos sob busca otimizada, e estabilidade em seis partições), seu relatório e os scripts correspondentes. Os ramos `exp/arvore-codificacao` e `exp/arvore-estabilidade` contêm esses experimentos, e os ramos `exp/preprocessamento`, `exp/ensembles`, `exp/baselines`, `exp/estabilidade`, `exp/boosting-extra` e `exp/bayesiana` guardam uma tentativa anterior com outros classificadores, fora do escopo da atividade. Nenhum deles é usado pelo `main`.

## Auditoria numérica do relatório

Nenhum número é digitado no relatório sem rastro. O script `relatorio/auditoria.py` segue três passos:

1. **Fonte única de verdade:** `adult.csv` e os arquivos de `resultados/` (`*.json`, `*.csv`), incluindo `modelo_final.json`, gravado pelo notebook 04.
2. **Registro de fatos:** o script deriva, com nome e fórmula, todo valor que o texto pode citar (contagens, acurácias, desvios, diferenças, proporções, tamanhos de árvore), em `python relatorio/auditoria.py registro`. As diferenças entre métricas são calculadas sobre os valores exatos, antes do arredondamento, e as tentativas do Optuna são numeradas a partir de 0, como nos arquivos de resultados.
3. **Auditor:** varre o Markdown, extrai cada número e exige que ele seja o arredondamento de algum fato do registro, na precisão exibida. Números estruturais (Tabela 3, Seção 4.1, anos das citações, seeds, espaços de busca do Apêndice) são reconhecidos e ignorados. Qualquer número sem rastro reprova o texto, e o rastro completo (número → fatos compatíveis) é gravado em `relatorio/relatorio_auditoria.txt` para conferência humana, em especial dos inteiros, que podem casar com mais de um fato.

O auditor cobre os números. As afirmações qualitativas (ordem entre candidatos, contagens em palavras, interpretações) passaram por uma revisão independente que conferiu cada uma contra os arquivos de resultados.

## Padrão de escrita do relatório

O relatório segue padrão acadêmico, impessoal, em primeira pessoa do plural combinada com voz passiva, como pedido na atividade. O texto é autossuficiente para um leitor sem acesso ao repositório: a configuração de partida é apresentada como "árvore inicial", cada hiperparâmetro é explicado em palavras na primeira menção, e não há referências a notebooks, arquivos ou histórico do projeto. Os termos "precision", "recall" e "seed" ficam em inglês. Sobre esse padrão foram aplicados os princípios estruturais da skill [ASD-STE100](https://github.com/danyuchn/asd-ste100-skill): frases de até 35 palavras, um tópico por parágrafo, um termo por conceito (por exemplo, sempre "conjunto de teste", "árvore inicial", "árvore final", "registros incompletos"), verbos no lugar de nominalizações, sem adjetivos de marketing, listas para enumerações de três ou mais itens, nenhum ponto e vírgula na prosa, afirmações limitadas ao que os dados mostram e nenhum fato acrescentado às fontes. A única regra da skill não aplicada é a voz ativa obrigatória, porque conflita com a exigência de voz passiva da atividade. O texto foi verificado por contagens automáticas (frases, pontos e vírgulas, citações de tabelas, figuras e referências), pela auditoria numérica e por uma revisão independente.
