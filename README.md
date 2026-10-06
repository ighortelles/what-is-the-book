# What Is the Book?

Projeto acadêmico de Introdução à Visão Computacional para **detectar livros em
imagens**. A aplicação localiza cada livro e devolve sua caixa delimitadora
(*bounding box*), usando apenas técnicas clássicas de Visão Computacional — sem
Deep Learning.

## Metodologia

O pipeline usa o OpenCV para extrair características **HOG** (*Histogram of
Oriented Gradients*) de regiões candidatas. O trabalho mantém dois classificadores:
Regressão Logística e MLP, priorizando melhorias no treinamento e na detecção.
Cada modelo diferencia recortes que contêm livros de recortes de fundo. Durante a inferência,
o algoritmo normaliza a resolução, percorre janelas de diferentes tamanhos e aplica NMS (*Non-Maximum
Suppression*) para remover caixas sobrepostas e redundantes.

```text
imagem -> normalização espacial -> janelas -> HOG -> classificador em lotes -> NMS -> caixas originais
```

## Dataset

O projeto utiliza um dataset anotado em formato COCO, localizado em `data/`:

- `data/train/`: 156 imagens e 204 anotações; usado para treinar os classificadores.
- `data/valid/`: 45 imagens e 77 anotações; usado para ajustar limiares.
- `data/test/`: 22 imagens e 37 anotações; reservado para a avaliação final.

As caixas anotadas da categoria `Book` são lidas do arquivo
`_annotations.coco.json` de cada divisão. O diretório `data/` não está neste Github devido ao seu tamanho. Para acessar o dataset completo, consulte: https://universe.roboflow.com/hifsaiftikhar77-gmail-com/day-24-book-detection

## Como os modelos são treinados

Os dois modelos recebem a mesma matriz de características HOG e os mesmos
rótulos. Cada exemplo corresponde a um **recorte de imagem**, representado por
um vetor de **5.940 características**, com rótulo `1` para livro ou `0` para fundo.
Os classificadores aprendem a distinguir esses recortes; a localização dos livros
na imagem completa é feita pela busca com janelas durante a detecção.

### Construção dos exemplos

A função `training_features()`, em `src/detection/hog_detector.py`, utiliza somente
imagens da divisão `train`:

1. Normaliza o maior lado da imagem para 640 pixels, preservando a proporção, e
   transforma as caixas para essa escala. Caixas fora da imagem são limitadas à
   região visível; anotações sem área são ignoradas.
2. Recorta cada livro anotado e acrescenta até duas janelas de busca com IoU de
   pelo menos 0,60 com sua caixa. Esses exemplos recebem o rótulo `1`.
3. Sorteia até 15 janelas de fundo por imagem. A soma das áreas de interseção com
   livros, dividida pela área da janela, deve ser no máximo 1%. Isso rejeita uma
   janela dentro de um livro grande mesmo que sua IoU seja pequena. O rótulo é `0`.
4. Extrai HOG dos exemplos selecionados. Janelas ambíguas não são usadas como fundo.

Com os parâmetros padrão, há até 612 positivos e 2.340 negativos. A quantidade
real depende da disponibilidade de janelas válidas e da remoção de duplicatas.
A semente aleatória é fixada em `42` para tornar a amostragem reproduzível.
Na verificação com o dataset atual, foram gerados 2.740 recortes: 595 positivos
e 2.145 negativos, formando uma matriz de características de tamanho `(2740, 5940)`.

Os tamanhos das janelas são estimados somente a partir de `train`, agrupando as
larguras e alturas normalizadas com K-Means em escala logarítmica. São usados até
12 tamanhos-base, incluindo proporções verticais e horizontais. A configuração é
passada tanto à extração dos exemplos quanto ao treinamento do classificador.

### Pré-processamento e características HOG

Após a normalização espacial da imagem completa, cada recorte percorre:

```text
recorte -> escala de cinza -> CLAHE -> redimensionamento para 96 × 128 -> HOG
```

- **Escala de cinza:** representa intensidades e reduz a entrada para um canal.
  O HOG descreve mudanças de intensidade e não utiliza diretamente a cor da capa.
- **CLAHE:** realça o contraste local para tornar bordas e detalhes mais visíveis.
- **Redimensionamento:** ajusta cada recorte para 96 pixels de largura e 128 de
  altura, garantindo um vetor de características com tamanho fixo.
- **HOG:** descreve a distribuição das direções de gradiente, representando
  contornos e textura. Usa células de `8 × 8` pixels, blocos de `16 × 16`, passo de
  `8 × 8` e nove direções de gradiente. O vetor resultante possui 5.940 valores.

O treinamento é executado por `classifier.fit(features, labels)`. As entradas são:

```python
train_features.shape  # (quantidade_de_recortes, 5940)
train_labels.shape  # (quantidade_de_recortes,)
```

### Configuração de cada classificador

As configurações estão em `src/detection/classifiers.py`.

| Modelo | Entrada efetiva | Treinamento e configuração atual |
| --- | --- | --- |
| Regressão Logística | HOG padronizado | Ajusta pesos para separar livro e fundo. Usa balanceamento das classes e até 3.000 iterações. |
| MLP | HOG padronizado | Ajusta os pesos de duas camadas ocultas, com 128 e 64 neurônios. Usa até 1.000 épocas e parada antecipada. |

A padronização usa `StandardScaler`, que calcula a média e o desvio de cada
característica durante o treinamento e reutiliza esses valores na previsão.
O pickle preserva o scaler junto com o classificador. Todos os modelos mantidos
utilizam essa padronização.

O MLP reserva uma parte dos exemplos de `train` para controlar a parada antecipada.
Essa divisão interna é diferente da pasta `valid`, utilizada para avaliar os dois
modelos. Os limites de iterações e épocas não significam que todos os modelos
executem necessariamente o máximo configurado.

### Da classificação de recortes à detecção de livros

Para cada imagem nova, `detect()`:

1. Normaliza o maior lado para 640 pixels e utiliza os tamanhos-base salvos junto
   com o modelo, multiplicados por `0,8`, `1,0` e `1,25`. Inclui também uma janela
   da imagem inteira e posições junto às bordas.
2. Extrai HOG e faz previsões em lotes de até 256 janelas. O passo máximo padrão
   é de 32 pixels, reduzido para janelas pequenas, com mínimo de 16 pixels quando
   o usuário mantém o passo padrão.
3. Mantém as janelas cuja pontuação é maior ou igual ao limiar de detecção.
4. Aplica NMS vetorizado, com limiar de IoU de `0,35`, e converte as caixas de
   volta às coordenadas da imagem original.

A função `candidate_coverage()` mede a cobertura geométrica das janelas sem
classificar imagens. Na verificação do pipeline corrigido, 75 dos 77 livros de
`valid` possuem pelo menos uma janela com IoU >= 0,50 (97,4%). Isso é um limite
superior geométrico, não a revocação de um modelo treinado. Os resultados anteriores
dos classificadores estão registrados em [docs/baseline.md](docs/baseline.md).

A Regressão Logística usa `decision_function`. O MLP usa `predict_proba`,
com a pontuação calculada como `probabilidade_de_livro - 0,5`.
Portanto, o limiar zero tem interpretações diferentes, e as pontuações não são
diretamente comparáveis entre modelos. O limiar de cada modelo deve ser ajustado
usando o conjunto de validação.

### Limitações do treinamento atual

O dataset continua pequeno, e a amostragem depende de anotações completas: livros
não anotados podem aparecer em recortes tratados como fundo. Há uma rodada limitada
de hard negative mining em train, com até oito fundos de alta pontuação em 30
imagens escolhidas com semente 42. Janelas com ocupação anotada total maior que 1%
são excluídas, mesmo quando sua IoU é baixa. A seleção de limiares contempla os
dois classificadores, com grades próprias para suas escalas de pontuação.

Os recortes continuam sendo redimensionados para `96 × 128` ao extrair HOG, o que
deforma suas proporções. Os dois livros de validação não alcançados pelas janelas
e os erros dos classificadores devem ser analisados nos próximos experimentos.

## Estrutura

```text
src/
  app/
    main.py             # entrada da aplicação Streamlit
    interface.py        # upload único, comparação de todos os modelos e caixas
  detection/
    classifiers.py      # classificadores usados nos experimentos
    hog_detector.py     # extração de exemplos, varredura multi-escala e NMS
    hard_negatives.py   # mineração limitada de falsos alarmes em train
    windows.py          # configuração e tamanhos estimados nas anotações de train
  evaluation/
    detection.py        # IoU, TP, FP, FN, precisão, revocação e F1-score
    thresholds.py       # comparação de limiares sem repetir previsões
    run_logistic.py     # experimento reproduzível pela linha de comando
    run_precision.py    # regularização, negativos difíceis e MLP
    plot_thresholds.py  # gráfico exportável, gerado a partir da tabela CSV
  preprocess/
    hog.py              # grayscale, CLAHE, redimensionamento e HOG
    image.py            # normalização espacial, com fatores de escala
  utils/
    coco.py             # leitura das anotações COCO
    geometry.py         # operações geométricas, incluindo IoU
    model_io.py         # salvar e carregar modelos em pickle
    path.py             # caminhos centralizados do projeto, dataset e modelos
    presentation.py     # tabelas e figuras do notebook, exportação versionada
  main.ipynb            # ponto de entrada e demonstração do pipeline
```

## Instalação

O projeto usa Python 3.13 e [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --group dev
```

## Execução

Abra `src/main.ipynb` em um ambiente Jupyter que use o interpretador `.venv` do
projeto. O notebook funciona como roteiro da apresentação, nesta ordem:

1. Introdução: objetivo, fluxo e diferença entre classificar recortes e localizar livros.
2. Dados: contagens das divisões, 15 imagens anotadas de train e pré-processamento ilustrado.
3. Treinamento: entradas, referência, regularização, negativos difíceis e MLP, em células separadas.
4. Validação: grades, métricas completas, curvas de limiar e seleção com recall mínimo de 50%.
5. Comparação: gráficos, evolução histórica, todas as caixas de um exemplo de valid e limitações.

Por padrão, `RETRAIN_MODELS=False` e `REEVALUATE_VALIDATION=False`: as células
carregam os modelos, o manifesto mais recente `precision_run_*.json` e os CSVs,
sem repetir treinamento ou a validação completa. A demonstração final processa
somente uma imagem. `EXPORT_FIGURES=True` exporta PNGs para o artigo sem substituir
figuras anteriores. O notebook anterior foi preservado em `tmp/main-before-presentation-*`.

Para reproduzir a rodada completa e gerar novos artefatos versionados:

```powershell
uv run python -m src.evaluation.run_precision
```

No notebook, ativar `RETRAIN_MODELS` também força nova validação e salvamento.
Esses experimentos manuais geram pickles e CSVs separados; para atualizar o
manifesto usado automaticamente na apresentação, execute o comando acima.

## Experimento de limiares da regressão

Um limiar altera a decisão de aceitar uma caixa, não os pesos aprendidos. Por isso,
o modelo é treinado uma única vez. São comparados `0`, `0,5`, `1`, `1,5`, `2`, `3`,
`4`, `5`, `6`, `8` e `10` na escala de `decision_function`.

As pontuações e o NMS são calculados uma vez por imagem, no menor limiar. Como o
NMS guloso processa caixas por pontuação decrescente, filtrar as caixas mantidas
por um limiar maior equivale a repetir esse mesmo NMS após o filtro. Assim, HOG e
previsões não precisam ser recalculados para cada limiar.

A seleção padrão maximiza a precisão entre os limiares com revocação de pelo menos
50% em `valid`. Há desempate por F1 e menor limiar. Se nenhum resultado atende à
restrição, o experimento informa isso; não seleciona silenciosamente uma solução
que deixa de detectar quase todos os livros. `MINIMUM_RECALL` é configurável.

Para executar o mesmo experimento fora do notebook:

```powershell
uv run python -m src.evaluation.run_logistic
```

O pickle preserva o limiar escolhido em `score_threshold_`, usado como valor
inicial no app. A tabela completa é salva em `docs/results/`, vinculada ao nome da
versão do modelo. O MLP usa sua escala própria de pontuação; os limiares acima não
devem ser aplicados diretamente a ele.

Na rodada focada em precisão, comparamos a referência de 30 negativos com
regularização mais forte (`C=0,01`) e uma rodada de negativos difíceis, mantendo
essas duas alterações separadas. O MLP recebe os mesmos recortes aleatórios da
referência. A grade da regressão foi ampliada até 25; a do MLP vai de 0 a 0,499
(probabilidades mínimas de 50% a 99,9%). `evaluate_models_thresholds` compartilha
HOG por lote entre os modelos, sem armazenar todas as janelas em disco. O conjunto
test não participa. A variante de 60 negativos ficou fora desta rodada curta.

### Resultados já executados em valid

| Negativos por imagem | Limiar | TP | FP | FN | Precisão | Revocação | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Até 15, referência | 0 | 53 | 8.567 | 24 | 0,61% | 68,83% | 0,01219 |
| Até 15 | 10 | 43 | 1.239 | 34 | 3,35% | 55,84% | 0,06328 |
| Até 30 | 10 | 40 | 896 | 37 | 4,27% | 51,95% | 0,07897 |

O limiar 10 foi o melhor da grade sob a restrição de revocação mínima de 50%,
nos dois treinos. É o maior limiar testado; não é um ótimo global comprovado.
A variante com até 30 negativos reduziu FP em 89,5% frente à referência, mas
perdeu 13 TP. Sua precisão continua baixa; os resultados não são de teste final.

Os arquivos `logistic_regression_001.pkl` (até 15 negativos) e
`logistic_regression_002.pkl` (até 30) estão disponíveis localmente em `models/`.
A versão 002 era a melhor das duas referências históricas, com limiar 10.
Para a recomendação atual, consulte a comparação do notebook e o campo
`recommended_model` do manifesto mais recente em `docs/results/`.

As comparações completas estão em
[tabela com 15 negativos](docs/results/logistic_regression_001_thresholds.csv) e
[tabela com 30 negativos](docs/results/logistic_regression_002_thresholds.csv), com
gráficos PNG na mesma pasta. A seção de evolução histórica preserva a referência
sem confundi-la com os resultados da rodada nova. O experimento com 60 negativos
não foi executado.

Após esta mudança de pipeline, salve primeiro os modelos já treinados que deseja
preservar. Antes de um novo experimento, reinicie o kernel e execute as células
desde os imports para carregar os módulos atualizados. É necessário regenerar os
exemplos e retreinar; as métricas anteriores não descrevem o pipeline corrigido.

### Rodada focada em precisão — 06/10/2026

Resultados reais nas mesmas 45 imagens de valid, com limiares escolhidos para
maximizar precisão mantendo recall ≥ 50%:

| Experimento | Limiar | TP | FP | FN | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Regressão, referência de 30 negativos | 10 | 40 | 896 | 37 | 4,27% | 51,95% | 0,07897 |
| Regressão, C=0,01 | 6 | 39 | 780 | 38 | 4,76% | 50,65% | 0,08705 |
| Regressão, negativos difíceis | 4 | 39 | 2.843 | 38 | 1,35% | 50,65% | 0,02636 |
| MLP, mesmos recortes aleatórios | 0,495 | 43 | 948 | 34 | 4,34% | 55,84% | 0,08052 |

**Escolha da rodada:** `logistic_regression_004`, C=0,01 e limiar 6. Em relação à
referência anterior de 30 negativos, FP caiu 12,9% (896 → 780), com perda de um TP.
A precisão passou de 4,27% para 4,76%, ganho de 0,49 ponto percentual. Ela ainda
é baixa. O MLP recuperou mais livros, mas teve precisão menor. A rodada limitada
de negativos difíceis piorou o resultado e não foi escolhida.

O protocolo completo levou 842 segundos no ambiente local. A referência foi
carregada da versão 002, não retreinada; a versão 003 guarda a nova avaliação da
grade ampliada. As versões 004 e 005 são novos treinos, e `mlp.pkl` é o MLP desta
rodada. Nada foi sobrescrito. O manifesto associa cada versão aos seus CSVs e
registra os IDs e caixas usados na mineração:
[precision_run_001.json](docs/results/precision_run_001.json).

Ampliar a grade não tornou o limiar ideal globalmente conhecido. Por exemplo,
a regressão C=0,01 com limiar 10 teve precisão de 15,70%, mas recall de apenas
24,68%, por isso não foi elegível. A escolha e os resultados são de validação;
test continua reservado para a avaliação final.

## Salvamento e reutilização de modelos

A função `save_model()` preserva os arquivos existentes em `models/`. O primeiro
salvamento usa o nome solicitado; os seguintes recebem o primeiro sufixo numérico
livre, como `mlp_001.pkl` e `mlp_002.pkl`. O caminho efetivamente salvo é retornado
pela função e exibido na célula de salvamento do notebook.

```python
from src.utils.model_io import load_model, save_model

saved_path = save_model(mlp_model, "mlp")
print(saved_path)  # mlp.pkl ou uma nova versão, se o arquivo já existir

# Carrega exatamente a versão indicada, sem repetir o treinamento.
restored_model = load_model(saved_path.stem)
```

Os pickles mantêm o classificador e sua normalização. Cada novo salvamento cria um
arquivo, inclusive quando o mesmo modelo é salvo novamente. O modo apresentação
não cria novos pickles; apenas novos experimentos deliberados fazem salvamentos.
Novos modelos também armazenam `detector_config_`, com os tamanhos das janelas e
a resolução normalizada. Pickles antigos continuam carregáveis, mas usam uma
configuração padrão de busca; não representam o treinamento corrigido.
Modelos com limiar ajustado armazenam também `score_threshold_` e informações de
treinamento em `training_metadata_`.

## Aplicação Streamlit

Salve os modelos treinados usando a célula de persistência do notebook. A aplicação
compara **todos os arquivos `.pkl` da Regressão Logística e do MLP** disponíveis
em `models/` sobre uma única imagem enviada, sem repetir o treinamento. Não há
seletor de modelo: cada versão gera sua previsão independente, não uma votação.

Os títulos descrevem os experimentos em vez de mostrar apenas números:

| Arquivo | Rótulo na interface |
| --- | --- |
| `logistic_regression.pkl` | Baseline inicial, 672 recortes |
| `logistic_regression_001.pkl` | Protocolo corrigido, 15 negativos |
| `logistic_regression_002.pkl` | Referência, 30 negativos, C=1 |
| `logistic_regression_003.pkl` | Referência reavaliada, mesmos pesos |
| `logistic_regression_004.pkl` | Regularização forte, C=0,01 |
| `logistic_regression_005.pkl` | 30 negativos + 172 negativos difíceis |
| `mlp.pkl` | HOG, camadas 128/64, 30 negativos |

O nome técnico do arquivo permanece visível para rastreabilidade. Novas versões
não listadas acima são descritas pelos metadados, quando disponíveis. As versões
002 e 003 devem produzir as mesmas caixas com os mesmos limiares e passo, pois
guardam os mesmos pesos. A 003 corresponde à reavaliação da grade ampliada.

Na raiz do projeto, execute:

```powershell
uv sync
uv run streamlit run src/app/main.py
```

No navegador, envie uma imagem e clique em **Comparar todos os modelos**. A tela
mostra um resumo e as imagens anotadas de cada modelo, em duas colunas, com todas
as caixas e um painel expansível de coordenadas/pontuações. Cada modelo começa
com seu próprio limiar salvo; versões antigas sem esse atributo usam zero.
O painel **Limiares individuais**, na barra lateral, permite ajustar cada valor
separadamente. O MLP usa três casas decimais. O passo da busca é comum aos modelos.

Modelos com as mesmas janelas compartilham HOG por lote para reduzir o custo; o
baseline antigo, sem configuração salva, usa outro grupo com as janelas padrão.
O tempo exibido é o total da inferência compartilhada, não um tempo individual.
Falhas de carregamento ou previsão são indicadas por modelo e não equivalem a
uma previsão negativa. Os modelos que funcionam continuam gerando resultados.

As previsões permanecem na sessão ao expandir os detalhes, sem rodar novamente.
Alterar a imagem, os arquivos, os limiares ou o passo invalida o resultado anterior
e exige clicar novamente no botão. Um resultado sem caixas não garante ausência
de livros. Sem anotações reais da imagem enviada, o app não mede precisão, recall,
TP ou FP; essas métricas pertencem à avaliação do dataset.

Após atualizar módulos Python do projeto, reinicie o servidor com `Ctrl+C` e o
comando acima para evitar imports antigos em memória. Apenas atualizar a página
pode não renovar módulos importados.

## Métricas de avaliação

Uma detecção é considerada correta quando sua IoU com uma caixa real é maior ou
igual a 0,50. O notebook reporta:

- Verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN).
- Precisão: `TP / (TP + FP)`.
- Revocação: `TP / (TP + FN)`.
- F1-score: média harmônica entre precisão e revocação.

O limiar `SCORE_THRESHOLD` deve ser escolhido somente com `valid`. Após essa
decisão, o conjunto `test` é usado uma única vez para o resultado final.

## Histórico dos experimentos

As alterações implementadas, seus motivos e as verificações estão registradas em
[docs/historico-alteracoes.md](docs/historico-alteracoes.md). Os resultados do
protocolo inicial foram preservados em [docs/baseline.md](docs/baseline.md).

## Qualidade de código

O projeto utiliza Ruff para lint e formatação:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```

Para verificar a apresentação e salvar as saídas no próprio notebook, sem
retreinar (mantenha as opções iniciais em `False`):

```powershell
$env:PYTHONUTF8 = "1"
uv run jupyter execute src/main.ipynb --inplace --timeout=180
```

O `nbclient` é uma dependência de desenvolvimento usada nessa verificação.
