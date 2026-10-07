# Histórico de alterações e decisões técnicas

Este documento registra o que mudou, por que mudou e o que foi verificado.
Novas alterações no pipeline devem ser acrescentadas em entradas datadas,
preservando os registros anteriores. As métricas do protocolo inicial estão em
[baseline.md](baseline.md).

## 06/10/2026 — Correção da amostragem e da busca de livros

### Problemas encontrados

O detector inicial usava janelas verticais de largura fixa, aplicadas em cinco
escalas da imagem. A maior janela correspondia a aproximadamente `192 × 384`
pixels na imagem original. Havia fotos de `3072 × 4096` pixels com livros maiores
que `2000 × 3000` pixels. Mesmo com posicionamento ideal, apenas 35 dos 77 livros
de `valid` poderiam alcançar a IoU mínima de 0,50.

Os negativos eram aceitos por terem IoU menor que 0,10 com as anotações. Uma janela
pequena inteiramente dentro de um livro grande pode satisfazer esse critério e ser
rotulada incorretamente como fundo. Os positivos, por sua vez, eram recortados
exatamente das caixas reais, enquanto as janelas da aplicação incluíam margens ou
apenas partes do objeto.

Na inspeção de `valid`, as dimensões dos arquivos COCO correspondiam às imagens
reais. Foi encontrada uma caixa que ultrapassava os limites da imagem. Isso não
justificou descartar o dataset; os problemas mais evidentes estavam no pipeline.

### Alterações implementadas

| Alteração | Motivo | Implementação |
| --- | --- | --- |
| Normalização do maior lado para 640 pixels, preservando a proporção | Tornar comparáveis imagens de diferentes resoluções e reduzir o custo nas fotos grandes | `src/preprocess/image.py`: `normalize_size()` |
| Registro dos fatores reais de escala em largura e altura | Converter corretamente caixas e coordenadas, inclusive com arredondamento no redimensionamento | `normalize_size()` e `detect()` |
| Limitação das caixas à região visível da imagem | Evitar recortes incorretos e avaliar o objeto visível quando a anotação ultrapassa a borda | `src/utils/geometry.py`: `clip_box()`; treino e avaliação ignoram caixas sem área |
| Até 12 tamanhos-base estimados nas anotações de train | Adequar as janelas às dimensões e proporções dos livros, incluindo orientações horizontais | `src/detection/windows.py`: `fit_window_config()` usa K-Means sobre larguras e alturas em escala logarítmica |
| Variações de tamanho por fatores 0,8, 1,0 e 1,25 | Cobrir objetos próximos, mas não idênticos, aos tamanhos-base | `DetectorConfig` e `window_boxes()` |
| Inclusão de janelas nas bordas e de uma janela da imagem inteira | Não perder livros junto às extremidades ou que ocupam grande parte da foto | `axis_positions()` e `window_boxes()` |
| Filtro de negativos por ocupação anotada, em vez de apenas IoU | Rejeitar fragmentos dentro de livros grandes que antes poderiam ser tratados como fundo | `overlap_matrices()` e `training_features()` |
| Até 15 negativos por imagem, em vez de três | Ampliar a quantidade e a variedade de exemplos de fundo | `training_features()` amostra sem reposição entre as janelas de busca elegíveis |
| Até duas janelas positivas adicionais por livro, com IoU >= 0,60 | Aproximar os exemplos de treinamento das janelas encontradas durante a detecção | `training_features()` mantém também o recorte da anotação e remove duplicatas entre as janelas extras |
| Previsões em lotes de até 256 descritores | Reduzir o número de chamadas individuais ao scaler e ao classificador | `src/detection/hog_detector.py`: `detect()` |
| NMS vetorizado, mantendo o limiar de IoU em 0,35 | Reduzir o custo dos cálculos de sobreposição sem mudar o objetivo da supressão | `non_maximum_suppression()` |
| Configuração de busca armazenada no classificador como `detector_config_` | Reutilizar no app os mesmos tamanhos e a mesma resolução adotados no experimento | `train_classifier()` e persistência do pipeline em pickle |
| Medição de cobertura independente do modelo | Distinguir falhas na geração de caixas de erros de classificação | `src/evaluation/detection.py`: `candidate_coverage()` |
| Atualização do notebook e preservação do baseline | Evitar apresentar resultados antigos como resultados do novo protocolo | Células de experimento tiveram suas saídas antigas limpas; métricas completas registradas foram preservadas em `docs/baseline.md` |

O filtro de negativos admite no máximo 1% de ocupação: soma as interseções com as
caixas dos livros e divide pela área da janela. Anotações sobrepostas podem ser
contadas mais de uma vez, tornando o critério conservador. A regra depende de
anotações completas e não garante ausência de livros não anotados.

### Configuração e verificações finais

| Item | Protocolo inicial | Protocolo corrigido |
| --- | --- | --- |
| Normalização espacial da imagem completa | Não havia | Maior lado de 640 pixels |
| Tamanhos de busca | Fixos, com proporções verticais | Estimados somente em train, com diferentes proporções |
| Positivos de treinamento | 204 | 595 |
| Negativos de treinamento | 468 | 2.145 |
| Total de exemplos de treinamento | 672 | 2.740 |
| Características por recorte | 5.940 | 5.940 |
| Cobertura em valid | No máximo 35/77, supondo posição ideal | 75/77 com as posições efetivamente geradas |

A configuração final usa semente `42`, `n_init=10` no K-Means, passo máximo de
busca de 32 pixels e passo adaptativo de pelo menos 16 pixels para janelas pequenas
quando o passo padrão é mantido. A busca final gera 742.515 janelas nas 45 imagens
de validação. A configuração foi estimada a partir de `train`; `valid` foi usado
para verificar a cobertura, sem fornecer anotações ao agrupamento.

Durante a implementação, uma versão com passo mínimo de oito pixels atingiu a
mesma cobertura de 75/77, mas gerou 1.754.963 janelas. O mínimo foi elevado para
16 pixels para reduzir esse custo. As contagens de treinamento apresentadas acima
correspondem à configuração final, não à versão intermediária.

Foram executados sete testes de regressão, cobrindo ocupação de livro dentro de
uma janela, negativos amostrados, normalização e retorno às coordenadas originais,
equivalência das previsões em lotes, bordas, NMS e persistência da configuração.
Todos passaram. Ruff, sintaxe do notebook e inicialização do app também passaram.
Os pickles existentes e os arquivos originais do dataset foram preservados.

**75/77 representa cobertura geométrica de 97,4%, não revocação de classificação.**
Ainda é necessário retreinar e avaliar os quatro classificadores com o novo
protocolo. Um teste funcional com Regressão Logística em apenas três imagens foi
executado durante a implementação; ele não constitui uma avaliação completa e
não deve ser comparado diretamente ao baseline das 45 imagens.

### O que permaneceu igual e o que falta avaliar

Continuam sendo utilizados Regressão Logística, SVM linear, SVM RBF e MLP, com os
mesmos hiperparâmetros de classificação anteriores. O HOG continua recebendo
recortes em escala de cinza, com CLAHE e redimensionamento para `96 × 128` pixels.
São mantidos nove bins de orientação, células `8 × 8`, blocos `16 × 16` e passo
`8 × 8`, resultando em 5.940 características.

Ainda não foram implementados hard negative mining, PCA, preservação de proporção
no recorte HOG ou escolha automática de limiar por modelo. A rede MLP permanece
com camadas ocultas de 128 e 64 neurônios. O RBF ainda habilita estimativas de
probabilidade, embora a detecção use sua `decision_function`.

Pickles antigos podem ser carregados, mas, por não conterem `detector_config_`,
usam a configuração padrão da busca atual. Portanto, executar um pickle antigo
no código atual não reproduz necessariamente o resultado histórico do baseline.
Para novos experimentos, é necessário regenerar os exemplos e retreinar.

## 06/10/2026 — Organização e preservação dos experimentos

Estas decisões antecederam a correção do pipeline e foram incluídas neste registro
para manter o contexto do experimento:

- **Retirada de Random Forest e KNN:** os tempos de avaliação inviabilizavam a
  comparação dentro do prazo. Foram removidos das configurações, do notebook e do
  app. Os pickles foram movidos para `tmp/retired-models-*`, permitindo recuperação.
- **Renomeação de `hog_svm.py` para `hog_detector.py`:** o detector passou a atender
  diferentes classificadores. Imports e documentação foram atualizados.
- **Salvamento sem sobrescrita:** novos arquivos recebem o primeiro sufixo livre,
  como `mlp_001.pkl` e `mlp_002.pkl`. A criação exclusiva impede que salvamentos
  simultâneos sobrescrevam um arquivo. A função retorna o caminho efetivamente salvo.
- **Seleção de versões no app:** os nomes numerados passaram a ser reconhecidos,
  permitindo escolher qual modelo salvo será executado.
- **Recarga da persistência no notebook:** os módulos de caminhos e salvamento são
  recarregados nessa célula para permitir salvar os modelos já treinados sem
  reiniciar o kernel. Antes de iniciar um novo experimento com alterações nos
  módulos de detecção, deve-se salvar os modelos atuais e reiniciar o kernel.

O sufixo do pickle indica um salvamento, não identifica sozinho o protocolo de
treinamento. Um mesmo modelo pode ser salvo várias vezes sem ser retreinado.

## 06/10/2026 — Foco em Regressão Logística e comparação de limiares

### Decisão e motivos

O escopo ativo passou a manter somente Regressão Logística e MLP, para priorizar
a qualidade do treinamento e da decisão dentro do prazo. Os SVMs foram retirados
da fábrica de classificadores, das células, da comparação e das opções do app.
Seus resultados anteriores permanecem nos registros históricos.

Antes desta etapa, a regressão corrigida foi avaliada nas 45 imagens de `valid`
com limiar zero: TP = 53, FP = 8.567, FN = 24, precisão = 0,00614849,
revocação = 0,68831169 e F1 = 0,01218811. A localização melhorou, mas o excesso de
falsos positivos motivou a seleção de um limiar mais exigente.

### Alterações implementadas

- `src/evaluation/thresholds.py` compara vários limiares reutilizando as previsões
  e o NMS do menor limiar. Cada imagem fornece métricas de detecção por IoU, e não
  métricas de classificação isolada dos recortes. Não há repetição de HOG por limiar.
- A regressão é treinada uma vez sobre os mesmos 2.740 recortes, com os parâmetros
  anteriores. Limiares modificam a decisão, não exigem um novo treinamento.
- A grade inicial contém 0, 0,5, 1, 1,5, 2, 3, 4, 5, 6, 8 e 10. A seleção prioriza
  precisão com revocação mínima de 50% em valid, com desempate por F1 e menor limiar.
- O limiar escolhido é salvo em `score_threshold_`; o app o carrega como valor
  padrão. A versão também preserva `training_metadata_` e `detector_config_`.
- As tabelas CSV são salvas em `docs/results/`, associadas ao nome do pickle e sem
  sobrescrever relatórios anteriores.
- O notebook passou a ter células para treino, comparação, gráfico e seleção do
  limiar. A demonstração usa valid; test fica reservado à avaliação final.
- `src/evaluation/run_logistic.py` executa o mesmo experimento pela linha de comando,
  com progresso a cada cinco imagens. Usa uma thread do OpenCV nesse processo para
  evitar overhead de paralelismo a cada recorte pequeno.
- A comparação com mais negativos é uma etapa posterior e opcional, desativada
  inicialmente. Compara até 30 e 60 negativos por imagem mantendo os demais
  parâmetros. É necessário analisar a grade de limiares antes de ativá-la.

O MLP ainda usa limiar padrão zero. Sua pontuação é probabilidade menos 0,5,
diferente da escala da regressão. A tabela entre os dois modelos deve indicar
que a regressão foi ajustada na validação e o MLP ainda não recebeu esse ajuste.

### Resultados efetivamente obtidos nas 45 imagens de valid

Foram executados dois treinos de Regressão Logística. Cada treino passou uma
única vez pelas imagens de validação e reutilizou as previsões para os onze
limiares. Nenhuma imagem de test foi utilizada.

| Experimento | Exemplos de treino | Limiar | TP | FP | FN | Precisão | Revocação | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Até 15 negativos, referência | 2.740 | 0 | 53 | 8.567 | 24 | 0,61% | 68,83% | 0,01219 |
| Até 15 negativos | 2.740 | 10 | 43 | 1.239 | 34 | 3,35% | 55,84% | 0,06328 |
| Até 30 negativos | 4.885 | 10 | 40 | 896 | 37 | 4,27% | 51,95% | 0,07897 |

Os dois conjuntos contêm 595 positivos. Os negativos aumentaram de 2.145 para
4.290. Foram mantidos HOG, janelas, positivos, semente 42, regularização e
`class_weight="balanced"`. A quantidade modifica também a amostra sorteada:
não foi imposta uma relação de subconjunto entre os negativos dos dois treinos.

O limiar 10 foi o melhor da grade para ambos sob a restrição de revocação mínima
de 50%. Por ser o maior valor testado, não é um ótimo global comprovado. O primeiro
experimento levou 441,7 segundos e o segundo 428,6 segundos, incluindo preparação,
treinamento, avaliação e salvamento. Esses tempos pertencem ao ambiente local e
não devem ser interpretados como uma vantagem causal de usar mais negativos.

O ajuste do limiar reduziu FP em 85,5% no treino com 15 negativos. Mantendo limiar
10, a variante com 30 negativos reduziu FP de 1.239 para 896, perdendo três TP.
Comparada à referência no limiar zero, a redução total foi de 89,5%, com perda de
13 TP. A precisão final de 4,27% ainda é baixa; a POC mantém limitações importantes.

Artefatos associados:

- `models/logistic_regression_001.pkl`: até 15 negativos, limiar selecionado 10.
- [Tabela de 15 negativos](results/logistic_regression_001_thresholds.csv) e
  [gráfico](results/logistic_regression_001_thresholds.png).
- `models/logistic_regression_002.pkl`: até 30 negativos, limiar selecionado 10.
- [Tabela de 30 negativos](results/logistic_regression_002_thresholds.csv) e
  [gráfico](results/logistic_regression_002_thresholds.png).

As versões anteriores não foram sobrescritas. Os pickles SVM foram movidos para
`tmp/retired-svm-*`, mantendo recuperação possível. O app oferece somente
Regressão Logística e MLP e utiliza o limiar gravado na versão selecionada.

A célula "Resultados já calculados" lê esses artefatos sem repetir os treinos.
O experimento com 60 negativos permanece preparado, mas não foi executado. Hard
negative mining e a avaliação final em test continuam pendentes. O MLP foi mantido
no projeto, mas não foi retreinado automaticamente nesta rodada.

Foram verificados dez testes, leitura e preservação dos relatórios, ausência de
SVMs no código ativo e carregamento do limiar salvo na interface Streamlit.

## 06/10/2026 — Rodada focada em precisão e notebook de apresentação

### Hipóteses e mudanças implementadas

- Manter somente Regressão Logística e MLP, sem adicionar novas famílias.
- Testar `C=0,01` contra `C=1` com os mesmos 4.885 recortes (595 positivos e
  4.290 negativos aleatórios). A regularização mais forte é uma hipótese para
  reduzir ajuste excessivo às texturas de train; não uma melhoria garantida.
- Adicionar uma rodada de negativos difíceis à referência C=1, sem alterar C
  nesse experimento. A referência é `logistic_regression_002`, treinada em train.
  Mineração em 30 imagens de train, escolhidas sem reposição com semente 42;
  até oito janelas por imagem, em ordem de pontuação após NMS. Somente janelas
  com ocupação anotada total ≤ 1% são consideradas fundo. Foram obtidos 172
  descritores adicionais (5.057 recortes no novo treino, mantendo 595 positivos).
- Treinar MLP sobre os mesmos 4.885 exemplos aleatórios da referência. Não recebe
  os negativos difíceis nesta rodada; essa diferença é explícita na comparação.
- Ampliar a grade da regressão até 25 e ajustar também os limiares do MLP, na
  escala `p(livro) - 0,5`. O critério comum permanece maior precisão com recall
  mínimo de 50%, IoU de correspondência ≥ 0,50 e desempate por F1/menor limiar.
- `detect_many` e `evaluate_models_thresholds` compartilham HOG por lote entre
  modelos com as mesmas janelas. As caixas, NMS e correspondências continuam
  separados por modelo. Equivalência com execuções individuais é testada.
- `run_precision.py` reproduz o protocolo limitado e salva novos pickles, CSVs
  e um manifesto versionado `precision_run_*.json`. A referência carregada não
  é retreinada: recebe uma nova avaliação da grade ampliada e uma nova cópia
  versionada com o limiar correspondente, preservando seu arquivo de origem.

### Organização e apresentação

O notebook anterior foi preservado em
`tmp/main-before-presentation-20261006-a1e7.ipynb`, incluindo suas saídas antigas.
O novo notebook organiza: (1) introdução do pipeline; (2) descrição dos dados,
15 imagens anotadas de train e pré-processamento ilustrado; (3) treinamento por
modelo/variante; (4) validação e limiares; (5) comparação, gráficos e discussão.

O modo padrão lê o manifesto, os pickles e as tabelas já calculadas, sem repetir
treinos ou a validação completa. A demonstração usa o primeiro registro de valid
e exibe todas as caixas previstas, sem escolher o melhor caso nem aplicar um
limite artificial de caixas. Figuras podem ser exportadas com nomes versionados
para o artigo. O README acompanha o novo roteiro e explica as opções de execução.
O nbclient foi adicionado às dependências de desenvolvimento para verificar e
salvar as saídas reais do notebook.

### Cuidados de interpretação

Train fornece os exemplos, as janelas e os negativos difíceis; valid escolhe
modelos e limiares. Test não participa da rodada. A divisão interna de recortes
do early stopping do MLP pode conter recortes correlacionados nos dois lados;
seu escore não é apresentado como métrica final de detecção. Resultados de valid
são usados para seleção e não substituem a avaliação final independente.

Os testes funcionais cobrem regularização, orçamento da mineração, exclusão
de fragmentos anotados e equivalência da avaliação compartilhada. Os resultados
quantitativos e a decisão da rodada são registrados no complemento abaixo.

### Resultados completos da rodada em valid

Avaliação nas 45 imagens e 77 livros anotados. Os quatro experimentos compartilham
as mesmas janelas e HOG; o conjunto test não foi avaliado. Tempo total: 842,0 s,
incluindo preparação, dois novos treinos de regressão, um treino de MLP, mineração,
validação e salvamento.

| Experimento | Recortes | Limiar | TP | FP | FN | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Referência: 30 negativos, C=1 | 4.885 | 10 | 40 | 896 | 37 | 4,27% | 51,95% | 0,07897 |
| Regularização C=0,01 | 4.885 | 6 | 39 | 780 | 38 | 4,76% | 50,65% | 0,08705 |
| Negativos difíceis, C=1 | 5.057 | 4 | 39 | 2.843 | 38 | 1,35% | 50,65% | 0,02636 |
| MLP, negativos aleatórios | 4.885 | 0,495 | 43 | 948 | 34 | 4,34% | 55,84% | 0,08052 |

**Decisão:** manter `logistic_regression_004`, com C=0,01 e limiar 6, como escolha
de precisão sob recall mínimo de 50%. Em comparação com a versão 002, FP caiu de
896 para 780 (12,9%), com um TP a menos. O ganho de precisão é de aproximadamente
0,49 ponto percentual (4,27% → 4,76%). Não é uma solução de alta precisão.

O MLP com limiar 0,495 equivale a probabilidade mínima de 99,5%, recupera mais
livros que a regressão escolhida e mantém precisão ligeiramente menor. A rodada
de negativos difíceis piorou a precisão; foi preservada como resultado negativo,
sem incorporá-la à configuração recomendada. Isso não demonstra que mineração
de negativos seja sempre prejudicial, apenas que esta rodada limitada não ajudou.

Há pontos de maior precisão que não atendem ao recall mínimo: C=0,01 e limiar 10
produz TP=19, FP=102, FN=58, precisão 15,70%, recall 24,68% e F1=0,19192. Escolher
somente precisão ou F1 levaria a outra troca entre falsos alarmes e livros perdidos;
mantivemos o critério anunciado antes do experimento.

Artefatos, todos novos e sem substituir versões anteriores:

- `logistic_regression_003.pkl`: pesos da versão 002 e referência na grade ampliada.
- `logistic_regression_004.pkl`: regressão C=0,01, limiar 6, recomendada.
- `logistic_regression_005.pkl`: regressão com 172 negativos difíceis, limiar 4.
- `mlp.pkl`: MLP da rodada, limiar 0,495.
- [Manifesto](results/precision_run_001.json), com protocolos, seleção e fontes da mineração.
- [Referência](results/logistic_regression_003_thresholds.csv),
  [regularização](results/logistic_regression_004_thresholds.csv),
  [negativos difíceis](results/logistic_regression_005_thresholds.csv) e
  [MLP](results/mlp_thresholds.csv): grades completas, inclusive opções rejeitadas.

O notebook foi executado no modo apresentação com saídas reais salvas, sem novos
treinos. Os gráficos de dados e pré-processamento foram conferidos visualmente.
Foram executadas as 18 células de código sem erro, com seis figuras e nove
tabelas HTML embutidas. As figuras finais de comparação e limiares também foram
exportadas e conferidas: [comparação](results/comparacao_modelos.png) e
[limiares](results/curvas_limiares.png). A nova referência reproduziu exatamente
as métricas da versão 002 nos limiares compartilhados pelas duas grades.
O app passou a mostrar três casas decimais e passo 0,001 para limiares do MLP,
evitando arredondar visualmente 0,495 para 0,50. A lógica de pontuação não mudou.
O teste da interface carregou `mlp.pkl` e confirmou limiar 0,495 e formato `%.3f`.
Ruff e quatorze testes passaram; as versões antigas foram preservadas.

### Próximo passo antes de relatar desempenho final

A avaliação independente em test continua pendente. É necessário congelar o
modelo e o limiar escolhidos antes dela. As conclusões desta rodada referem-se
somente à seleção em valid, não ao desempenho final fora dos dados de ajuste.

## 06/10/2026 — Streamlit com rótulos descritivos e comparação de todos os modelos

### Solicitação e implementação

Substituir os números de versões por descrições compreensíveis e analisar uma
mesma imagem com todos os modelos salvos, sem obrigar a escolha de um único modelo.

- A interface lista todos os pickles da Regressão Logística e do MLP. O seletor
  foi removido; um botão executa a comparação de todas as versões disponíveis.
- Os rótulos explicam baseline, negativos aleatórios, regularização, mineração
  e arquitetura do MLP. O arquivo original continua visível em cada resultado.
  A referência reavaliada é identificada como tendo os mesmos pesos da versão 002.
- Cada modelo utiliza seu limiar salvo, ajustável individualmente na barra
  lateral. O MLP mantém três casas decimais; o passo é comum a todos os modelos.
- HOG é compartilhado por grupo de configurações de janelas iguais. O baseline
  original usa a configuração padrão em outro grupo. As caixas continuam
  independentes, sem votação, fusão ou remoção artificial de falsos alarmes.
- O tempo reportado é o total da comparação, não uma latência individual
  atribuída indevidamente aos modelos que compartilham a extração HOG.
- A tela exibe resumo, imagens em duas colunas e coordenadas/pontuações em painéis
  expansíveis. Os resultados persistem nos reruns da sessão; mudanças na imagem,
  arquivos ou parâmetros invalidam a comparação anterior para não mostrar saídas
  antigas como se fossem novas.
- Falhas de um pickle não interrompem o carregamento dos demais. Se um grupo
  falhar na previsão, a execução individual isola o modelo inválido e preserva
  resultados dos outros. Erros são exibidos como indisponibilidade, não como
  ausência de livros.

### Verificações e limites

Foram adicionados cinco testes: rótulos, limiares individuais, extração HOG
compartilhada, grupos distintos/erros de carregamento, isolamento de erro de
previsão e fluxo da interface com upload único e invalidação dos resultados.
O total passou a dezenove testes, todos aprovados. Nenhum peso, pickle, CSV ou
resultado de avaliação foi alterado; não houve novo treinamento.

A integração também foi verificada com os sete pickles reais no primeiro
registro de train, sem usar test. Todos produziram previsões sem erro, com seus
limiares individuais. As versões 002/003 retornaram caixas e pontuações idênticas.
A comparação levou 25,03 s no processo de verificação (OpenCV com uma thread,
BLAS limitado a duas); isso não é um benchmark de latência individual nem uma
nova avaliação de precisão.

O app compara previsões, não desempenho: sem caixas reais da imagem enviada, não
há cálculo de precisão, recall ou TP/FP. O README descreve o novo fluxo. A
avaliação final em test continua pendente, independente dessa mudança de interface.

## 06/10/2026 — Exemplos de validação para cada modelo no notebook

A seção 6.3 passa a mostrar os quatro experimentos da comparação, em vez de
somente o modelo recomendado: regressão de referência, regressão com C=0,01,
regressão com negativos difíceis e MLP. Todos analisam o primeiro registro de
valid, sem selecionar uma imagem por apresentar melhor resultado.

Cada modelo usa seu limiar selecionado em valid. A extração HOG é compartilhada
por meio de `detect_many`, preservando previsões independentes. A seção inclui
uma tabela de TP, FP, FN, limiar e quantidade de caixas para essa imagem e uma
figura por experimento, com anotações reais à esquerda e todas as previsões à
direita. As figuras recebem chaves diferentes no dicionário de exportação, para
não substituir umas às outras. Se um modelo não tiver limiar elegível, o fallback
é explicitamente identificado.

As caixas reais usadas nas contagens são limitadas à área visível da imagem,
como na avaliação agregada. Não foram alterados pesos, limiares selecionados,
CSVs ou critérios de comparação. Não há novo treinamento nem avaliação em test;
os exemplos são qualitativos e as contagens referem-se somente à imagem exibida.

## 06/10/2026 — Identificação explícita da POC

A introdução do notebook e o README identificam a aplicação Streamlit, em
`src/app/`, como a Prova de Conceito (POC) do trabalho. O texto diferencia o
papel do notebook (metodologia, treinamento e avaliação) do papel da aplicação
(demonstração prática com uma imagem enviada pelo usuário e os modelos salvos).

Foram incluídos o comando de execução e o fluxo de envio/comparação no notebook.
A documentação esclarece que a demonstração pode usar uma imagem nova ou
previamente obtida, não repete o treino e não substitui a avaliação quantitativa.
A alteração é somente documental; código, pesos, limiares, métricas e saídas
já salvas do notebook foram preservados.

## 06/10/2026 — Avaliação final dos modelos em test

### Protocolo congelado

Foi adicionada a avaliação final nas 22 imagens de test, com 37 livros anotados,
após a seleção dos modelos e limiares em valid. Os quatro arquivos avaliados são
`logistic_regression_003.pkl`, `logistic_regression_004.pkl`,
`logistic_regression_005.pkl` e `mlp.pkl`, com limiares 10, 6, 4 e 0,495.
Mantivemos os pesos, janelas, passo 32 e IoU de correspondência ≥ 0,50. Não houve
treinamento, mineração de negativos, seleção de modelos ou otimização de limiares
em test. A recomendação anterior, da versão 004 com limiar 6, foi mantida.

`src/evaluation/test_set.py` lê o manifesto de valid e usa um único limiar por
modelo. O relatório inclui hashes das imagens/anotações e dos pickles, impedindo
reutilizar silenciosamente resultados de arquivos diferentes. Quando o protocolo
é idêntico, os reruns leem o relatório salvo sem repetir a inferência. Novos
relatórios são versionados e não substituem os anteriores.

### Resultados nas 22 imagens

| Experimento | Limiar | TP | FP | FN | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Referência, 30 negativos | 10 | 19 | 499 | 18 | 3,67% | 51,35% | 0,06847 |
| Regularização C=0,01 | 6 | 19 | 456 | 18 | 4,00% | 51,35% | 0,07422 |
| Negativos difíceis, C=1 | 4 | 22 | 1.560 | 15 | 1,39% | 59,46% | 0,02718 |
| MLP | 0,495 | 19 | 501 | 18 | 3,65% | 51,35% | 0,06822 |

Tempo total: 267,76 s, incluindo carregamento e inferência compartilhada dos
modelos. O relatório completo é
[test_evaluation_001.json](results/test_evaluation_001.json). A regressão C=0,01
teve precisão de 4,00% em test, comparada a 4,76% em valid. As métricas continuam
limitadas e não foram usadas para reajustar o detector.

### Notebook, documentação e verificações

A nova seção 6.4 mostra os snapshots e os limiares congelados, as contagens e
métricas finais e um gráfico valid × test, sem misturar métricas de seleção com
as de avaliação final. As conclusões passam à seção 6.5. A seção 6.3 e a imagem
de demonstração escolhida pelo usuário foram preservadas. O README passa a
registrar a avaliação final como concluída; registros anteriores de test pendente
permanecem no histórico porque descrevem etapas anteriores.

Foram adicionados cinco testes sobre origem dos limiares, recomendação mantida
de valid mesmo se outro modelo obtiver maior precisão em test, cache de resultados,
hashes, relatórios versionados e rejeição de gráficos com limiares diferentes.
Os vinte e quatro testes passaram, assim como o Ruff. Nenhum pickle, relatório
de valid ou treinamento foi alterado. O artigo em edição não foi modificado.
O notebook executou as 21 células de código sem erro e carregou o relatório de
test já salvo, sem repetir a avaliação. O
[gráfico valid × test](results/comparacao_valid_test.png) foi exportado e
conferido visualmente; a tabela e o gráfico também estão embutidos no notebook.

Após observar os resultados de test, novos ajustes não podem continuar tratando
essa mesma divisão como avaliação final não observada; deve-se adotar um novo
protocolo independente. O comando de consulta e a apresentação reutilizam a
avaliação salva, sem transformar test em uma grade de otimização.

## Como registrar os próximos experimentos

Acrescente uma entrada datada contendo:

1. Problema observado e hipótese da alteração.
2. Arquivos e parâmetros modificados, com justificativa.
3. Divisões utilizadas: train para aprendizado, valid para ajustes e test para a
   avaliação final após escolher a configuração.
4. Quantidades de positivos e negativos, configuração das janelas e nomes exatos
   dos pickles gerados, associando cada salvamento ao seu protocolo.
5. Resultados completos de TP, FP, FN, precisão, revocação, F1 e tempo de execução.
6. Comparação com o protocolo anterior, limitações e decisão de manter ou descartar
   a alteração. Resultados de subconjuntos devem indicar quais imagens foram usadas.

Não substituir registros antigos por resultados novos. Uma melhoria geométrica,
um teste funcional e uma avaliação completa devem ser identificados separadamente.
