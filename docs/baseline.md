# Resultados registrados do pipeline inicial

Estes resultados foram extraídos das saídas salvas do notebook antes da correção
da amostragem e da geração das janelas. Não representam o pipeline atual.

Protocolo: 45 imagens e 77 livros anotados em `valid`, IoU mínima de 0,50 e limiar
de classificação zero. O treino usava 672 recortes, com 5.940 características HOG
por recorte, e até três negativos aleatórios por imagem.

| Modelo | TP | FP | FN | Precisão | Revocação | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regressão Logística | 17 | 6.654 | 60 | 0,002548 | 0,220779 | 0,005039 |
| SVM linear | 17 | 14.587 | 60 | 0,001164 | 0,220779 | 0,002316 |
| SVM RBF | 14 | 2.446 | 63 | 0,005691 | 0,181818 | 0,011037 |

Não havia resultado completo do MLP salvo no arquivo na ocasião da revisão.

Pelo tamanho das janelas antigas, somente 35 dos 77 livros poderiam alcançar IoU
de 0,50 mesmo com posicionamento ideal. A nova configuração, estimada apenas em
`train`, alcança 75 dos 77 livros com as posições de busca realmente geradas.
Essa cobertura geométrica não é a revocação observada de um classificador.

É necessário retreinar e avaliar os quatro modelos para completar a comparação.
Nenhum resultado novo completo de classificação é apresentado neste registro.
