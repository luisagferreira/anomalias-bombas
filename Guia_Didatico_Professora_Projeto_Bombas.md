# Guia: Detecção de Anomalias em Bombas Industriais com Autoencoders 1D, 2D (GADF) e Híbridos Multimodais

Este documento está organizado em **8 módulos** para explicar detalhadamente cada decisão do projeto, as fórmulas, as arquiteturas, o fluxo dos tensores (entrada, gargalo e saída), o significado das métricas, as matrizes de confusão e os resultados obtidos em GPU.

---

## Módulo 1 — Contextualização do Problema e Resumo do Dataset
*(Como introduzir o problema para a turma ou banca)*

> **Fala sugerida:**
> *"Na indústria de saneamento e bombeamento pesado, monitorar apenas se a corrente elétrica ultrapassou um limite fixo não basta para prever falhas incipientes ou distinguir uma mudança normal de regime de carga de um fenômeno destrutivo como a cavitação. Nosso objetivo neste projeto foi construir e comparar 7 arquiteturas de Aprendizado Profundo — operando no domínio do tempo (1D), no domínio de imagens polares (2D GADF) e em fusão multimodal (Híbridos 1D+2D) — para detectar anomalias operacionais reais em 6 bombas industriais ao longo de todo o ano de 2026."*

### 1.1. Estrutura Física dos Dados Coletados
* **Período analisado:** `01/01/2026` a `26/10/2026` ($\approx 300\text{ dias}$ contínuos, totalizando $2.574.750$ linhas por conjunto).
* **Equipamentos monitorados:** **6 bombas industriais** (`Bomba_1305`, `Bomba_1307`, `Bomba_2307`, `Bomba_3307`, `Bomba_4307` e `Bomba_5307`) + o **`Dataset_Completo`** (planta inteira com as 6 bombas analisadas simultaneamente).
* **4 Sensores por Bomba ($C = 4$ nas bombas individuais; $C = 24$ no `Dataset_Completo`):**
  1. **`BR_YAR` (Corrente elétrica / Amperagem em $\text{A}$):** Principal indicador de esforço eletromecânico e oscilações de torque (cavitação/mancal).
  2. **`BR_YL` (Nível do reservatório/sucção em $\text{m}$ ou $\%$):** Condição de contorno hidráulica (nível baixo favorece cavitação na sucção).
  3. **`BR_YLOAD` (Carga / Potência solicitada em $\%$):** Define o regime operacional da máquina.
  4. **`BR_YP` (Pressão de descarga em $\text{bar}$ / $\text{kgf/cm}^2$):** Entrega hidráulica efetiva da bomba.

### 1.2. A Diferença entre a Taxa do Relé Físico e a Taxa do Historiador PI
Uma dúvida clássica em bancas é a relação entre os **$961,5\text{ Hz}$** do relé e os **$10\text{ segundos}$** do arquivo CSV:
1. **No Hardware em Campo (Relé GE Multilin 369):**
   * Amostra a onda senoidal de $60\text{ Hz}$ à razão de $16\text{ amostras/ciclo}$:
     $$f_{s,\text{relé}} = 16 \times 60\text{ Hz} \approx 961,5\text{ Hz} \quad \left(T_s = \frac{1}{961,5} \approx 1,04\text{ ms}\right)$$
   * Frequência máxima de Nyquist no relé: $f_{\text{Nyquist}} \approx 480,8\text{ Hz}$.
2. **No Historiador OSIsoft PI (Arquivos CSV do Projeto):**
   * O sistema supervisório consolida e grava uma leitura a cada **$\Delta t = 10\text{ segundos}$** ($f_{s,\text{PI}} = 0,1\text{ Hz}$), capturando a dinâmica de processo, transientes hidromecânicos e volatilidade de corrente.

### 1.3. Janelamento Multiescala e o Significado do Tensor `(N, C, L)`
Para avaliar em qual horizonte temporal as anomalias ficam mais nítidas, deslizamos janelas com **50% de sobreposição (*stride* = $L/2$)** em 4 escalas:

| Escala da Janela | Passos de Tempo ($L$, a cada $10\text{ s}$) | Passo de Deslizamento (*Stride*) | Quantidade Total de Janelas ($N$) | Formato do Tensor 1D `(N, C, L)` (1 Bomba / 6 Bombas) |
| :---: | :---: | :---: | :---: | :---: |
| **`15min`** | **$L = 90$** ($900\text{ s}$) | $45$ ($7,5\text{ min}$) | **$N = 57.215$** | `(57215, 4, 90)` / `(57215, 24, 90)` |
| **`30min`** | **$L = 180$** ($1.800\text{ s}$) | $90$ ($15\text{ min}$) | **$N = 28.607$** | `(28607, 4, 180)` / `(28607, 24, 180)` |
| **`1h`** | **$L = 360$** ($3.600\text{ s}$) | $180$ ($30\text{ min}$) | **$N = 14.303$** | `(14303, 4, 360)` / `(14303, 24, 360)` |
| **`2h`** | **$L = 720$** ($7.200\text{ s}$) | $360$ ($60\text{ min}$) | **$N = 7.151$** | `(7151, 4, 720)` / `(7151, 24, 720)` |

### 1.4. Rótulos Operacionais de Anomalia no Ano Completo ($\sigma_{\text{5min}} \ge 2,5\text{ A}$)
Seguindo o critério operacional de campo, o ano inteiro foi dividido em **85.825 blocos de 5 minutos** por bomba ($514.950$ blocos nas 6 bombas), calculando-se o desvio padrão amostral da corrente elétrica ($\sigma_{\text{5min}}$). Todo bloco com **$\sigma_{\text{5min}} \ge 2,5\text{ A}$** é catalogado como anomalia (**69.081 blocos anômalos** no ano):

| Dataset / Bomba | Blocos de `5min` com $\sigma \ge 2,5\text{ A}$ | Janelas Anômalas (`15min`) | Janelas Anômalas (`30min`) | Janelas Anômalas (`1h`) | Janelas Anômalas (`2h`) | Perfil Operacional da Bomba |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Bomba_1305`** | 1.245 (`1,45%`) | 2.155 (`3,77%`) | 1.953 (`6,83%`) | 1.725 (`12,06%`) | 1.457 (`20,37%`) | Anomalias raras e concentradas |
| **`Bomba_1307`** | 10.217 (`11,90%`) | 10.242 (`17,90%`) | 6.471 (`22,62%`) | 4.099 (`28,66%`) | 2.643 (`36,96%`) | Transientes moderados e frequentes |
| **`Bomba_2307`** | 24.987 (`29,11%`) | 23.230 (`40,60%`) | 13.069 (`45,68%`) | 7.229 (`50,54%`) | 4.008 (`56,05%`) | Alta incidência de variação de carga/corrente |
| **`Bomba_3307`** | 26.897 (`31,34%`) | 21.415 (`37,43%`) | 11.904 (`41,61%`) | 6.733 (`47,07%`) | 3.876 (`54,20%`) | Bomba mais crítica (cavitação/oscilação severa) |
| **`Bomba_4307`** | 4.005 (`4,67%`) | 4.842 (`8,46%`) | 3.426 (`11,98%`) | 2.399 (`16,77%`) | 1.674 (`23,41%`) | Instabilidades intermitentes |
| **`Bomba_5307`** | 1.730 (`2,02%`) | 2.454 (`4,29%`) | 1.879 (`6,57%`) | 1.421 (`9,93%`) | 1.081 (`15,12%`) | Baixa taxa de falhas, picos agudos |
| **`Dataset_Completo`** | **69.081** (`13,42%`) | **39.032** (`68,22%`) | **21.054** (`73,60%`) | **11.245** (`78,62%`) | **5.950** (`83,21%`) | Planta consolidada (concatenação das 6 bombas) |

---

## Módulo 2 — Técnicas Escolhidas e Decisões Metodológicas
*(Por que fizemos cada escolha técnica?)*

### 2.1. Por que usar Redes Neurais se o operador olha para $\sigma \ge 2,5\text{ A}$?
1. **Contexto Multivariado (Corrente + Nível + Carga + Pressão):** Uma regra fixa de desvio padrão olha apenas para a corrente isolada. Já o Autoencoder aprende a relação física entre os 4 sensores (ex.: qual deveria ser a corrente e a pressão para um dado nível de sucção e carga).
2. **Generalização para Outras Falhas Hidromecânicas:** Quando ocorre perda de eficiência hidráulica, desalinhamento ou mudança de nível sem necessariamente ultrapassar o limiar fixo de corrente, o erro de reconstrução multivariado ($\|\mathbf{x} - \hat{\mathbf{x}}\|^2$) acusa o desvio de estado.
3. **Detecção Não Supervisionada:** Na fase de reconstrução, o Autoencoder é treinado **exclusivamente com janelas normais**, sem depender de rótulos humanos para aprender a física normal da máquina.

### 2.2. Validação Cruzada sem Vazamento Temporal: `StratifiedGroupKFold` pelo Rótulo de Anomalia (5 Folds)
* **O Problema do *Data Leakage* em Séries Temporais:** Como as janelas têm 50% de sobreposição, dividir janelas aleatoriamente colocaria metades do mesmo evento no treino e no teste.
* **Por que NÃO usamos mais o quantil de carga na estratificação?** Como agora possuímos o rótulo de anomalia do operador para todos os meses do ano (`y = 0` Normal, `y = 1` Anomalia), não é necessário dividir em 10 quantis de carga. Estratificar diretamente pelo rótulo `y` elimina subgrupos pequenos e garante equilíbrio exato de anomalias entre os 5 folds.
* **Passo 1 — O Sorteio dos 5 Folds (`StratifiedGroupKFold`):**
  1. Dividimos o ano em **blocos temporais contíguos (`block_id`)** e removemos as janelas de fronteira entre blocos (`is_boundary_overlap`).
  2. O `StratifiedGroupKFold(n_splits=5)` olha para o **rótulo anual de anomalia (`y = 0` ou `y = 1`)** e para o **`block_id`** para distribuir a mesma quantidade de janelas normais e anômalas em cada um dos 5 folds:
     $$\text{sgkf.split}\left(X,\; y_{\text{Anomalia}},\; \text{groups}=\text{block\_id}\right)$$

| Dataset (`15min`) | Fold 0 (`Norm / Anom`) | Fold 1 (`Norm / Anom`) | Fold 2 (`Norm / Anom`) | Fold 3 (`Norm / Anom`) | Fold 4 (`Norm / Anom`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305`** | `11.082 / 438` | `11.067 / 453` | `11.091 / 428` | `10.913 / 415` | `10.907 / 421` |
| **`Bomba_3307`** | `7.264 / 4.256` | `7.167 / 4.353` | `7.057 / 4.270` | `7.070 / 4.258` | `7.242 / 4.278` |
| **`Dataset_Completo`** | `3.636 / 7.692` | `3.681 / 7.838` | `3.650 / 7.870` | `3.619 / 7.901` | `3.597 / 7.731` |

* **Passo 2 — As 5 Rodadas de Treino e Teste Cego (*Out-of-Fold*):**
  Em cada uma das 5 rodadas, **4 Folds (80%)** treinam o modelo e **1 Fold (20%)** funciona como prova cega:
  1. **Nos 4 Folds de Treino:** O Autoencoder usa o rótulo `y == 0` para aprender a reconstruir **somente as janelas normais**, e a cabeça classificadora híbrida usa `y` (`0` e `1`) do treino para calibrar a fronteira de alarme.
  2. **No Fold de Teste (Prova Cega):** O rótulo real é escondido do modelo; o modelo analisa apenas os sensores e emite sua predição (`0` ou `1`). Só depois comparamos com o gabarito real do operador para contabilizar `VP`, `VN`, `FP` e `FN`.
  3. **Normalização `Min-Max` em $[-1, 1]$ por Fold:** Calculada exclusivamente nos folds de treino.

### 2.3. Transformação de Séries Temporais em Imagens 2D: PAA ($32 \times 32$) + `GADF`
Para usar Redes Convolucionais 2D (`CNN2D`), transformamos cada série temporal 1D de comprimento $L$ em uma imagem polar $32 \times 32$:
1. **Redução via PAA (*Piecewise Aggregate Approximation*):** Comprime a série de $L \in \{90, 180, 360, 720\}$ para $M = 32$ segmentos médios.
   * *Por que usar PAA $32 \times 32$?* Sem o PAA, as matrizes teriam tamanho $L \times L$ (chegando a $720 \times 720$), o que exigiria **mais de $310\text{ GiB}$** de armazenamento. Com PAA $32 \times 32$ em `float16` e concatenando as 6 bombas em memória, todo o conjunto GADF ocupa apenas **$4,91\text{ GiB}$**.
2. **Coordenadas Polares e Matriz `GADF` (*Gramian Angular Difference Field*):**
   Cada ponto normalizado $\tilde{x}_i \in [-1, 1]$ é mapeado no ângulo $\phi_i = \arccos(\tilde{x}_i) \in [0, \pi]$, gerando a matriz $32 \times 32$:
   $$\text{GADF}_{i,j} = \sin(\phi_i - \phi_j) = \tilde{x}_j \sqrt{1 - \tilde{x}_i^2} - \tilde{x}_i \sqrt{1 - \tilde{x}_j^2}$$
3. **Por que escolhemos utilizar SOMENTE o `GADF` (descartando o `GASF`)?**
   * O `GASF` ($\cos(\phi_i + \phi_j)$) representa a soma estática de nível, enquanto o **`GADF` ($\sin(\phi_i - \phi_j)$) mede exatamente a diferença direcional (transições e oscilações temporais entre os instantes $i$ e $j$)**.
   * Ao usar apenas o `GADF`, reduzimos pela metade os canais 2D (de $2C$ para $C$) e aumentamos o `ROC-AUC` dos modelos 2D/Híbridos em **+2,4 a +3,65 pontos percentuais**, além de elevar a separação não supervisionada de `9,16x` para **`13,22x`**!

---

## Módulo 3 — Resumo das 7 Arquiteturas Avaliadas
*(Como explicar a estrutura interna de cada rede neural)*

Todas as 7 arquiteturas foram projetadas com **exatamente o mesmo gargalo latente ($\mathbf{z} \in \mathbb{R}^{32}$)** para permitir uma comparação científica justa entre 3 paradigmas:

| # | Arquitetura | Paradigma | Entrada | Estrutura do Codificador (*Encoder*) | Gargalo | Decodificador (*Decoder*) | Params ($C=4$) | Params ($C=24$) |
| :-: | :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **1** | **`AE_CNN1D`** | 1D Convolucional | `(B, C, L)` | 2× `Conv1d` ($16 \to 32$, $k=5, s=2$) + `BatchNorm1d` + `GELU` + `AdaptiveAvgPool1d(8)` $\to 256$ | **$Z = 32$** | `Linear(32→64→C×L)` + `Tanh` | 36.760 | 155.360 |
| **2** | **`AE_LSTM1D`** | 1D Recorrente | `(B, C, L)` | Divide em **15 patches** $\to$ `Linear(32)` + `ReLU` $\to$ `Bi-LSTM(hidden=32)` ($64$) + `MeanPool` | **$Z = 32$** | `Linear(32→64→C×L)` + `Tanh` | 45.288 | 166.128 |
| **3** | **`AE_Transformer1D`** | 1D Atenção | `(B, C, L)` | **15 patches** + `PosEmbed(1,15,32)` $\to$ `TransformerEncoder` ($d=32, H=4, d_{ff}=64$) $\to 480$ | **$Z = 32$** | `Linear(32→64→C×L)` + `Tanh` | 50.728 | 171.568 |
| **4** | **`AE_CNN2D_GAF`** | 2D Espacial (`GADF`) | `(B, C, 32, 32)` | 2× `Conv2d` ($16 \to 32$, $k=3, s=2$) + `BatchNorm2d` + `GELU` + `AdaptiveAvgPool2d(4,4)` $\to 512$ | **$Z = 32$** | `Decoder2D` (`ConvTransp2d`) + `Decoder1D` (cruzado) | 87.684 | 220.024 |
| **5** | **`Hybrid_CNN1D_CNN2D`** | Híbrido 1D+2D | 1D + GADF | `EncoderCNN1D` ($32$) + `EncoderCNN2D` ($32$) $\to$ `Concat(64)` $\to$ `Linear(64→32)` | **$Z = 32$** | `Decoder1D` + `Decoder2D` | 101.012 | 234.952 |
| **6** | **`Hybrid_LSTM1D_CNN2D`** | Híbrido 1D+2D | 1D + GADF | `EncoderLSTM1D` ($32$) + `EncoderCNN2D` ($32$) $\to$ `Concat(64)` $\to$ `Linear(64→32)` | **$Z = 32$** | `Decoder1D` + `Decoder2D` | 109.540 | 245.720 |
| **7** | **`Hybrid_Transformer_CNN2D`** | Híbrido 1D+2D | 1D + GADF | `EncoderTransf1D` ($32$) + `EncoderCNN2D` ($32$) $\to$ `Concat(64)` $\to$ `Linear(64→32)` | **$Z = 32$** | `Decoder1D` + `Decoder2D` | 114.980 | 251.160 |

### Destaques de Projeto das Arquiteturas para Explicar em Aula:
* **Por que dividir em `N_PATCHES = 15` no `AE_LSTM1D` e no `AE_Transformer1D`?** Porque todos os comprimentos de janela ($90, 180, 360, 720$) são exatamente divisíveis por $15$ ($P = 6, 12, 24, 48$ amostras por patch). Isso reduz sequências longas para apenas $15$ tokens temporais, evitando desaparecimento de gradiente (*vanishing gradient*) na LSTM e custo quadrático excessivo no Transformer.
* **Por que o `AE_CNN2D_GAF` possui uma cabeça de reconstrução cruzada 1D?** Além de reconstruir a imagem `GADF` $32 \times 32$, o vetor latente $\mathbf{z} \in \mathbb{R}^{32}$ também reconstrói a série temporal 1D original, forçando o codificador 2D a preservar a informação de amplitude física.
* **Dupla Etapa de Detecção (Não Supervisionada + Cabeça Híbrida *Out-of-Fold*):**
  1. **Modo Não Supervisionado Puro:** O Autoencoder minimiza o erro de reconstrução **apenas nas janelas normais (`y = 0`)**. Quando entra uma janela com cavitação/anomalia, o erro de reconstrução dispara.
  2. **Modo Híbrido Discriminativo (*Out-of-Fold*):** Um classificador leve (`MLP`) recebe o vetor latente $\mathbf{z} \in \mathbb{R}^{32}$ concatenado com $6 \times C$ descritores residuais (MSE 1D, erro máximo 1D, desvio padrão intrajanela, derivada de 1ª ordem, média de operação e MSE 2D GADF por canal) e estima a probabilidade de anomalia nos folds de teste.

---

## Módulo 4 — O Significado Didático de Cada Métrica
*(Como explicar cada coluna das tabelas de resultados)*

### 4.1. Os 4 Quadrantes da Matriz de Confusão na Planta Industrial
* **$\text{VN}$ (Verdadeiro Negativo):** A bomba operava **normalmente** e o modelo disse **Normal** *(acerto silencioso)*.
* **$\text{VP}$ (Verdadeiro Positivo):** A bomba estava em **anomalia ($\sigma \ge 2,5\text{ A}$)** e o modelo disparou **Alarme** *(detecção correta de falha)*.
* **$\text{FP}$ (Falso Positivo / Falso Alarme):** A bomba estava **normal**, mas o modelo disparou **Alarme indevido** *(gera fadiga de alarme no operador)*.
* **$\text{FN}$ (Falso Negativo / Omissão):** A bomba estava em **anomalia**, mas o modelo disse **Normal** *(o erro mais perigoso na manutenção)*.

### 4.2. Fórmulas e Interpretação Prática das Métricas

| Métrica | Fórmula | Como Explicar em Uma Frase (Modo Professora) |
| :--- | :---: | :--- |
| **Acurácia (`Accuracy`)** | $\frac{\text{VP} + \text{VN}}{\text{Total}}$ | *"De todas as janelas do ano, que porcentagem o modelo acertou no total?"* |
| **Precisão (`Precision`)** | $\frac{\text{VP}}{\text{VP} + \text{FP}}$ | *"Quando o alarme toca no painel, qual a chance de ser uma anomalia real (e não um alarme falso)?"* |
| **Sensibilidade (`Recall`)** | $\frac{\text{VP}}{\text{VP} + \text{FN}}$ | *"De todas as anomalias reais que aconteceram no ano, quantas por cento o modelo conseguiu capturar?"* |
| **`F1-Score`** | $2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$ | *"É o termômetro de equilíbrio: só dá nota alta se o modelo tiver poucos alarmes falsos E poucas falhas perdidas ao mesmo tempo."* |
| **Falsos Alarmes (`FAR`)** | $\frac{\text{FP}}{\text{VN} + \text{FP}}$ | *"Em quantos por cento do tempo normal da bomba o sistema incomodou o operador à toa?"* |
| **`ROC-AUC`** | Área sob $\text{Recall} \times \text{FAR}$ | *"Nota de $0,5$ a $1,0$ que mede o poder de separação global do modelo antes mesmo de escolher um limiar fixo de alarme."* |
| **`PR-AUC`** | Área sob $\text{Precisão} \times \text{Recall}$ | *"A métrica mais rigorosa quando as falhas são raras (como nas bombas 1305 e 5307), pois foca no desempenho sobre a classe anômala."* |
| **Razão de Score Não-Sup.** | $\frac{\text{Média}(\text{Erro}_{\text{Anom}})}{\text{Média}(\text{Erro}_{\text{Norm}})}$ | *"Sem usar nenhum rótulo do operador, quantas vezes o erro do Autoencoder salta quando a bomba entra em anomalia?"* |
| **`RMSE 1D` / `MAE 1D`** | $\sqrt{\frac{1}{CL}\sum (x - \hat{x})^2}$ | *"Qual a distância média entre a curva real dos sensores e a curva reconstruída pelo Autoencoder na escala $[-1, 1]$?"* |

---

## Módulo 5 — Resultados Globais e Por Bomba (GPU NVIDIA RTX 5080)

### 5.1. Ranking Global das 7 Arquiteturas na Detecção Anual (`StdDev >= 2,5`)
*(Treinamento em 100% do dataset anual limpo, 25 épocas, 5 folds estratificados pelo rótulo de anomalia, somente `GADF`)*

| Pos. | Arquitetura | Modalidade | ROC-AUC Híbrido | PR-AUC | Acurácia Média | F1-Score Médio | Precisão Média | Recall Médio | Falsos Alarmes | ROC-AUC Não-Sup. | Razão Score Não-Sup. |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1º** | **`AE_LSTM1D`** | 1D | **0.9808** | 0.9310 | **94,94%** | 0.8708 | 87,55% | **86,82%** | **5,02%** | 0.8669 | 4,57x |
| **2º** | **`AE_Transformer1D`** | 1D | **0.9795** | 0.9280 | **94,74%** | 0.8660 | 87,01% | 86,36% | 5,33% | 0.8579 | 3,88x |
| **3º** | **`AE_CNN1D`** | 1D | **0.9781** | 0.9279 | **94,57%** | 0.8665 | 87,11% | 86,38% | 5,57% | 0.8639 | 4,68x |
| **4º** | **`Hybrid_Transformer_CNN2D`** | **1D + GADF** | **0.9767** | **0.9366** | **94,66%** | **0.8736** | **88,54%** | 86,30% | 5,30% | 0.8618 | 12,17x |
| **5º** | **`Hybrid_LSTM1D_CNN2D`** | **1D + GADF** | **0.9762** | **0.9341** | **94,37%** | **0.8711** | 87,65% | 86,64% | 5,78% | 0.8669 | 13,19x |
| **6º** | **`Hybrid_CNN1D_CNN2D`** | **1D + GADF** | **0.9742** | 0.9318 | **94,19%** | 0.8683 | 87,16% | 86,59% | 6,56% | **0.8707** | **13,22x** |
| **7º** | **`AE_CNN2D_GAF`** | **2D GADF** | **0.9728** | 0.9277 | **93,93%** | 0.8621 | 86,60% | 86,05% | 6,23% | 0.8352 | 9,43x |

### 5.2. As 3 Grandes Conclusões Científicas da Tabela Acima
1. **Qual é o melhor modelo em `ROC-AUC` e `Acurácia` global?**
   O **`AE_LSTM1D`** (`ROC-AUC = 0.9808`, `Acurácia = 94,94%`, `Falsos Alarmes = 5,02%`), pois com 100% dos dados do ano ele analisa diretamente os passos de $10\text{ segundos}$ sem compressão espacial.
2. **Qual é o melhor modelo em `F1-Score`, `PR-AUC` e `Precisão`?**
   O modelo multimodal **`Hybrid_Transformer_CNN2D`** (`F1 = 0.8736`, `PR-AUC = 0.9366`, `Precisão = 88,54%`), seguido pelo **`Hybrid_LSTM1D_CNN2D`** (`F1 = 0.8711`, `PR-AUC = 0.9341`). A fusão do sinal 1D com a imagem `GADF` reduz os falsos positivos nas bombas mais desbalanceadas.
3. **Qual é o melhor modelo no regime puramente Não Supervisionado (sem rótulos)?**
   O **`Hybrid_CNN1D_CNN2D`** atinge o maior `ROC-AUC Não Supervisionado` (**`0.8707`**) e uma **Razão de Score Anomalia/Normal de `13,22x`** (quase $3\times$ maior que os modelos 1D puros, que ficam em $\approx 4,5\text{x}$).

---

## Módulo 6 — Análise Detalhada das Matrizes de Confusão por Bomba
*(Números exatos de `VN`, `FP`, `FN` e `VP` para explicar em sala/banca)*

Abaixo estão os resultados *Out-of-Fold* (5 folds) da melhor configuração de cada bomba e da janela de `15min`:

### 6.1. Melhores Matrizes de Confusão por Equipamento

| Equipamento | Janela | Arquitetura | Normais Certas (`VN`) | Falsos Alarmes (`FP`) | Anomalias Perdidas (`FN`) | Anomalias Detectadas (`VP`) | Acurácia | `F1-Score` | `ROC-AUC` |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305`** | `30min` | `Hybrid_Transformer_CNN2D` | **25.819** | 368 (`1,41%`) | 291 | **1.625** (`84,81%`) | **97,66%** | **0.8314** | **0.9925** |
| **`Bomba_1307`** | `30min` | `AE_LSTM1D` | **21.297** | 455 (`2,09%`) | 564 | **5.813** (`91,16%`) | **96,38%** | **0.9194** | **0.9934** |
| **`Bomba_2307`** | `30min` | `AE_LSTM1D` | **14.751** | 562 (`3,67%`) | 540 | **12.290** (`95,79%`) | **96,08%** | **0.9571** | **0.9950** |
| **`Bomba_3307`** | `15min` | `AE_LSTM1D` | **34.772** | 720 (`2,03%`) | 1.077 | **20.136** (`94,92%`) | **96,83%** | **0.9573** | **0.9944** |
| **`Bomba_4307`** | `1h` | `Hybrid_LSTM1D_CNN2D` | **11.235** | 261 (`2,27%`) | 379 | **1.932** (`83,60%`) | **95,36%** | **0.8579** | **0.9817** |
| **`Bomba_5307`** | `30min` | `AE_LSTM1D` | **26.080** | 210 (`0,80%`) | 381 | **1.466** (`79,37%`) | **97,90%** | **0.8322** | **0.9895** |
| **`Dataset_Completo`** | `15min` | `AE_Transformer1D` | **16.095** | 1.966 (`10,89%`) | 1.630 | **37.050** (`95,79%`) | **93,66%** | **0.9537** | **0.9834** |
| **`Dataset_Completo`** | `2h` | `AE_LSTM1D` | **949** | 185 (`16,31%`) | 112 | **5.395** (`97,97%`) | **95,53%** | **0.9732** | **0.9784** |

### 6.2. Como Interpretar Esses Números na Apresentação:
1. **Na `Bomba_3307` (a mais crítica da planta, com $21.213$ janelas anômalas de `15min`):**
   * O modelo acertou **34.772 janelas normais (`VN`)** e detectou **20.136 janelas de anomalia (`VP`)**, errando apenas $2,03\%$ de falsos alarmes ($\text{Acurácia} = 96,83\%$, $\text{ROC-AUC} = 0.9944$).
2. **Na `Bomba_1305` e `Bomba_4307` (onde as anomalias são mais esparsas e sutis):**
   * Os modelos **Híbridos com `GADF` (`Hybrid_Transformer_CNN2D` e `Hybrid_LSTM1D_CNN2D`)** superaram todos os modelos 1D puros, alcançando `ROC-AUC = 0.9925` na `Bomba_1305` (`97,66%` de acurácia e apenas `1,41%` de falsos alarmes).
3. **No `Dataset_Completo` ($24$ sensores concatenados das 6 bombas):**
   * Na janela de `15min` ($56.741$ janelas limpas avaliadas), o `AE_Transformer1D` capturou **37.050 de 38.680 janelas anômalas (`Recall = 95,79%`)** com **`Precisão = 94,96%`** e **`F1-Score = 0.9537`**.
   * Na janela de `2h`, o `AE_LSTM1D` atingiu **`Recall = 97,97%`** e **`F1-Score = 0.9732`**.


---

## Módulo 7 — Tabelas Comparativas: Anomalias do Operador vs. Anomalias Detectadas por Cada Modelo

### 7.1. Tabela Geral da Quantidade de Anomalias Apontadas pelo Operador ($\sigma_{\text{5min}} \ge 2,5\text{ A}$)

A tabela abaixo resume o total de ocorrências de anomalia segundo o critério do operador ao longo do ano (`01/01/2026` a `26/10/2026`), tanto em blocos de 5 minutos quanto no conjunto de janelas limpas avaliadas nos 5 folds em cada escala temporal:

| Dataset / Bomba | Blocos Brutos de `5min` ($\sigma \ge 2,5\text{ A}$) | Janelas Anômalas Avaliadas (`15min`) | Janelas Anômalas Avaliadas (`30min`) | Janelas Anômalas Avaliadas (`1h`) | Janelas Anômalas Avaliadas (`2h`) | Soma nas 4 Escalas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305`** | 1.245 (`1,45%`) | **2.135** | **1.916** | **1.671** | **1.378** | **7.100** |
| **`Bomba_1307`** | 10.217 (`11,90%`) | **10.163** | **6.377** | **3.987** | **2.467** | **22.994** |
| **`Bomba_2307`** | 24.987 (`29,11%`) | **23.015** | **12.830** | **6.974** | **3.708** | **46.527** |
| **`Bomba_3307`** | 26.897 (`31,34%`) | **21.213** | **11.689** | **6.502** | **3.609** | **43.013** |
| **`Bomba_4307`** | 4.005 (`4,67%`) | **4.803** | **3.374** | **2.311** | **1.563** | **12.051** |
| **`Bomba_5307`** | 1.730 (`2,02%`) | **2.432** | **1.847** | **1.371** | **993** | **6.643** |
| **Soma 6 Bombas** | **69.081** (`13,42%`) | **63.761** | **38.033** | **22.816** | **13.718** | **138.328** |
| **`Dataset_Completo`** | **69.081** (planta) | **38.680** | **20.670** | **10.845** | **5.507** | **75.702** |
| **Total Avaliado (28 Configs)** | — | **102.441** | **58.703** | **33.661** | **19.225** | **214.030** |

---

### 7.2. Comparação Individual para Cada um dos 7 Modelos (Operador vs. Modelo — Janela de `15min` e Acumulado)

Em cada tabela abaixo:
* **Anomalias do Operador (`VP + FN`):** Quantidade real de janelas anômalas apontadas pelo critério do operador.
* **Detectadas pelo Modelo (`VP`):** Quantas dessas anomalias reais o modelo acertou (e a porcentagem de captura = `Recall`).
* **Não Detectadas (`FN`):** Quantas anomalias reais do operador o modelo deixou passar.
* **Falsos Alarmes (`FP`):** Quantas janelas normais o modelo sinalizou por engano.
* **Total Sinalizado pelo Modelo (`VP + FP`):** Total de alertas disparados pelo modelo.

#### 1. Modelo `AE_CNN1D` (Autoencoder Convolucional 1D)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.419** | 66,46% | 716 | 741 | 2.160 | 97,43% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.151** | 90,04% | 1.012 | 941 | 10.092 | 96,56% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.650** | 94,07% | 1.365 | 3.557 | 25.207 | 91,32% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **20.135** | 94,92% | 1.078 | 737 | 20.872 | 96,80% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.844** | 80,03% | 959 | 769 | 4.613 | 96,95% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.886** | 77,55% | 546 | 495 | 2.381 | 98,16% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **36.929** | 95,47% | 1.751 | 2.073 | 39.002 | 93,26% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.085** | **91,10%** | 5.676 | 7.240 | 65.325 | **96,21%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **198.142** | **92,58%** | 15.888 | 18.220 | 216.362 | **95,37%** |

---

#### 2. Modelo `AE_LSTM1D` (Autoencoder Patch-LSTM Bidirecional 1D)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.379** | 64,59% | 756 | 736 | 2.115 | 97,37% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.044** | 88,99% | 1.119 | 1.069 | 10.113 | 96,14% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.936** | 95,31% | 1.079 | 1.674 | 23.610 | 95,15% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **20.136** | 94,92% | 1.077 | 720 | 20.856 | 96,83% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.844** | 80,03% | 959 | 871 | 4.715 | 96,78% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.799** | 73,97% | 633 | 390 | 2.189 | 98,20% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **36.978** | 95,60% | 1.702 | 1.987 | 38.965 | 93,50% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.138** | **91,18%** | 5.623 | 5.460 | 63.598 | **96,74%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **198.609** | **92,79%** | 15.421 | 15.884 | 214.493 | **95,75%** |

---

#### 3. Modelo `AE_Transformer1D` (Autoencoder Patch-Transformer 1D)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.432** | 67,07% | 703 | 914 | 2.346 | 97,15% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.180** | 90,33% | 983 | 1.094 | 10.274 | 96,34% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.923** | 95,26% | 1.092 | 2.029 | 23.952 | 94,50% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **20.137** | 94,93% | 1.076 | 786 | 20.923 | 96,72% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.698** | 76,99% | 1.105 | 725 | 4.423 | 96,78% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.846** | 75,90% | 586 | 453 | 2.299 | 98,17% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **37.050** | 95,79% | 1.630 | 1.966 | 39.016 | 93,66% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.216** | **91,30%** | 5.545 | 6.001 | 64.217 | **96,61%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **198.086** | **92,55%** | 15.944 | 16.153 | 214.239 | **95,65%** |

---

#### 4. Modelo `AE_CNN2D_GAF` (Autoencoder Convolucional 2D — Somente `GADF`)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.443** | 67,59% | 692 | 588 | 2.031 | 97,74% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.109** | 89,63% | 1.054 | 911 | 10.020 | 96,54% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.828** | 94,84% | 1.187 | 1.583 | 23.411 | 95,12% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **19.899** | 93,81% | 1.314 | 860 | 20.759 | 96,17% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.908** | 81,37% | 895 | 731 | 4.639 | 97,13% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.814** | 74,59% | 618 | 424 | 2.238 | 98,16% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **36.588** | 94,59% | 2.092 | 2.134 | 38.722 | 92,55% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.001** | **90,97%** | 5.760 | 5.097 | 63.098 | **96,81%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **196.580** | **91,85%** | 17.450 | 16.513 | 213.093 | **95,39%** |

---

#### 5. Modelo `Hybrid_CNN1D_CNN2D` (Fusão Multimodal `CNN-1D + CNN-2D GADF`)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.471** | 68,90% | 664 | 682 | 2.153 | 97,63% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.068** | 89,23% | 1.095 | 832 | 9.900 | 96,60% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.734** | 94,43% | 1.281 | 1.395 | 23.129 | 95,28% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **20.114** | 94,82% | 1.099 | 971 | 21.085 | 96,35% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.875** | 80,68% | 928 | 697 | 4.572 | 97,14% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.823** | 74,96% | 609 | 418 | 2.241 | 98,19% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **36.680** | 94,83% | 2.000 | 2.421 | 39.101 | 92,21% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.085** | **91,10%** | 5.676 | **4.995** | 63.080 | **96,86%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **197.524** | **92,29%** | 16.506 | 17.050 | 214.574 | **95,45%** |

---

#### 6. Modelo `Hybrid_LSTM1D_CNN2D` (Fusão Multimodal `Patch-LSTM 1D + CNN-2D GADF`)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.503** | 70,40% | 632 | 722 | 2.225 | 97,61% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.105** | 89,59% | 1.058 | 977 | 10.082 | 96,41% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.733** | 94,43% | 1.282 | 1.435 | 23.168 | 95,21% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **19.861** | 93,63% | 1.352 | 802 | 20.663 | 96,20% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.927** | 81,76% | 876 | 718 | 4.645 | 97,19% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.911** | 78,58% | 521 | 517 | 2.428 | 98,17% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **36.816** | 95,18% | 1.864 | 2.165 | 38.981 | 92,90% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **58.040** | **91,03%** | 5.721 | 5.171 | 63.211 | **96,80%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **197.066** | **92,07%** | 16.964 | 15.969 | 213.035 | **95,53%** |

---

#### 7. Modelo `Hybrid_Transformer_CNN2D` (Fusão Multimodal `Patch-Transformer 1D + CNN-2D GADF`)

| Dataset / Escala | Janelas Normais (`VN + FP`) | Anomalias do Operador (`VP + FN`) | Detectadas pelo Modelo (`VP`) | Taxa de Captura (`Recall`) | Não Detectadas (`FN`) | Falsos Alarmes (`FP`) | Total Sinalizado (`VP + FP`) | Acurácia |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_1305` (`15min`)** | 54.588 | **2.135** | **1.413** | 66,18% | 722 | 555 | 1.968 | 97,75% |
| **`Bomba_1307` (`15min`)** | 46.580 | **10.163** | **9.184** | 90,37% | 979 | 961 | 10.145 | 96,58% |
| **`Bomba_2307` (`15min`)** | 33.704 | **23.015** | **21.688** | 94,23% | 1.327 | 1.660 | 23.348 | 94,73% |
| **`Bomba_3307` (`15min`)** | 35.492 | **21.213** | **19.898** | 93,80% | 1.315 | 760 | 20.658 | 96,34% |
| **`Bomba_4307` (`15min`)** | 51.944 | **4.803** | **3.834** | 79,83% | 969 | 676 | 4.510 | 97,10% |
| **`Bomba_5307` (`15min`)** | 54.295 | **2.432** | **1.862** | 76,56% | 570 | 540 | 2.402 | 98,04% |
| **`Dataset_Completo` (`15min`)** | 18.061 | **38.680** | **37.068** | 95,83% | 1.612 | 2.533 | 39.601 | 92,69% |
| **Soma 6 Bombas (`15min`)** | 276.603 | **63.761** | **57.879** | **90,77%** | 5.882 | 5.152 | 63.031 | **96,76%** |
| **Total Geral (28 Configs)** | 523.334 | **214.030** | **197.611** | **92,33%** | 16.419 | **15.859** | 213.470 | **95,62%** |


---

## Módulo 8 — Anatomia dos Tensores em Cada Modelo: O Que Entra, Como Passa pelo Gargalo e O Que Sai?

> **Fala sugerida para a aula:**
> *"Toda rede neural deste projeto recebe **tensores** (matrizes multidimensionais normalizadas no intervalo $[-1, 1]$). Em um Autoencoder, o tensor de entrada passa por um funil de compressão chamado **Gargalo (Espaço Latente $\mathbf{z}$)**, onde a rede é obrigada a resumir toda a janela em apenas **32 números** `(B, 32)`. A partir desses 32 números, o Decodificador tenta reconstruir exatamente o mesmo tensor da entrada. Como o Autoencoder só aprende a reconstruir janelas **normais**, quando entra uma janela com anomalia, a curva reconstruída na saída diverge da entrada, gerando um erro alto que dispara o alarme."*

### 8.1. Legenda das Dimensões dos Tensores
* **$B$ (*Batch Size* / Lote):** Quantidade de janelas processadas simultaneamente na GPU (ex.: $B = 256$ no treino, ou $B = 1$ ao avaliar uma única janela).
* **$C$ (*Canais / Sensores*):**
  * **$C = 4$** para uma bomba individual (`Corrente YAR`, `Nível YL`, `Carga YLOAD`, `Pressão YP`).
  * **$C = 24$** para o `Dataset_Completo` ($6\text{ bombas} \times 4\text{ sensores}$ concatenados).
* **$L$ (*Comprimento Temporal da Janela*):**
  * $L = 90$ (`15min`), $L = 180$ (`30min`), $L = 360$ (`1h`) ou $L = 720$ (`2h`), sempre com passos de $10\text{ s}$.
* **$32 \times 32$ (*Resolução da Imagem GADF*):** Matriz quadrada de diferenças angulares ($\sin(\phi_i - \phi_j)$) após redução PAA para $32$ segmentos.

---

### 8.2. Passo a Passo de Entrada $\rightarrow$ Gargalo $\rightarrow$ Saída nos 7 Modelos
*(Exemplo numérico concreto para **1 bomba ($C=4$)** na janela de **15 minutos ($L=90$)**)*

#### 1. `AE_CNN1D` (Autoencoder Convolucional 1D)
* **O que entra?** Um **tensor 3D de séries temporais** de formato **`(B, C, L)`** $\rightarrow$ na janela de 15 min, **`(B, 4, 90)`** ($4 \times 90 = 360$ valores numéricos em $[-1, 1]$ por janela).
* **Caminho até o Gargalo (*Encoder*):**
  1. `Conv1d(4 → 16, k=5, s=2)` + `BatchNorm1d` + `GELU` $\rightarrow$ reduz o tempo pela metade: **`(B, 16, 45)`**
  2. `Conv1d(16 → 32, k=5, s=2)` + `BatchNorm1d` + `GELU` $\rightarrow$ reduz novamente: **`(B, 32, 23)`**
  3. `AdaptiveAvgPool1d(8)` + `Flatten` $\rightarrow$ fixa em $32 \times 8$: **`(B, 256)`**
  4. `Linear(256 → 32)` + `GELU` $\rightarrow$ **Gargalo Latente $\mathbf{z}$ de tamanho `(B, 32)`** (comprimiu $360$ números em **$32$ números**!).
* **Caminho do Gargalo até a Saída (*Decoder1D*):**
  1. `Linear(32 → 64)` + `GELU` $\rightarrow$ **`(B, 64)`**
  2. `Linear(64 → 360)` + `Tanh` + `Reshape` $\rightarrow$ **`(B, 4, 90)`**
* **O que é a saída?**
  1. **Saída de Reconstrução (`rec_1d`):** O tensor **`(B, 4, 90)`** com as **4 curvas de sensores reconstruídas** pelo modelo em $[-1, 1]$.
  2. **Saída de Diagnóstico:** O erro quadrático médio $\text{MSE}_{1\text{D}} = \|X_{1\text{D}} - \hat{X}_{1\text{D}}\|^2$ (escalar por janela) e a probabilidade de anomalia $p \in [0, 1]$ da cabeça classificadora.

---

#### 2. `AE_LSTM1D` (Autoencoder Patch-LSTM Bidirecional 1D)
* **O que entra?** O mesmo **tensor 3D temporal `(B, C, L)`** $\rightarrow$ **`(B, 4, 90)`** ($360$ números por janela).
* **Caminho até o Gargalo (*Encoder*):**
  1. **Divisão em 15 Patches (`Patch Split`):** Quebra os $90$ instantes em $15$ blocos de $P = 6$ pontos ($4 \times 6 = 24$ valores por patch) $\rightarrow$ **`(B, 15, 24)`**
  2. `Linear(24 → 32)` + `ReLU` $\rightarrow$ **`(B, 15, 32)`**
  3. `Bi-LSTM(input=32, hidden=32, bidirectional=True)` $\rightarrow$ lê os 15 patches para frente e para trás ($32 + 32 = 64$): **`(B, 15, 64)`**
  4. `Mean Pooling` nos 15 patches $\rightarrow$ **`(B, 64)`**
  5. `Linear(64 → 32)` + `GELU` $\rightarrow$ **Gargalo Latente $\mathbf{z}$ de tamanho `(B, 32)`**.
* **Caminho até a Saída (*Decoder1D*):** `Linear(32 → 64)` + `GELU` $\rightarrow$ `Linear(64 → 360)` + `Tanh` $\rightarrow$ **`(B, 4, 90)`**.
* **O que é a saída?** As **4 séries temporais reconstruídas `(B, 4, 90)`**, o erro de reconstrução $\text{MSE}_{1\text{D}}$ e a probabilidade de anomalia $p \in [0, 1]$.

---

#### 3. `AE_Transformer1D` (Autoencoder Patch-Transformer 1D)
* **O que entra?** O **tensor 3D temporal `(B, C, L)`** $\rightarrow$ **`(B, 4, 90)`**.
* **Caminho até o Gargalo (*Encoder*):**
  1. **Divisão em 15 Patches:** **`(B, 15, 24)`**
  2. `Linear(24 → 32)` + soma do `Positional Embedding` `(1, 15, 32)` $\rightarrow$ **`(B, 15, 32)`**
  3. `TransformerEncoderLayer(d_model=32, nhead=4, dim_ff=64)` $\rightarrow$ aplica atenção entre todos os 15 patches: **`(B, 15, 32)`**
  4. `Flatten(15 × 32)` $\rightarrow$ **`(B, 480)`**
  5. `Linear(480 → 32)` + `GELU` $\rightarrow$ **Gargalo Latente $\mathbf{z}$ de tamanho `(B, 32)`**.
* **Caminho até a Saída (*Decoder1D*):** `Linear(32 → 64 → 360)` + `Tanh` $\rightarrow$ **`(B, 4, 90)`**.
* **O que é a saída?** As **4 séries temporais reconstruídas `(B, 4, 90)`**, o erro $\text{MSE}_{1\text{D}}$ e a probabilidade $p \in [0, 1]$.

---

#### 4. `AE_CNN2D_GAF` (Autoencoder Convolucional 2D — Somente `GADF`)
* **O que entra?** Um **tensor 4D de imagens `GADF`** de formato **`(B, C, 32, 32)`** $\rightarrow$ para 1 bomba ($C=4$), **`(B, 4, 32, 32)`** ($4 \times 1.024 = 4.096$ pixels em $[-1, 1]$ por janela; no `Dataset_Completo`, **`(B, 24, 32, 32)`**).
* **Caminho até o Gargalo (*Encoder2D*):**
  1. `Conv2d(4 → 16, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\rightarrow$ reduz imagem de $32\times 32$ para $16\times 16$: **`(B, 16, 16, 16)`**
  2. `Conv2d(16 → 32, k=3, s=2, p=1)` + `BatchNorm2d` + `GELU` $\rightarrow$ reduz para $8\times 8$: **`(B, 32, 8, 8)`**
  3. `AdaptiveAvgPool2d((4, 4))` + `Flatten` $\rightarrow$ $32 \times 4 \times 4$: **`(B, 512)`**
  4. `Linear(512 → 32)` + `GELU` $\rightarrow$ **Gargalo Latente $\mathbf{z}$ de tamanho `(B, 32)`** (comprimiu $4.096$ pixels em **$32$ números**!).
* **Caminho do Gargalo até a Saída (Dupla Cabeça: *Decoder2D* + *Decoder1D Cruzado*):**
  * **Cabeça 1 (`Decoder2D`):** `Linear(32 → 1024)` $\rightarrow$ `Reshape(B, 16, 8, 8)` $\rightarrow$ `ConvTranspose2d(16 → 16, k=4, s=2)` **`(B, 16, 16, 16)`** $\rightarrow$ `ConvTranspose2d(16 → 4, k=4, s=2)` + `Tanh` $\rightarrow$ **`(B, 4, 32, 32)`**.
  * **Cabeça 2 (`Decoder1D` cruzado):** `Linear(32 → 64 → 360)` + `Tanh` $\rightarrow$ **`(B, 4, 90)`**.
* **O que é a saída?**
  Duas reconstruções simultâneas: **as 4 imagens GADF $32\times 32$ reconstruídas `(B, 4, 32, 32)`** E **as 4 curvas temporais 1D reconstruídas `(B, 4, 90)`**, além do erro combinado ($\text{MSE}_{1\text{D}} + 0,5 \cdot \text{MSE}_{2\text{D}}$) e da probabilidade $p \in [0, 1]$.

---

#### 5, 6 e 7. Modelos Híbridos Multimodais (`Hybrid_CNN1D_CNN2D`, `Hybrid_LSTM1D_CNN2D`, `Hybrid_Transformer_CNN2D`)
* **O que entra?** **Dois tensores simultâneos em paralelo:**
  1. **Ramo 1D:** Tensor temporal **`(B, C, L)`** $\rightarrow$ ex.: **`(B, 4, 90)`** ($360$ valores).
  2. **Ramo 2D:** Tensor de imagens `GADF` **`(B, C, 32, 32)`** $\rightarrow$ ex.: **`(B, 4, 32, 32)`** ($4.096$ pixels).
* **Caminho até o Gargalo (Fusão Multimodal):**
  1. O `Encoder1D` (CNN1D, LSTM1D ou Transformer1D) comprime o sinal 1D em **$\mathbf{z}_1$ de tamanho `(B, 32)`**.
  2. O `EncoderCNN2D` comprime a imagem GADF em **$\mathbf{z}_2$ de tamanho `(B, 32)`**.
  3. **Concatenação:** Junta $[\mathbf{z}_1, \mathbf{z}_2]$ formando um vetor **`(B, 64)`**.
  4. **Camada de Fusão (`Linear(64 → 32)` + `GELU`):** Funde as informações temporal e espacial no **Gargalo Latente Único $\mathbf{z}$ de tamanho `(B, 32)`**.
* **Caminho até a Saída (*Decoder1D* + *Decoder2D*):**
  * A partir do mesmo vetor fundido **`(B, 32)`**, o `Decoder1D` gera **`(B, 4, 90)`** e o `Decoder2D` gera **`(B, 4, 32, 32)`**.
* **O que é a saída?**
  1. **Sinal 1D reconstruído `(B, C, L)`** + **Imagem GADF 2D reconstruída `(B, C, 32, 32)`**;
  2. **Score de erro multimodal** ($\text{MSE}_{1\text{D}} + 0,5 \cdot \text{MSE}_{2\text{D}}$) e **probabilidade de anomalia $p \in [0, 1]$**.

---

### 8.3. Tabela-Resumo: Fluxo de Tensores (Entrada $\rightarrow$ Gargalo $\rightarrow$ Saída)

| Arquitetura | Tensor(es) de Entrada | Dimensão Pré-Gargalo | Gargalo Latente ($\mathbf{z}$) | Tensor(es) de Saída Reconstruídos | O Que a Saída Representa Fisicamente? |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`AE_CNN1D`** | 1D: `(B, C, L)` | `(B, 256)` | **`(B, 32)`** | 1D: **`(B, C, L)`** | Curvas dos sensores reconstruídas + Erro MSE 1D + Probabilidade $p \in [0, 1]$ |
| **`AE_LSTM1D`** | 1D: `(B, C, L)` | `(B, 64)` | **`(B, 32)`** | 1D: **`(B, C, L)`** | Curvas dos sensores reconstruídas + Erro MSE 1D + Probabilidade $p \in [0, 1]$ |
| **`AE_Transformer1D`** | 1D: `(B, C, L)` | `(B, 480)` | **`(B, 32)`** | 1D: **`(B, C, L)`** | Curvas dos sensores reconstruídas + Erro MSE 1D + Probabilidade $p \in [0, 1]$ |
| **`AE_CNN2D_GAF`** | 2D: `(B, C, 32, 32)` | `(B, 512)` | **`(B, 32)`** | 2D: **`(B, C, 32, 32)`** e 1D: **`(B, C, L)`** | Imagens GADF reconstruídas + Curvas 1D cruzadas + Erro 1D/2D + Probabilidade $p \in [0, 1]$ |
| **`Hybrid_CNN1D_CNN2D`** | 1D `(B, C, L)` + 2D `(B, C, 32, 32)` | Concat `(B, 64)` | **`(B, 32)`** | 1D: **`(B, C, L)`** e 2D: **`(B, C, 32, 32)`** | Reconstrução simultânea do sinal 1D e da imagem GADF + Probabilidade $p \in [0, 1]$ |
| **`Hybrid_LSTM1D_CNN2D`** | 1D `(B, C, L)` + 2D `(B, C, 32, 32)` | Concat `(B, 64)` | **`(B, 32)`** | 1D: **`(B, C, L)`** e 2D: **`(B, C, 32, 32)`** | Reconstrução simultânea do sinal 1D e da imagem GADF + Probabilidade $p \in [0, 1]$ |
| **`Hybrid_Transformer_CNN2D`** | 1D `(B, C, L)` + 2D `(B, C, 32, 32)` | Concat `(B, 64)` | **`(B, 32)`** | 1D: **`(B, C, L)`** e 2D: **`(B, C, 32, 32)`** | Reconstrução simultânea do sinal 1D e da imagem GADF + Probabilidade $p \in [0, 1]$ |


---

## Módulo 9 — Análise Comparativa das 11 Formulações de Rótulo (Estado Atual vs. Previsão de Futuro vs. Alerta Precoce vs. Distribuição de Probabilidade)

Para avaliar como a escolha da formulação do rótulo ($y$) altera o poder de diagnóstico e de antecipação das 7 arquiteturas na GPU (**NVIDIA GeForce RTX 5080**), executamos o benchmark completo ([`avaliar_abordagens_rotulos_gpu.py`](file:///c:/Users/202510482/Downloads/Dataset_completo_bombas-20261104T171802Z-1-001/avaliar_abordagens_rotulos_gpu.py)) comparando **11 abordagens de rotulagem** divididas em 4 famílias:

1. **Família 1 — Abordagem Atual (Detecção Binária Instantânea):**
   * `1. Atual Binário (0/1)`: Rótulo $y \in \{0, 1\}$ indicando se houve marcação do operador ($\sigma_{\text{5min}} \ge 2,5\text{ A}$) dentro da própria janela $[t_0, t_1]$.
2. **Família 2 — Previsão Geral de Horizonte Futuro (*Future Horizon Forecasting*):**
   * `2. Futuro Geral (H=15min)`, `3. Futuro Geral (H=30min)` e `4. Futuro Geral (H=60min)`: Usa os sensores do presente $[t_0, t_1]$ para prever se haverá anomalia no intervalo futuro $(t_1, \; t_1 + H]$.
3. **Família 3 — Alerta Precoce Puro (*Early Warning / Precursor Detection*):**
   * `5. Alerta Precoce Puro (H=15min)`, `6. Alerta Precoce Puro (H=30min)` e `7. Alerta Precoce Puro (H=60min)`: Avaliado **exclusivamente nas janelas em que a bomba ainda está Normal no presente ($y_{\text{atual}} = 0$)**, testando se o modelo distingue janelas normais que permanecerão normais daquelas que entrarão em anomalia nos próximos $H$ minutos.
4. **Família 4 — Distribuição Contínua de Probabilidade (% de Anomalia Intra-Janela e Antes/Depois):**
   * `8. % Duração Intra-Janela (τ=0)`: Fração exata de blocos de $5\text{ min}$ anômalos dentro da janela ($0\%$, $33,3\%$, $66,7\%$, $100\%$).
   * `9. Dist. Prob. Curta (τ_antes=15m, τ_depois=7,5m)`: Kernel Gaussiano Assimétrico na vizinhança imediata do ponto do operador.
   * `10. Dist. Prob. Média (τ_antes=30m, τ_depois=15m)`: Rampa precursora moderada de $30\text{ min}$ antes e amortecimento de $15\text{ min}$ depois.
   * `11. Dist. Prob. Longa (τ_antes=60m, τ_depois=30m)`: Rampa precursora longa de $60\text{ min}$ antes e $30\text{ min}$ depois.

---

### 9.1. Tabela Comparativa Global das 11 Abordagens de Rótulo (Média das 7 Arquiteturas × 7 Datasets em 5 Folds)

Fonte: [`resumo_global_11_abordagens_rotulos.csv`](file:///c:/Users/202510482/Downloads/Dataset_completo_bombas-20261104T171802Z-1-001/Dataset_completo_bombas/Resultados_Modelos_1D_2D/Deteccao_Anomalias_Operador/resumo_global_11_abordagens_rotulos.csv)

| # | Abordagem de Rótulo | Família | `ROC-AUC` no Alvo | `PR-AUC` | Acurácia | `F1-Score` | Precisão | `Recall` | Falsos Alarmes (`FAR`) | Erro da % (`MAE`) | Correlação Spearman ($\rho$) | `ROC-AUC` no Evento Real | Captura Precursora $30\text{m}$ Antes (`5% FAR`) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | **`% Duração Intra-Janela (τ=0)`** | **% Contínua** | **0.9908** | 0.8559 | **96,88%** | 0.8046 | 78,64% | 82,86% | **3,01%** | **4,80%** | 0.6236 | **0.9832** | 35,59% |
| **1** | **`Atual Binário (0/1)` *(Atual)*** | **Binário Atual** | **0.9832** | **0.9155** | **96,24%** | **0.8584** | **86,32%** | **85,45%** | 3,46% | 7,64% | 0.6041 | **0.9832** | 34,58% |
| **9** | **`Dist. Prob. Curta (15m / 7,5m)`** | **% Antes/Depois** | **0.9252** | **0.8394** | **92,56%** | **0.7877** | **82,72%** | 75,41% | 5,85% | 14,69% | **0.6530** | **0.9794** | **36,28%** *(1º)* |
| **2** | **`Previsão Geral Futuro (H=15min)`** | **Futuro Geral** | **0.9023** | 0.7055 | **92,20%** | 0.6803 | 68,34% | 68,08% | 6,63% | 21,08% | 0.5245 | 0.9642 | 34,72% |
| **10** | **`Dist. Prob. Média (30m / 15m)`** | **% Antes/Depois** | **0.8992** | **0.8185** | **89,82%** | **0.7604** | **80,12%** | 72,61% | 7,99% | 17,14% | **0.6520** | **0.9771** | **35,87%** *(2º)* |
| **3** | **`Previsão Geral Futuro (H=30min)`** | **Futuro Geral** | **0.8859** | 0.7195 | **89,63%** | 0.6894 | 69,48% | 69,78% | 9,20% | 23,02% | 0.5374 | 0.9596 | 34,08% |
| **11** | **`Dist. Prob. Longa (60m / 30m)`** | **% Antes/Depois** | **0.8732** | **0.8104** | **85,08%** | **0.7428** | **77,13%** | 72,93% | 13,71% | 19,90% | **0.6487** | **0.9743** | 35,26% |
| **4** | **`Previsão Geral Futuro (H=60min)`** | **Futuro Geral** | **0.8695** | 0.7368 | **86,58%** | 0.6954 | 70,73% | 69,92% | 12,58% | 25,28% | 0.5440 | 0.9591 | 35,63% |
| **5** | **`Alerta Precoce Puro (H=15min | Atual=0)`** | **Precursor Puro** | **0.7994** | 0.2879 | **89,60%** | 0.3495 | 29,81% | 43,57% | 7,93% | 32,44% | 0.2516 | 0.9522 | 35,02% |
| **6** | **`Alerta Precoce Puro (H=30min | Atual=0)`** | **Precursor Puro** | **0.7851** | 0.3717 | **85,50%** | 0.4128 | 36,57% | 49,04% | 11,39% | 34,39% | 0.3014 | 0.9548 | 34,41% |
| **7** | **`Alerta Precoce Puro (H=60min | Atual=0)`** | **Precursor Puro** | **0.7758** | 0.4637 | **79,63%** | 0.4768 | 42,81% | **56,83%** | 17,74% | 35,89% | 0.3533 | 0.9512 | 35,15% |

---

### 9.2. Desempenho Preditivo por Bomba (`Futuro 30min`, `Alerta Precoce 30min` e `Distribuição de Probabilidade 30m/15m`)

| Equipamento | Melhor Modelo | `1. Atual Binário` (`ROC-AUC`) | `3. Futuro Geral 30min` (`ROC-AUC`) | `6. Alerta Precoce Puro 30min` (`ROC-AUC`) | `10. Dist. Prob. Média 30m/15m` (`ROC-AUC`) | Captura Precursora $30\text{m}$ Antes (`5% FAR`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bomba_2307`** | `AE_LSTM1D` / `AE_Transformer1D` | **0.9920** | **0.9685** | **0.8910** | **0.9723** | **55,0%** das falhas avisadas $30\text{m}$ antes! |
| **`Bomba_1307`** | `AE_CNN1D` / `AE_LSTM1D` | **0.9910** | **0.9290** | **0.8402** | **0.9302** | **45,6%** das falhas avisadas $30\text{m}$ antes! |
| **`Bomba_5307`** | `Hybrid_Transformer_CNN2D` / `AE_LSTM1D` | **0.9804** | **0.8503** | **0.7856** | **0.8525** | **43,0%** das falhas avisadas $30\text{m}$ antes! |
| **`Bomba_4307`** | `AE_LSTM1D` / `AE_CNN1D` | **0.9855** | **0.8831** | **0.8040** | **0.8877** | **40,6%** das falhas avisadas $30\text{m}$ antes! |
| **`Bomba_3307`** | `AE_Transformer1D` / `AE_LSTM1D` | **0.9939** | **0.9480** | **0.7525** | **0.9398** | **32,3%** das falhas avisadas $30\text{m}$ antes |
| **`Dataset_Completo`** | `Hybrid_LSTM1D_CNN2D` / `AE_Transformer1D` | **0.9837** | **0.9574** | **0.8319** | **0.9696** | **30,3%** das falhas avisadas $30\text{m}$ antes |
| **`Bomba_1305`** | `Hybrid_CNN1D_CNN2D` | **0.9720** | **0.7243** | **0.7014** | **0.7790** | **17,3%** das falhas avisadas $30\text{m}$ antes |

---

### 9.3. Análise Comparativa Crítica: Qual Abordagem Escolher para Cada Objetivo?

1. **Se o objetivo é Diagnosticar com Máxima Precisão se a Bomba Está em Anomalia Agora (`Estado Atual`):**
   * **Vencedoras:** **`8. % Duração Intra-Janela (τ=0)` (`ROC-AUC = 0.9908`, `Acurácia = 96,88%`, `FAR = 3,01%`)** e **`1. Atual Binário (0/1)` (`ROC-AUC = 0.9832`, `F1 = 0.8584`, `PR-AUC = 0.9155`)**.
   * *Descoberta importante:* Trocar o rótulo binário seco (`0/1`) pela **porcentagem de duração intra-janela (`8_Pct_IntraJanela_Pura`)** reduziu os falsos alarmes de `3,46%` para **`3,01%`**, diminuiu o erro de probabilidade (`MAE`) de `7,64%` para **`4,80%`** e elevou o `ROC-AUC` para **`0.9908`**, pois o modelo deixa de receber peso máximo em janelas onde a oscilação durou apenas um terço do tempo.

2. **Se o objetivo é Prever o Futuro e Dar Alerta Gradual de Risco Antes da Falha (`Distribuição de Probabilidade vs. Previsão Binária de Futuro`):**
   * **As abordagens de Distribuição de Probabilidade (`9_Dist_Prob_Curta_15m_7m` e `10_Dist_Prob_Media_30m_15m`) superam claramente a Previsão Binária de Futuro (`2_Futuro_Geral_15min` e `3_Futuro_Geral_30min`) em todas as métricas:**
     * Em $15\text{ min}$: A Distribuição Curta ($\tau_{\text{ant}}=15\text{m}, \tau_{\text{dep}}=7,5\text{m}$) atinge **`ROC-AUC = 0.9252`**, **`PR-AUC = 0.8394`** e **`F1 = 0.7877`**, contra `ROC-AUC = 0.9023`, `PR-AUC = 0.7055` e `F1 = 0.6803` da previsão binária de $15\text{ min}$ (**ganho de $+10,7$ pontos de F1 e $+13,4$ pontos de PR-AUC!**).
     * Em $30\text{ min}$: A Distribuição Média ($\tau_{\text{ant}}=30\text{m}, \tau_{\text{dep}}=15\text{m}$) atinge **`ROC-AUC = 0.8992`**, **`PR-AUC = 0.8185`** e **`F1 = 0.7604`**, superando a previsão binária de $30\text{ min}$ (`F1 = 0.6894`) e mantendo **`ROC-AUC = 0.9771`** na detecção do evento real do operador!
     * Além disso, a **Correlação de Spearman com o gradiente temporal de risco salta de `0.52–0.54` (no binário futuro) para `0.6530` (na distribuição de probabilidade)**, e a **taxa de captura precursora $30\text{ min}$ antes da falha (com apenas $5\%$ de falsos alarmes) atinge o pico global de `36,28%` na média (chegando a `55,0%` na `Bomba_2307` e `45,6%` na `Bomba_1307`)**.

3. **Qual a Largura de Tempo ($\tau_{\text{antes}}, \tau_{\text{depois}}$) Ideal na Distribuição de Probabilidade?**
   * **Largura Curta ($\tau_{\text{ant}}=15\text{m}, \tau_{\text{dep}}=7,5\text{m}$) e Média ($\tau_{\text{ant}}=30\text{m}, \tau_{\text{dep}}=15\text{m}$)** oferecem o melhor compromisso físico: preservam altíssima fidelidade ao evento do operador (`ROC-AUC` de `0.9794` e `0.9771`) enquanto ensinam a rede a elevar suavemente a probabilidade de risco $15\text{ a }30\text{ minutos}$ antes do disparo de corrente.
   * Já a **Largura Longa ($\tau_{\text{ant}}=60\text{m}, \tau_{\text{dep}}=30\text{m}$)** espalha demais o rótulo sobre períodos em que a bomba ainda operava sem precursores físicos visíveis, elevando a taxa de falsos alarmes para `13,71%`.

4. **O que revela o teste de `Alerta Precoce Puro` (`y_atual == 0`)?**
   * Mesmo quando exigimos que o modelo olhe **apenas para janelas 100% normais no presente** (sem nenhuma anomalia já em andamento), as redes neurais conseguem atingir **`ROC-AUC = 0.7994` ($H=15\text{ min}$)** e **`0.7851` ($H=30\text{ min}$)** na média global — chegando a **`ROC-AUC = 0.8910` na `Bomba_2307`**, **`0.8402` na `Bomba_1307`** e **`0.8319` no `Dataset_Completo`**. Isso comprova fisicamente que **existem padrões precursores multivariados nos sensores ($15\text{ a }30\text{ min}$ antes)** que permitem antecipar cerca de metade das anomalias antes mesmo de o desvio padrão da corrente atingir $2,5\text{ A}$!
