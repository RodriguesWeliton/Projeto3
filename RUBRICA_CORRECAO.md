# Rubrica de Correção — Processo Seletivo (Intensivo Maker | AI)

Base: **100 pontos**, distribuídos em 6 categorias. Aplica-se aos três projetos
(Classificação MNIST, Classificação CIFAR-10, Detecção de Máscaras Faciais),
com um **fator de dificuldade** aplicado ao final conforme o projeto escolhido.

---

## 1. Treinamento do Modelo (25 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | Script lança erro ao executar, ou não contém nenhuma implementação de treinamento/fine-tuning. | 0 pts |
| **50%** | Modelo treina, mas falta pelo menos **1** destes itens exigidos: split treino/validação; 3-4 blocos Conv2D+BatchNorm+MaxPooling (P1/P2); Dropout antes da saída (P1/P2); EarlyStopping (P1/P2); data augmentation (P2); uso do `yolo11n.pt` pré-treinado (P3). | 12.5 pts |
| **80%** | **Todos** os itens acima (aplicáveis ao projeto escolhido) presentes e funcionando. | 20 pts |
| **100%** | Atende 80% **e** o relatório justifica pelo menos 1 escolha de hiperparâmetro (épocas, batch size, dropout rate, `imgsz`) com motivo técnico — não apenas "usei esse valor". | 25 pts |

---

## 2. Métrica e Evidência de Resultado (15 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | Nenhuma métrica é calculada nem exibida no terminal. | 0 pts |
| **50%** | Métrica calculada, mas com pelo menos **1** problema: calculada sobre o conjunto de treino (não validação); métrica errada para a tarefa; ou não impressa no terminal. | 7.5 pts |
| **80%** | Métrica correta para a tarefa (**accuracy** em P1/P2, **mAP50** em P3), calculada sobre validação e impressa no terminal. | 12 pts |
| **100%** | Atende 80% **e** o valor numérico está transcrito no relatório (seção 4) **e**, no P3, há comentário sobre desempenho por classe (dado o desbalanceamento do dataset). | 15 pts |

---

## 3. Geração do Modelo Treinado (15 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | Nenhum arquivo de modelo é gerado, ou o salvamento lança erro. | 0 pts |
| **50%** | Arquivo gerado com nome/formato diferente do exigido (ex: `modelo.h5`, `model.keras`, ou `best.pt` não copiado para `model.pt`). | 7.5 pts |
| **80%** | Arquivo gerado com **exatamente** o nome e formato exigidos (`model.h5` em P1/P2, `model.pt` em P3) e carrega sem erro. | 12 pts |
| **100%** | Atende 80% **e** o tamanho real do arquivo (MB/KB) está reportado no relatório (seção 4) — não um placeholder. | 15 pts |

---

## 4. Conversão e Otimização para TFLite (20 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | Script de conversão não implementado, ou `model.tflite` não é gerado. | 0 pts |
| **50%** | `model.tflite` é gerado, mas **sem** técnica de otimização explícita no código, ou o arquivo gerado é maior que o modelo original. | 10 pts |
| **80%** | `model.tflite` gerado com técnica de otimização explicitamente aplicada no código (ex: `tf.lite.Optimize.DEFAULT`), e a técnica está nomeada corretamente na seção 3 do relatório. | 16 pts |
| **100%** | Atende 80% **e** o relatório (seção 4) compara numericamente o tamanho antes/depois (ex: "model.h5: 2.1 MB → model.tflite: 540 KB"). | 20 pts |

---

## 5. Inferência com o Modelo Otimizado (15 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | `run_inference.py` não existe, lança erro ao executar, ou carrega um artefato diferente do `model.tflite` exigido. | 0 pts |
| **50%** | Script executa e carrega o `model.tflite` corretamente, mas roda inferência em **menos de 5 amostras**, ou não imprime o resultado de forma legível no terminal. | 7.5 pts |
| **80%** | Script roda inferência em pelo menos 5 amostras usando o `model.tflite`, com saída clara no terminal (predito vs. real em P1/P2; nº de detecções em P3). | 12 pts |
| **100%** | Atende 80% **e** o relatório (seção 6) traz a saída do terminal colada **e** um comentário específico sobre um caso observado (acerto, erro, ou padrão notado nas imagens/predições geradas). | 15 pts |

---

## 6. Documentação Técnica — README (10 pts)

| Nível | Critério | Pontos |
|---|---|---|
| **0%** | As 6 seções do relatório não foram preenchidas, ou contêm só o texto placeholder original. | 0 pts |
| **50%** | As 6 seções foram preenchidas, mas com respostas genéricas de 1 linha, sem os detalhes pedidos (ex: bibliotecas sem versão; seção 4 sem números reais). | 5 pts |
| **80%** | As 6 seções preenchidas com o conteúdo especificamente pedido: arquitetura com número de blocos/camadas, bibliotecas com versões, técnica nomeada, métrica e tamanhos de arquivo com valores reais. | 8 pts |
| **100%** | Atende 80% **e** a seção 5 (Comentários Adicionais) traz discussão específica sobre uma dificuldade/decisão/limitação real, não frase genérica. | 10 pts |

---

## Fator de Dificuldade

Aplicado sobre a **soma das 6 categorias acima**, para refletir a diferença real
de complexidade de engenharia entre os três projetos (framework, tipo de tarefa,
uso de pré-treino, tempo de treino).

| Projeto escolhido | Fator | Nota máxima possível |
|---|---|---|
| 1 — Classificação MNIST | × 1.00 | 100 pts |
| 2 — Classificação CIFAR-10 | × 1.05 | 105 pts |
| 3 — Detecção de Máscaras Faciais (YOLO) | × 1.15 | 115 pts |

> **Nota final = (soma das 6 categorias) × fator do projeto escolhido**

---

## Observações de Aplicação

- Todos os critérios de 50%/80%/100% são **checklist binário** (item presente
  ou ausente) — não dependem de julgamento subjetivo do avaliador sobre
  "qualidade" ou "organização" do código.
- Em caso de dúvida sobre um item específico do checklist, o avaliador deve
  consultar o README do projeto correspondente (`projetos/N-.../README.md`),
  que define os requisitos técnicos exatos.
- A categoria "Inferência com o Modelo Otimizado" testa especificamente o
  artefato de edge (`model.tflite`) rodando em amostras individuais — é
  deliberadamente separada da categoria de métrica agregada, pois um modelo
  pode ter boa métrica em lote e ainda assim ter comportamento problemático
  em casos individuais (relevante especialmente em datasets desbalanceados).
- A aprovação/reprovação automática da CI (GitHub Actions) é um pré-requisito
  técnico mínimo, mas **não substitui** esta rubrica — a CI confere se os
  artefatos existem, atingem um limiar mínimo de desempenho, e se a inferência
  roda sem erro; a rubrica avalia a qualidade e completude do trabalho entregue.
