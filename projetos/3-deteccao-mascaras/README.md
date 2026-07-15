# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Weliton Rodrigues

### 1️⃣ Resumo da Abordagem

O treinamento (fine-tuning) foi baseado no modelo pré-treinado **YOLO11n** (`yolo11n.pt`). Os principais hiperparâmetros configurados foram:
* **Épocas:** 20 (adequado para fine-tuning rápido de modelos pré-treinados YOLO em CPU)
* **Tamanho de Imagem (imgsz):** 416 (resolução reduzida para acelerar o treino no CPU sem perder significativamente a capacidade de detecção de rostos maiores)
* **Batch Size:** 8 (limite seguro para evitar estouro de memória no CPU do Windows)
* **Device:** CPU (treinamento local)

O dataset apresenta um forte desbalanceamento de classes, com pouquíssimos exemplos de `mask_weared_incorrect`. Para fins desse desafio de engenharia e pipelines de Edge AI, não foi aplicada técnica extra de balanceamento, mas a diferença de acurácia entre as classes foi devidamente analisada e documentada.

### 2️⃣ Bibliotecas Utilizadas

As principais bibliotecas de desenvolvimento utilizadas foram:
* `ultralytics == 8.4.95` (gerenciamento do ciclo de vida da YOLO)
* `torch == 2.5.1+cpu` e `torchvision == 0.20.1+cpu` (framework principal de Deep Learning)
* `onnx == 1.20.1` (formato intermediário de intercâmbio de modelos)
* `onnx2tf == 2.6.3` (conversor otimizado de ONNX para TensorFlow/TFLite)
* `tensorflow == 2.17.0` (backend para geração do modelo TFLite)

### 3️⃣ Técnica de Otimização do Modelo

A exportação nativa para TFLite do Ultralytics (`format="tflite"`) no Windows possui limitações de plataforma devido à dependência da nova biblioteca `LiteRT` do Google (que restringe compilações PyTorch -> TFLite para Linux e macOS). 

Para otimizar o modelo localmente no Windows:
1. O modelo PyTorch (`model.pt`) foi exportado para o formato **ONNX** (`model.onnx`) com tamanho de imagem 640.
2. Usamos a ferramenta **`onnx2tf`** para ler o arquivo ONNX, compilar as operações e salvá-lo como **TensorFlow Lite** (`model.tflite`) aplicando otimizações de operadores.

### 4️⃣ Resultados Obtidos

Após os treinos e otimizações, os seguintes resultados foram alcançados no conjunto de validação:

* **mAP50 Global:** 0.6887 (68.87%)
* **mAP50-95 Global:** 0.4783 (47.83%)
* **mAP50 por Classe:**
  * `with_mask`: 0.9480 (94.8%)
  * `without_mask`: 0.6870 (68.7%)
  * `mask_weared_incorrect`: 0.4300 (43.0%)
* **Tamanho dos arquivos:**
  * `model.pt`: 5.18 MB
  * `model.tflite`: 10.04 MB (Float32)

### 5️⃣ Comentários Adicionais (Opcional)

1. **Bug do PyTorch 2.8.0 no Windows:** Durante a exportação para ONNX, o exportador legado do PyTorch 2.8.0 (versão de desenvolvimento/nightly instalada) apresentava um erro crítico de `Access Violation` (segfault) no compilador C++ da JIT. Para corrigir esse comportamento e viabilizar a exportação local, atualizamos os pacotes de ambiente para as versões estáveis `torch==2.5.1` e `torchvision==0.20.1`.
2. **Desempenho da Classe Minoritária:** O baixo mAP50 de **43.0%** na classe `mask_weared_incorrect` reflete diretamente a escassez de amostras dessa classe no dataset de treino. Isso confirma a intuição de que o modelo tem dificuldades em generalizar o uso incorreto de máscaras, enquanto apresenta excelente desempenho na classe `with_mask` (**94.8%**).

### 6️⃣ Exemplo de Inferência

Abaixo está o log de saída do terminal ao executar `run_inference.py` no conjunto de imagens de validação selecionadas:

```text
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                         10  [10x with_mask]
maksssksksss107.jpg                          1  [1x with_mask]
maksssksksss11.jpg                          25  [2x mask_weared_incorrect, 22x with_mask, 1x without_mask]
maksssksksss113.jpg                          3  [3x with_mask]
maksssksksss12.jpg                          13  [10x with_mask, 3x without_mask]
----------------------------------------------------------------------
TOTAL                                       52

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
```

Ao abrir as imagens anotadas em `runs/detect/inferencia_exemplos/predicoes/`, foi possível observar que:
* As caixas delimitadoras (bounding boxes) cobrem com precisão rostos em plano médio e close-ups.
* A contagem de pessoas com máscara (`with_mask`) é extremamente precisa em fotos de grupo.
* O modelo detectou corretamente as pessoas sem máscara (`without_mask`) na imagem `maksssksksss12.jpg` e as pessoas com a máscara cobrindo apenas a boca (`mask_weared_incorrect`) na imagem `maksssksksss11.jpg`.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
