import os
import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# Hiperparâmetros de fine-tuning
# - epochs=20: número modesto de épocas escolhido porque o YOLO11n já vem
#   pré-treinado no COCO, e em fine-tuning converge rapidamente (geralmente
#   entre 10-20 épocas). Mais épocas aumentariam o tempo sem ganho proporcional
#   em mAP, especialmente na classe minoritária "mask_weared_incorrect".
# - imgsz=416: resolução de entrada reduzida (padrão seria 640). Escolhida
#   para reduzir o tempo de treino em CPU — o YOLO11n nano foi projetado para
#   ser eficiente em resoluções menores, mantendo boa acurácia.
# - batch=8: batch size compatível com uso de CPU (valores maiores causam
#   consumo excessivo de memória RAM).
EPOCHS = 20
IMGSZ = 416
BATCH = 8

# Caminho para o dataset (relativo à pasta do projeto)
DATA_YAML = "dataset/data.yaml"
MODEL_OUTPUT = "model.pt"


def main():
    print("=" * 60)
    print("Projeto 3 — Fine-tuning do YOLO11n para Detecção de Máscaras")
    print("=" * 60)
    print(f"  Epochs   : {EPOCHS}")
    print(f"  imgsz    : {IMGSZ}")
    print(f"  Batch    : {BATCH}")
    print(f"  Device   : cpu")
    print(f"  Dataset  : {DATA_YAML}")
    print("=" * 60)

    # 1. Verificar se existe checkpoint para resumir
    last_checkpoint = "runs/detect/train/weights/last.pt"
    if os.path.exists(last_checkpoint):
        print(f"🔄 Checkpoint encontrado em {last_checkpoint}. Retomando treinamento...")
        model = YOLO(last_checkpoint)
        results = model.train(resume=True)
        from pathlib import Path
        save_dir = Path("runs/detect/train")
    else:
        # Carregar o modelo pré-treinado YOLO11n
        model = YOLO("yolo11n.pt")
        # Fine-tuning no dataset de máscaras faciais
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device="cpu",
            verbose=True,
        )
        save_dir = results.save_dir

    # 3. Copiar os melhores pesos para model.pt (nome exigido)
    best_pt = save_dir / "weights" / "best.pt"
    shutil.copy(best_pt, MODEL_OUTPUT)
    print(f"\n✅ Modelo treinado salvo em: {MODEL_OUTPUT}")

    # 4. Calcular e imprimir a métrica no conjunto de validação
    print("\n--- Avaliação no Conjunto de Validação ---")
    val_model = YOLO(MODEL_OUTPUT)
    metrics = val_model.val(data=DATA_YAML, split="val", verbose=True)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS (Conjunto de Validação):")
    print(f"  mAP50      : {map50:.4f}")
    print(f"  mAP50-95   : {map50_95:.4f}")
    print("")
    print("  mAP50 por classe:")
    class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]
    for i, name in enumerate(class_names):
        try:
            ap_per_class = metrics.box.ap[:, 0]  # AP at IoU=0.5
            print(f"    {name}: {float(ap_per_class[i]):.4f}")
        except Exception:
            pass
    print("=" * 60)
    print("\n⚠️  Nota sobre desbalanceamento de classes:")
    print("   A classe 'mask_weared_incorrect' é minoritária no dataset.")
    print("   Espera-se desempenho inferior nessa classe comparado às demais.")

    # 5. Reportar tamanho do arquivo gerado
    if os.path.isfile(MODEL_OUTPUT):
        size_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
        print(f"\n📁 Tamanho do model.pt: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
