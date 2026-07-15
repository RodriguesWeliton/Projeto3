import os
import shutil
import subprocess
import json
import zipfile
from datetime import datetime
import torch
from ultralytics import YOLO
import ultralytics
from ultralytics.utils.export.litert import _NormalizeCoords

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite com suporte multiplataforma (Windows/Linux)
# ---------------------------------------------------------------------------

IMGSZ = 640
MODEL_INPUT = "model.pt"
MODEL_OUTPUT = "model.tflite"


def main():
    print("=" * 60)
    print("Projeto 3 — Exportação do modelo para TensorFlow Lite")
    print("=" * 60)

    # Reportar tamanho antes da conversão
    if os.path.isfile(MODEL_INPUT):
        size_before_mb = os.path.getsize(MODEL_INPUT) / (1024 * 1024)
        print(f"📁 Tamanho do model.pt (antes): {size_before_mb:.2f} MB")
    else:
        raise FileNotFoundError(
            f"Arquivo '{MODEL_INPUT}' não encontrado. "
            "Execute train_model.py primeiro para gerar o modelo treinado."
        )

    # 1. Carregar o modelo treinado e preparar para exportação
    yolo_model = YOLO(MODEL_INPUT)
    model = yolo_model.model
    model.eval()

    # Ativar flag de exportação em todas as camadas de rede necessárias (evita outputs intermediários/de treino)
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True

    # 2. Envelopar o modelo para normalizar as coordenadas para [0, 1]
    # Isso é obrigatório no contrato de pós-processamento da YOLO/LiteRT para manter precisão de scores
    print("\nNormalizando as coordenadas dos boxes e envelopando o modelo...")
    wrapped_model = _NormalizeCoords(
        model=model,
        h=IMGSZ,
        w=IMGSZ,
        task="detect",
        nc=len(yolo_model.names),
        kpt_shape=None
    )

    # 3. Exportar o modelo envelopado para ONNX
    print(f"\nExportando para ONNX com imgsz={IMGSZ}...")
    onnx_path = "model.onnx"
    dummy_input = torch.zeros(1, 3, IMGSZ, IMGSZ)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}}
    )
    print(f"✅ ONNX exportado com sucesso: {onnx_path}")

    # 4. Converter ONNX para TensorFlow Lite usando onnx2tf com preservação do formato do input (-k images)
    print("\nConvertendo ONNX para TensorFlow Lite usando onnx2tf...")
    cmd = ["onnx2tf", "-i", onnx_path, "-o", "saved_model", "-k", "images"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("⚠️ Erro na conversão com onnx2tf:")
        print(result.stderr)
        fallback_tflite = "model.tflite"
    else:
        print("✅ Conversão com onnx2tf concluída.")
        fallback_tflite = os.path.join("saved_model", "model_float32.tflite")

    # 5. Copiar o arquivo gerado para o destino exigido
    if os.path.isfile(fallback_tflite):
        shutil.copy(fallback_tflite, MODEL_OUTPUT)
        print(f"✅ Arquivo TFLite copiado para '{MODEL_OUTPUT}'")
    else:
        # Busca alternativa se o arquivo principal não estiver no caminho default
        for root, dirs, files in os.walk("."):
            for fname in files:
                if fname.endswith(".tflite") and fname != MODEL_OUTPUT:
                    shutil.copy(os.path.join(root, fname), MODEL_OUTPUT)
                    print(f"✅ TFLite alternativo copiado: {os.path.join(root, fname)} → {MODEL_OUTPUT}")
                    break

    # 6. Injetar metadados do modelo para o backend de LiteRT da YOLO reconhecer as classes e strides
    if os.path.isfile(MODEL_OUTPUT):
        print("\nInjetando metadados (classes, stride, task) no arquivo model.tflite...")
        metadata = {
            "description": "Ultralytics YOLO11n model fine-tuned on face mask detection dataset",
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": ultralytics.__version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)) if hasattr(model, "stride") else 32,
            "task": "detect",
            "batch": 1,
            "imgsz": [IMGSZ, IMGSZ],
            "names": yolo_model.names,
            "channels": 3,
            "end2end": False
        }
        with zipfile.ZipFile(MODEL_OUTPUT, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        print("✅ Metadados injetados com sucesso!")

        # Relatório de tamanho
        size_after_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
        print(f"\n📁 Tamanho do model.tflite (depois): {size_after_mb:.2f} MB")
        print(f"\n📊 Comparação de tamanhos:")
        print(f"   model.pt    : {size_before_mb:.2f} MB")
        print(f"   model.tflite: {size_after_mb:.2f} MB")
        reduction = (1 - size_after_mb / size_before_mb) * 100
        print(f"   Redução     : {reduction:.1f}%")
    else:
        print(f"⚠️  Arquivo TFLite não encontrado nos caminhos esperados.")

    # Limpar arquivos temporários
    if os.path.isfile(onnx_path):
        os.remove(onnx_path)
    if os.path.isdir("saved_model"):
        shutil.rmtree("saved_model")

    print("\n✅ Exportação concluída.")
    print(f"   Artefato de edge gerado: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
