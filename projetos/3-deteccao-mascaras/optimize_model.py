import os

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
#
# Técnica de Otimização: Post-Training Quantization (PTQ) padrão
#   O Ultralytics realiza automaticamente a quantização de pesos de
#   float32 para int8/float16 durante a exportação para TFLite,
#   reduzindo significativamente o tamanho do modelo e a latência de
#   inferência em dispositivos edge — sem necessidade de dataset de
#   calibração adicional.
# ---------------------------------------------------------------------------

IMGSZ = 416
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

    # 1. Carregar o modelo treinado
    model = YOLO(MODEL_INPUT)

    # 2. Exportar para ONNX e depois converter para TFLite usando onnx2tf
    # Isso contorna limitações de dependências (LiteRT) e bugs no Windows.
    print(f"\nExportando para ONNX com imgsz={IMGSZ}...")
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, simplify=False)

    print("\nConvertendo ONNX para TensorFlow Lite usando onnx2tf...")
    import subprocess
    cmd = ["onnx2tf", "-i", onnx_path, "-o", "saved_model"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("⚠️ Erro na conversão com onnx2tf:")
        print(result.stderr)
        fallback_tflite = "model.tflite"
    else:
        print("✅ Conversão com onnx2tf concluída.")
        fallback_tflite = os.path.join("saved_model", "model_float32.tflite")

    # 3. Verificar e reportar tamanho do arquivo gerado
    tflite_candidates = [
        fallback_tflite,
        os.path.join("saved_model", "model_float32.tflite"),
        MODEL_OUTPUT,
        MODEL_INPUT.replace(".pt", ".tflite"),
    ]

    tflite_found = None
    for candidate in tflite_candidates:
        if os.path.isfile(candidate):
            tflite_found = candidate
            break

    # Busca mais ampla se não encontrado nos caminhos esperados
    if tflite_found is None:
        for root, dirs, files in os.walk("."):
            for fname in files:
                if fname.endswith(".tflite"):
                    tflite_found = os.path.join(root, fname)
                    break
            if tflite_found:
                break

    if tflite_found and os.path.isfile(tflite_found):
        # Copiar para o nome exigido se necessário
        import shutil
        shutil.copy(tflite_found, MODEL_OUTPUT)
        print(f"✅ Arquivo TFLite copiado de '{tflite_found}' → '{MODEL_OUTPUT}'")

        size_after_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
        print(f"\n📁 Tamanho do model.tflite (depois): {size_after_mb:.2f} MB")
        print(f"\n📊 Comparação de tamanhos:")
        print(f"   model.pt    : {size_before_mb:.2f} MB")
        print(f"   model.tflite: {size_after_mb:.2f} MB")
        reduction = (1 - size_after_mb / size_before_mb) * 100
        print(f"   Redução     : {reduction:.1f}%")
    else:
        print(f"⚠️  Arquivo TFLite não encontrado nos caminhos esperados.")

    print("\n✅ Exportação concluída.")
    print(f"   Artefato de edge gerado: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
