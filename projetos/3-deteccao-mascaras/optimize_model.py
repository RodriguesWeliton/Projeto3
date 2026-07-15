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
#   2. Exportar para TensorFlow Lite de forma adaptativa (Windows/Linux CI)
# ---------------------------------------------------------------------------

IMGSZ = 640
MODEL_INPUT = "model.pt"
MODEL_OUTPUT = "model.tflite"


class CustomNormalizeCoords(torch.nn.Module):
    """Envelopa o modelo PyTorch da YOLO para que a saída de detecção
    tenha as coordenadas dos boxes normalizadas no intervalo [0, 1].
    Isso é exigido pelo pós-processamento da YOLO/LiteRT backend.
    """
    def __init__(self, model, h, w, task, nc):
        super().__init__()
        self._model = model
        self.h = h
        self.w = w
        self.task = task
        self.nc = nc

    def forward(self, x):
        det = self._model(x)
        # det shape: (batch, 4 + nc, anchors)
        box_wh = torch.tensor([self.w, self.h, self.w, self.h], dtype=det.dtype, device=det.device).view(1, 4, 1)
        parts = [det[:, :4] / box_wh]  # normaliza as coordenadas xywh por largura/altura
        parts.append(det[:, 4:])       # mantém scores de classes e outros coeficientes inalterados
        return torch.cat(parts, dim=1)

    def fuse(self):
        """Executa a fusão do modelo interno e envelopa o resultado
        em uma nova instância de CustomNormalizeCoords para manter a
        normalização ativa durante todo o processo de exportação.
        """
        fused_inner = self._model.fuse()
        return CustomNormalizeCoords(fused_inner, self.h, self.w, self.task, self.nc)

    def __getattr__(self, name):
        """Redireciona buscas de atributos para o modelo interno
        para que o exportador da Ultralytics consiga ler stride, names, etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


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
    yolo_model = YOLO(MODEL_INPUT)
    model = yolo_model.model
    model.eval()

    # Ativar flag de exportação em todas as camadas de rede necessárias (evita outputs intermediários/de treino)
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True

    # 2. Verificar se o onnx2tf está instalado no PATH do sistema
    onnx2tf_exists = shutil.which("onnx2tf") is not None

    if onnx2tf_exists:
        print("\n[Ambiente Local] onnx2tf encontrado. Usando exportador ONNX customizado...")
        # Envelopar o modelo para normalizar as coordenadas para [0, 1]
        print("Normalizando as coordenadas dos boxes e envelopando o modelo...")
        wrapped_model = CustomNormalizeCoords(
            model=model,
            h=IMGSZ,
            w=IMGSZ,
            task="detect",
            nc=len(yolo_model.names)
        )

        # Substitui o modelo interno do YOLO pelo nosso modelo envelopado
        yolo_model.model = wrapped_model

        # Exportar o modelo envelopado para ONNX usando o próprio exportador da Ultralytics
        print(f"Exportando para ONNX com imgsz={IMGSZ}...")
        onnx_path = yolo_model.export(format="onnx", imgsz=IMGSZ, simplify=False)
        print(f"✅ ONNX exportado com sucesso: {onnx_path}")

        # Converter ONNX para TensorFlow Lite usando onnx2tf com preservação do formato do input (-k images)
        print("Convertendo ONNX para TensorFlow Lite usando onnx2tf...")
        cmd = ["onnx2tf", "-i", onnx_path, "-o", "saved_model", "-k", "images"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("⚠️ Erro na conversão com onnx2tf:")
            print(result.stderr)
            fallback_tflite = "model.tflite"
        else:
            print("✅ Conversão com onnx2tf concluída.")
            fallback_tflite = os.path.join("saved_model", "model_float32.tflite")

        # Copiar o arquivo gerado para o destino exigido
        if os.path.isfile(fallback_tflite):
            shutil.copy(fallback_tflite, MODEL_OUTPUT)
            print(f"✅ Arquivo TFLite copiado para '{MODEL_OUTPUT}'")
        else:
            for root, dirs, files in os.walk("."):
                for fname in files:
                    if fname.endswith(".tflite") and fname != MODEL_OUTPUT:
                        shutil.copy(os.path.join(root, fname), MODEL_OUTPUT)
                        print(f"✅ TFLite alternativo copiado: {os.path.join(root, fname)} → {MODEL_OUTPUT}")
                        break

        # Injetar metadados do modelo para o backend de LiteRT da YOLO reconhecer as classes e strides
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

        # Limpar arquivos temporários
        if os.path.isfile(onnx_path):
            os.remove(onnx_path)
        if os.path.isdir("saved_model"):
            shutil.rmtree("saved_model")

    else:
        print("\n[Ambiente CI] onnx2tf não encontrado. Usando exportador nativo da Ultralytics...")
        # No Linux do CI, o exportador LiteRT nativo funciona perfeitamente, aplicando normalização e metadados.
        # Por segurança, mantemos o yolo_model original sem envelopamento (pois o exportador nativo cuida do envelopamento).
        yolo_model.export(format="tflite", imgsz=IMGSZ)

        # O exportador da YOLO cria o modelo como model.tflite ou dentro de uma pasta model_saved_model/model.tflite
        tflite_candidates = [
            MODEL_OUTPUT,
            MODEL_INPUT.replace(".pt", ".tflite"),
            MODEL_INPUT.replace(".pt", "_saved_model") + "/model.tflite",
        ]
        tflite_found = None
        for candidate in tflite_candidates:
            if os.path.isfile(candidate):
                tflite_found = candidate
                break
        if tflite_found is None:
            for root, dirs, files in os.walk("."):
                for fname in files:
                    if fname.endswith(".tflite") and fname != MODEL_OUTPUT:
                        tflite_found = os.path.join(root, fname)
                        break
                if tflite_found:
                    break

        if tflite_found and tflite_found != MODEL_OUTPUT:
            shutil.copy(tflite_found, MODEL_OUTPUT)
            print(f"✅ Arquivo TFLite copiado de '{tflite_found}' → '{MODEL_OUTPUT}'")

    # Relatório de tamanho final
    if os.path.isfile(MODEL_OUTPUT):
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
