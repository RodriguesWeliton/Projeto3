"""
Validação automática do Projeto 3 - Detecção de Máscaras Faciais (YOLO).

Uso: python validate_3_mascaras.py <caminho_da_pasta_do_projeto>

Usa a própria API do Ultralytics (model.val()) para avaliar mAP50 no conjunto
de validação do dataset já presente no repositório (projetos/.../dataset/).
Não treina nada aqui — apenas valida os artefatos ENTREGUES (model.pt e
model.tflite) pelo candidato.
"""

import os
import sys

from ultralytics import YOLO

MIN_MAP50_PT = 0.30
MIN_MAP50_TFLITE = 0.20


def fail(message):
    print(f"::error::{message}")
    sys.exit(1)


def warn(message):
    print(f"::warning::{message}")


def ok(message):
    print(f"✅ {message}")


def main():
    project_dir = sys.argv[1]
    pt_path = f"{project_dir}/model.pt"
    tflite_path = f"{project_dir}/model.tflite"
    data_yaml = f"{project_dir}/dataset/data.yaml"

    if not os.path.isfile(data_yaml):
        fail(f"Arquivo de dataset não encontrado: {data_yaml}. A pasta dataset/ não deveria ter sido alterada.")

    # --- model.pt ---
    if not os.path.isfile(pt_path):
        fail(f"Arquivo não encontrado: {pt_path}")

    model = YOLO(pt_path)
    metrics = model.val(data=data_yaml, split="val", verbose=False)
    map50_pt = float(metrics.box.map50)
    ok(f"model.pt -> mAP50 no conjunto de validação: {map50_pt:.3f}")

    if map50_pt < MIN_MAP50_PT:
        fail(
            f"mAP50 do model.pt ({map50_pt:.3f}) abaixo do mínimo esperado "
            f"({MIN_MAP50_PT:.2f}). Revise o fine-tuning em train_model.py."
        )

    # --- model.tflite ---
    if not os.path.isfile(tflite_path):
        fail(f"Arquivo não encontrado: {tflite_path}")

    pt_size = os.path.getsize(pt_path)
    tflite_size = os.path.getsize(tflite_path)
    ok(f"Tamanho model.pt: {pt_size / 1024:.1f} KB | model.tflite: {tflite_size / 1024:.1f} KB")

    model_tflite = YOLO(tflite_path, task="detect")
    metrics_tflite = model_tflite.val(data=data_yaml, split="val", verbose=False)
    map50_tflite = float(metrics_tflite.box.map50)
    ok(f"model.tflite -> mAP50 no conjunto de validação: {map50_tflite:.3f}")

    if map50_tflite < MIN_MAP50_TFLITE:
        fail(
            f"mAP50 do model.tflite ({map50_tflite:.3f}) abaixo do mínimo esperado "
            f"({MIN_MAP50_TFLITE:.2f}). Revise a exportação em optimize_model.py."
        )

    ok("Projeto 3 (Detecção de Máscaras Faciais) validado com sucesso.")


if __name__ == "__main__":
    main()
