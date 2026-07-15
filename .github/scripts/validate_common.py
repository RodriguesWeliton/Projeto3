"""
Funções utilitárias compartilhadas pelos scripts de validação (.github/scripts/validate_*.py).

Estes scripts avaliam os artefatos ENTREGUES pelo candidato (model.h5 e model.tflite
já commitados no repositório) — a CI não treina nada, apenas valida o que foi subido.
"""

import os
import sys

import numpy as np
import tensorflow as tf


def fail(message):
    print(f"::error::{message}")
    sys.exit(1)


def warn(message):
    print(f"::warning::{message}")


def ok(message):
    print(f"✅ {message}")


def load_keras_model(path):
    if not os.path.isfile(path):
        fail(f"Arquivo não encontrado: {path}")
    try:
        model = tf.keras.models.load_model(path)
    except Exception as e:
        fail(f"Não foi possível carregar {path} como modelo Keras. Erro: {e}")
    return model


def load_tflite_interpreter(path):
    if not os.path.isfile(path):
        fail(f"Arquivo não encontrado: {path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
    except Exception as e:
        fail(f"Não foi possível carregar {path} como modelo TFLite. Erro: {e}")
    return interpreter


def check_quantization_size(h5_path, tflite_path):
    """Checagem leve (não bloqueante): tflite deveria, em geral, ser menor que o h5."""
    h5_size = os.path.getsize(h5_path)
    tflite_size = os.path.getsize(tflite_path)
    ok(f"Tamanho model.h5: {h5_size / 1024:.1f} KB | model.tflite: {tflite_size / 1024:.1f} KB")
    if tflite_size >= h5_size:
        warn(
            "model.tflite não ficou menor que model.h5. Verifique se a técnica de "
            "quantização (ex: Dynamic Range Quantization) foi de fato aplicada na "
            "conversão. Isso não bloqueia a validação, mas deve ser comentado no README."
        )


def run_tflite_single_output(interpreter, x):
    """Roda o interpreter TFLite para um modelo com uma única saída (classificação)."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for i in range(len(x)):
        sample = np.expand_dims(x[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        preds.append(interpreter.get_tensor(output_details[0]["index"])[0])
    return np.array(preds)


def run_tflite_multi_output(interpreter, x):
    """
    Roda o interpreter TFLite para um modelo com múltiplas saídas (ex: classificação + bbox).
    Identifica automaticamente qual saída é a de classes (10 valores) e qual é a bbox
    (4 valores), sem depender da ordem em que o conversor TFLite as organizou.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(output_details) != 2:
        fail(
            f"Esperado exatamente 2 saídas no model.tflite (classe + bbox), "
            f"encontrado {len(output_details)}. Confira a arquitetura em train_model.py."
        )

    class_out_idx, bbox_out_idx = None, None
    for od in output_details:
        n = od["shape"][-1]
        if n == 10:
            class_out_idx = od
        elif n == 4:
            bbox_out_idx = od

    if class_out_idx is None or bbox_out_idx is None:
        fail(
            "Não foi possível identificar as saídas de classificação (10 valores) e "
            "bbox (4 valores) no model.tflite. Confira o formato de saída do modelo."
        )

    class_preds, bbox_preds = [], []
    for i in range(len(x)):
        sample = np.expand_dims(x[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        class_preds.append(interpreter.get_tensor(class_out_idx["index"])[0])
        bbox_preds.append(interpreter.get_tensor(bbox_out_idx["index"])[0])

    return np.array(class_preds), np.array(bbox_preds)


def identify_class_and_bbox_outputs(raw_outputs):
    """
    Recebe a saída de model.predict() para um modelo com 2 saídas (classe + bbox),
    que pode vir como lista, tupla ou dict dependendo de como o candidato construiu
    o modelo (Functional API com nomes, Model com múltiplas saídas, etc.), e
    identifica automaticamente qual array é de classificação (10 valores) e qual é
    de bbox (4 valores), sem depender de nomes específicos ou da ordem.
    """
    if isinstance(raw_outputs, dict):
        values = list(raw_outputs.values())
    elif isinstance(raw_outputs, (list, tuple)):
        values = list(raw_outputs)
    else:
        fail(
            "Esperadas 2 saídas (classe + bbox) no model.h5, mas o modelo retornou "
            "um único tensor. Confira se o modelo tem duas heads de saída."
        )

    if len(values) != 2:
        fail(f"Esperadas exatamente 2 saídas no model.h5 (classe + bbox), encontrado {len(values)}.")

    class_out, bbox_out = None, None
    for v in values:
        v = np.array(v)
        if v.shape[-1] == 10:
            class_out = v
        elif v.shape[-1] == 4:
            bbox_out = v

    if class_out is None or bbox_out is None:
        fail(
            "Não foi possível identificar as saídas de classificação (10 valores) e "
            "bbox (4 valores) no model.h5. Confira o formato de saída do modelo."
        )

    return class_out, bbox_out


def iou_batch(boxes_a, boxes_b):
    """IoU entre pares de boxes [xmin, ymin, xmax, ymax] normalizados, shape (N, 4)."""
    xmin = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    ymin = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    xmax = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    ymax = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter_w = np.clip(xmax - xmin, 0, None)
    inter_h = np.clip(ymax - ymin, 0, None)
    inter_area = inter_w * inter_h

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)

    union = area_a + area_b - inter_area
    union = np.where(union <= 0, 1e-6, union)
    return inter_area / union
