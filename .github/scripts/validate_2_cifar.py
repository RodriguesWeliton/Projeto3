"""
Validação automática do Projeto 2 - Classificação CIFAR-10.

Uso: python validate_2_cifar.py <caminho_da_pasta_do_projeto>
"""

import sys

import numpy as np
import tensorflow as tf

from validate_common import (
    load_keras_model,
    load_tflite_interpreter,
    check_quantization_size,
    run_tflite_single_output,
    fail,
    ok,
    warn,
)

N_TEST_SAMPLES = 300
MIN_ACC_H5 = 0.55
MIN_ACC_TFLITE = 0.45

AUGMENTATION_LAYER_NAMES = (
    "RandomFlip",
    "RandomRotation",
    "RandomZoom",
    "RandomTranslation",
    "RandomContrast",
)


def load_test_subset():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0  # já vem (N, 32, 32, 3)
    y_test = y_test.reshape(-1)  # cifar10 vem como (N,1)
    return x_test[:N_TEST_SAMPLES], y_test[:N_TEST_SAMPLES]


def check_augmentation_present(model):
    layer_types = [type(layer).__name__ for layer in model.layers]
    if not any(name in layer_types for name in AUGMENTATION_LAYER_NAMES):
        warn(
            "Não foi possível identificar camadas de data augmentation "
            "(RandomFlip/RandomRotation/RandomZoom/...) no model.h5. Se a augmentation "
            "foi aplicada de outra forma (ex: tf.data), comente isso no README do projeto. "
            "Esta checagem não bloqueia a validação."
        )
    else:
        ok("Camada(s) de data augmentation detectada(s) no modelo.")


def main():
    project_dir = sys.argv[1]
    h5_path = f"{project_dir}/model.h5"
    tflite_path = f"{project_dir}/model.tflite"

    x_test, y_test = load_test_subset()

    # --- model.h5 ---
    model = load_keras_model(h5_path)
    check_augmentation_present(model)

    expected_shape = (32, 32, 3)
    input_shape = model.input_shape[1:]
    if tuple(input_shape) != expected_shape:
        fail(
            f"model.h5 espera entrada {input_shape}, mas o contrato do Projeto 2 "
            f"exige entrada {expected_shape} (32x32 RGB normalizado)."
        )

    preds = model.predict(x_test, verbose=0)
    acc_h5 = float(np.mean(np.argmax(preds, axis=1) == y_test))
    ok(f"Acurácia model.h5 em {N_TEST_SAMPLES} amostras de teste: {acc_h5:.2%}")
    if acc_h5 < MIN_ACC_H5:
        fail(
            f"Acurácia do model.h5 ({acc_h5:.2%}) abaixo do mínimo esperado "
            f"({MIN_ACC_H5:.0%}). Revise o treinamento em train_model.py."
        )

    # --- model.tflite ---
    check_quantization_size(h5_path, tflite_path)
    interpreter = load_tflite_interpreter(tflite_path)
    tflite_preds = run_tflite_single_output(interpreter, x_test)
    acc_tflite = float(np.mean(np.argmax(tflite_preds, axis=1) == y_test))
    ok(f"Acurácia model.tflite em {N_TEST_SAMPLES} amostras de teste: {acc_tflite:.2%}")
    if acc_tflite < MIN_ACC_TFLITE:
        fail(
            f"Acurácia do model.tflite ({acc_tflite:.2%}) abaixo do mínimo esperado "
            f"({MIN_ACC_TFLITE:.0%}). Revise a conversão em optimize_model.py."
        )

    ok("Projeto 2 (Classificação CIFAR-10) validado com sucesso.")


if __name__ == "__main__":
    main()
