"""
Validação automática do Projeto 1 - Classificação MNIST.

Uso: python validate_1_mnist.py <caminho_da_pasta_do_projeto>

Carrega model.h5 e model.tflite JÁ ENTREGUES pelo candidato (não treina nada aqui)
e confere se atendem ao contrato mínimo do desafio.
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
)

N_TEST_SAMPLES = 300
MIN_ACC_H5 = 0.85
MIN_ACC_TFLITE = 0.75


def load_test_subset():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)  # (N, 28, 28, 1)
    return x_test[:N_TEST_SAMPLES], y_test[:N_TEST_SAMPLES]


def main():
    project_dir = sys.argv[1]
    h5_path = f"{project_dir}/model.h5"
    tflite_path = f"{project_dir}/model.tflite"

    x_test, y_test = load_test_subset()

    # --- model.h5 ---
    model = load_keras_model(h5_path)

    expected_shape = (28, 28, 1)
    input_shape = model.input_shape[1:]
    if tuple(input_shape) != expected_shape:
        fail(
            f"model.h5 espera entrada {input_shape}, mas o contrato do Projeto 1 "
            f"exige entrada {expected_shape} (28x28 grayscale normalizado)."
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

    ok("Projeto 1 (Classificação MNIST) validado com sucesso.")


if __name__ == "__main__":
    main()
