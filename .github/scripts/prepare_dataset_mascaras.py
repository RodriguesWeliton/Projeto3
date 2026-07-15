"""
FERRAMENTA DE PREPARAÇÃO DO DATASET — Projeto 3 (Detecção de Máscaras Faciais)

⚠️ Este script NÃO faz parte do desafio do candidato nem é executado pela CI.
É uma ferramenta de uso único, usada pelos mantenedores do repositório para
converter o dataset original do Kaggle (Face Mask Detection, andrewmvd, CC0 1.0)
do formato Pascal VOC (XML) para o formato YOLO, e gerar a estrutura final que
fica em `projetos/3-deteccao-mascaras/dataset/`.

Uso:
    1. Baixe o dataset original do Kaggle:
       https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
       (ou via CLI: kaggle datasets download -d andrewmvd/face-mask-detection)
    2. Extraia o .zip — deve conter duas pastas: "images/" (arquivos .png) e
       "annotations/" (arquivos .xml, um por imagem, formato Pascal VOC)
    3. Rode:
       python prepare_dataset_mascaras.py \
           --images /caminho/para/images \
           --annotations /caminho/para/annotations \
           --output /caminho/para/projetos/3-deteccao-mascaras/dataset \
           --val-split 0.2 --seed 42

O resultado é a estrutura pronta para treino com Ultralytics YOLO:
    dataset/
    ├── images/{train,val}/*.png
    ├── labels/{train,val}/*.txt
    └── data.yaml
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue  # ignora classes desconhecidas, se houver
        class_id = CLASSES.index(name)

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def write_yolo_label(txt_path, boxes):
    with open(txt_path, "w") as f:
        for class_id, xc, yc, w, h in boxes:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Pasta com as imagens originais (.png)")
    parser.add_argument("--annotations", required=True, help="Pasta com as anotações VOC (.xml)")
    parser.add_argument("--output", required=True, help="Pasta de destino (dataset/)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_files = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        raise SystemExit(f"Nenhuma imagem encontrada em {args.images}")

    rng = random.Random(args.seed)
    shuffled = image_files[:]
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * args.val_split)
    val_set = set(shuffled[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)

    n_ok, n_skipped = 0, 0
    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        xml_path = os.path.join(args.annotations, f"{stem}.xml")
        if not os.path.isfile(xml_path):
            n_skipped += 1
            continue

        boxes = parse_voc_annotation(xml_path)
        split = "val" if img_name in val_set else "train"

        shutil.copy2(os.path.join(args.images, img_name), os.path.join(args.output, "images", split, img_name))
        write_yolo_label(os.path.join(args.output, "labels", split, f"{stem}.txt"), boxes)
        n_ok += 1

    data_yaml_path = os.path.join(args.output, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, name in enumerate(CLASSES):
            f.write(f"  {i}: {name}\n")

    print(f"✅ Convertidas {n_ok} imagens ({n_skipped} sem anotação correspondente foram ignoradas).")
    print(f"   Train: {len(image_files) - n_val} | Val: {n_val}")
    print(f"   data.yaml gerado em: {data_yaml_path}")


if __name__ == "__main__":
    main()
