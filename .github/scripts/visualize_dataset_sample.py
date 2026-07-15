"""
FERRAMENTA DE VERIFICAÇÃO — Projeto 3 (Detecção de Máscaras Faciais)

⚠️ Este script NÃO faz parte do desafio do candidato nem é executado pela CI.
Desenha as bounding boxes (a partir dos arquivos .txt em formato YOLO) sobre uma
amostra aleatória de imagens do dataset, para conferência visual de que os
labels continuam corretamente alinhados às imagens (por exemplo, depois de um
redimensionamento/recompressão).

Uso (a partir da raiz do repositório):
    python .github/scripts/visualize_dataset_sample.py \
        --dataset projetos/3-deteccao-mascaras/dataset \
        --split train --n 12 --output amostra_verificacao --seed 42

Abra as imagens geradas em "amostra_verificacao/" e confira visualmente se as
caixas caem em cima dos rostos corretos.
"""

import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
CLASS_COLORS = {0: (0, 200, 0), 1: (220, 0, 0), 2: (230, 160, 0)}  # verde / vermelho / amarelo


def read_yolo_labels(label_path):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append((class_id, xc, yc, w, h))
    return boxes


def draw_boxes(image_path, label_path, output_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    boxes = read_yolo_labels(label_path)
    for class_id, xc, yc, w, h in boxes:
        # desnormaliza: coordenadas do YOLO sao relativas ao tamanho ATUAL da imagem,
        # entao isso funciona corretamente independente de redimensionamentos previos
        xmin = (xc - w / 2) * W
        ymin = (yc - h / 2) * H
        xmax = (xc + w / 2) * W
        ymax = (yc + h / 2) * H

        color = CLASS_COLORS.get(class_id, (0, 120, 255))
        label = CLASS_NAMES.get(class_id, f"classe_{class_id}")

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.rectangle([xmin, max(0, ymin - 16), xmin + 8 * len(label), ymin], fill=color)
        draw.text((xmin + 2, max(0, ymin - 15)), label, fill=(255, 255, 255))

    img.save(output_path, "JPEG", quality=92)
    return len(boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--n", type=int, default=12, help="Quantidade de imagens a amostrar (por split)")
    parser.add_argument("--output", default="amostra_verificacao")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = random.Random(args.seed)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    total_imgs, total_boxes, total_sem_label = 0, 0, 0

    for split in splits:
        img_dir = os.path.join(args.dataset, "images", split)
        label_dir = os.path.join(args.dataset, "labels", split)
        if not os.path.isdir(img_dir):
            print(f"⚠️ Pasta não encontrada, pulando: {img_dir}")
            continue

        all_imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        sample = rng.sample(all_imgs, min(args.n, len(all_imgs)))

        for fname in sample:
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            label_path = os.path.join(label_dir, f"{stem}.txt")
            out_path = os.path.join(args.output, f"{split}_{fname.rsplit('.',1)[0]}.jpg")

            n_boxes = draw_boxes(img_path, label_path, out_path)
            total_imgs += 1
            total_boxes += n_boxes
            if n_boxes == 0:
                total_sem_label += 1
                print(f"  ⚠️ Sem bounding box encontrada: {split}/{fname}")

    print(f"\n✅ {total_imgs} imagens processadas, {total_boxes} bounding boxes desenhadas.")
    if total_sem_label:
        print(f"⚠️ {total_sem_label} imagens da amostra não tinham nenhuma box — vale conferir manualmente.")
    print(f"Confira visualmente as imagens salvas em: {args.output}/")


if __name__ == "__main__":
    main()
