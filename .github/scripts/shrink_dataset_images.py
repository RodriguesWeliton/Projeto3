"""
FERRAMENTA DE OTIMIZAÇÃO DO DATASET — Projeto 3 (Detecção de Máscaras Faciais)

⚠️ Este script NÃO faz parte do desafio do candidato nem é executado pela CI.
É uma ferramenta de uso único para reduzir o tamanho do dataset já convertido
em projetos/3-deteccao-mascaras/dataset/ (imagens .png grandes → .jpg menores).

Como os rótulos YOLO (.txt) usam coordenadas NORMALIZADAS (0-1, relativas ao
tamanho da imagem), redimensionar a imagem NÃO exige alterar os arquivos de
label — só as imagens são tocadas.

Uso (a partir da raiz do repositório):
    python .github/scripts/shrink_dataset_images.py \
        --dataset projetos/3-deteccao-mascaras/dataset \
        --max-dim 800 --quality 90
"""

import argparse
import os

from PIL import Image


def shrink_folder(folder, max_dim, quality):
    n_ok, before_total, after_total = 0, 0, 0

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        src_path = os.path.join(folder, fname)
        before_total += os.path.getsize(src_path)

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = min(1.0, max_dim / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            stem = os.path.splitext(fname)[0]
            dst_path = os.path.join(folder, f"{stem}.jpg")
            img.save(dst_path, "JPEG", quality=quality, optimize=True)

        if src_path != dst_path and os.path.isfile(src_path):
            os.remove(src_path)

        after_total += os.path.getsize(dst_path)
        n_ok += 1

    return n_ok, before_total, after_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Pasta dataset/ (contendo images/train, images/val)")
    parser.add_argument("--max-dim", type=int, default=800, help="Maior dimensão (px) após redimensionar")
    parser.add_argument("--quality", type=int, default=90, help="Qualidade JPEG (1-95)")
    args = parser.parse_args()

    total_ok, total_before, total_after = 0, 0, 0
    for split in ("train", "val"):
        folder = os.path.join(args.dataset, "images", split)
        if not os.path.isdir(folder):
            continue
        n_ok, before, after = shrink_folder(folder, args.max_dim, args.quality)
        total_ok += n_ok
        total_before += before
        total_after += after
        print(f"  {split}: {n_ok} imagens | {before/1024/1024:.1f} MB -> {after/1024/1024:.1f} MB")

    print(f"\n✅ Total: {total_ok} imagens processadas")
    print(f"   Antes:  {total_before/1024/1024:.1f} MB")
    print(f"   Depois: {total_after/1024/1024:.1f} MB")
    if total_before > 0:
        print(f"   Redução: {(1 - total_after/total_before)*100:.0f}%")


if __name__ == "__main__":
    main()
