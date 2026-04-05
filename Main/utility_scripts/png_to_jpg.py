#!/usr/bin/env python3
"""Convert PNG images to JPG recursively in a directory tree."""

import argparse
import os
from pathlib import Path

from PIL import Image


def convert_png_to_jpg(root_dir: Path, keep_png: bool = False, quality: int = 95) -> int:
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    converted_count = 0

    for png_path in root_dir.rglob('*.png'):
        jpg_path = png_path.with_suffix('.jpg')

        # Skip if target exists and is newer than source
        if jpg_path.exists() and jpg_path.stat().st_mtime >= png_path.stat().st_mtime:
            continue

        try:
            with Image.open(png_path) as im:
                im = im.convert('RGB')
                im.save(jpg_path, 'JPEG', quality=quality)
            converted_count += 1
            print(f"Converted: {png_path} -> {jpg_path}")

            if not keep_png:
                try:
                    png_path.unlink()
                except Exception as e:
                    print(f"Warning: failed to remove {png_path}: {e}")

        except Exception as e:
            print(f"Error converting {png_path}: {e}")

    return converted_count


def main():
    # Hard-coded path and options
    root_dir = Path('/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Map-Data')
    keep_png = False
    quality = 95

    count = convert_png_to_jpg(root_dir, keep_png=keep_png, quality=quality)
    print(f"Done. Converted {count} file(s).")


if __name__ == '__main__':
    main()
