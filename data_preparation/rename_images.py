"""
Rename all images in a folder to 1.jpg, 2.jpg, 3.jpg ... (extension is kept)

Usage:
    python rename_images.py <folder>          # rename images in this folder only
    python rename_images.py <folder> -r       # also rename inside every subfolder (each folder starts from 1)
    python rename_images.py <folder> --dry    # preview only, nothing is renamed
"""

import os
import re
import sys
import uuid

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}


def natural_key(name):
    """Sort 'img2' before 'img10'"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def rename_folder(folder, dry_run=False):
    images = sorted(
        (f for f in os.listdir(folder)
         if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTS),
        key=natural_key
    )
    if not images:
        return 0

    plan = [(f, f"{i}{os.path.splitext(f)[1].lower()}") for i, f in enumerate(images, start=1)]

    if dry_run:
        for old, new in plan:
            print(f"  {old} -> {new}")
        return len(plan)

    # Two passes so that existing names like "2.jpg" are never overwritten
    temp = []
    for old, new in plan:
        tmp_name = f"__tmp_{uuid.uuid4().hex}{os.path.splitext(old)[1]}"
        os.rename(os.path.join(folder, old), os.path.join(folder, tmp_name))
        temp.append((tmp_name, new))
    for tmp_name, new in temp:
        os.rename(os.path.join(folder, tmp_name), os.path.join(folder, new))

    return len(plan)


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith('-'):
        print(__doc__)
        return

    root = args[0]
    recursive = '-r' in args
    dry_run = '--dry' in args

    if not os.path.isdir(root):
        print(f"Error: folder not found: {root}")
        return

    folders = [d for d, _, _ in os.walk(root)] if recursive else [root]
    total = 0
    for folder in folders:
        count = rename_folder(folder, dry_run)
        if count:
            print(f"{'[preview] ' if dry_run else ''}{folder}: {count} images")
            total += count

    print(f"Done. {total} images {'would be ' if dry_run else ''}renamed.")


if __name__ == "__main__":
    main()
