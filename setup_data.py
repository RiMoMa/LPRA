#!/usr/bin/env python3
"""Download the COCO val2017 images and annotations used by the pipeline.

The repository commits `data/split.json` (the fixed 150 calibration / 646 evaluation
split) but not the images. Run this once from the repo root:

    python setup_data.py                 # download + extract, keep the zips
    python setup_data.py --remove-zips   # delete the ~1.1 GB of archives afterwards

Result:
    data/coco/val2017/*.jpg                              (5,000 images)
    data/coco/annotations/instances_val2017.json         (20 MB)
"""
import argparse
import pathlib
import shutil
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
COCO = ROOT / "data" / "coco"
URLS = {
    "val2017.zip": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url, dest):
    """Download with a progress line, using a .part file so an interrupted run leaves no corrupt archive."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}  ({done / 1e6:.0f}/{total / 1e6:.0f} MB)", end="", flush=True)
    print()
    tmp.replace(dest)


def extract_images(zip_path, out_dir):
    """Extract every member under val2017/ into data/coco/val2017/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith("val2017/") and not m.is_dir()]
        for i, member in enumerate(members, 1):
            target = ROOT / "data" / "coco" / member.filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 500 == 0 or i == len(members):
                print(f"\r  extracted {i}/{len(members)} images", end="", flush=True)
    print()


def extract_member(zip_path, member_name, target):
    """Extract one member from an annotations archive."""
    if target.exists():
        print(f"already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"extracted {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-zips", action="store_true", help="delete archives after extraction")
    args = parser.parse_args()

    COCO.mkdir(parents=True, exist_ok=True)
    archives = {name: COCO / name for name in URLS}
    for name, dest in archives.items():
        download(URLS[name], dest)

    extract_images(archives["val2017.zip"], COCO / "val2017")
    extract_member(
        archives["annotations_trainval2017.zip"],
        "annotations/instances_val2017.json",
        COCO / "annotations" / "instances_val2017.json",
    )

    n_images = len(list((COCO / "val2017").glob("*.jpg")))
    print(f"val2017 images: {n_images}")
    if n_images != 5000:
        raise SystemExit("expected 5,000 val2017 images; re-run with --remove-zips or check the download")
    if not (ROOT / "data" / "split.json").exists():
        print("data/split.json is missing; regenerate it with: python 04_split.py")

    if args.remove_zips:
        for dest in archives.values():
            dest.unlink(missing_ok=True)
        print("removed archives")


if __name__ == "__main__":
    main()
