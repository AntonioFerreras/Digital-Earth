"""Uploads the baked tile files to the Hugging Face dataset that `earth download` reads.

Writes <data>/tiles/manifest.zon, the size and SHA-256 of every face file, and
uploads it with the files and dataset_card.md as the dataset's README.md. An
upload of unchanged files transfers nothing.

Log in once first: pixi run -e py hf auth login
Run: pixi run -e py python python/tiles/publish.py --data <root> [--repo <user>/<name>]
"""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi

from bake import LAST_LEVELS

FACE_COUNT = 6
HASH_CHUNK = 1 << 24


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="data root holding tiles/")
    parser.add_argument("--repo", default="AntonioFerreras/digital-earth-tiles", help="dataset repository")
    args = parser.parse_args()

    tiles = args.data / "tiles"
    paths = [f"{layer}/face{face}.tiles" for layer in LAST_LEVELS for face in range(FACE_COUNT)]
    lines = []
    for path in paths:
        print(f"hashing {path}", flush=True)
        lines.append(f'        .{{ .path = "{path}", .size = {(tiles / path).stat().st_size}, '
                     f'.sha256 = "{sha256(tiles / path)}" }},')
    (tiles / "manifest.zon").write_text(".{\n    .files = .{\n" + "\n".join(lines) + "\n    },\n}\n")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=tiles,
                      allow_patterns=paths + ["manifest.zon"], commit_message="Tile files")
    api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=Path(__file__).with_name("dataset_card.md"),
                    path_in_repo="README.md", commit_message="Dataset card")
    print(f"uploaded to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
