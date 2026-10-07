"""Download the pretrained models that DeepOF otherwise fetches at runtime and verify their checksums.

Used when building the container image, so that the image works without access to the download servers.

Usage: python download_models.py <target deepof package directory>
"""

import hashlib
import os
import sys
import urllib.request

# (path relative to the deepof package, sha256, download URLs in order of preference)
MODELS = [
    (
        "trained_models/arena_segmentation/sam_vit_h_4b8939.pth",
        "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        [
            "https://datashare.mpcdf.mpg.de/s/GccLGXXZmw34f8o/download",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        ],
    ),
    (
        "trained_models/deepof_supervised/deepof_supervised_huddle_estimator.pkl",
        "9e2cc9c3e4f0ea9f4295b83caf1c5f171dc36be22f278e23cfcdc0a06c985dd4",
        ["https://datashare.mpcdf.mpg.de/s/kiLpLy1dYNQrPKb/download"],
    ),
]


def download(url, path):
    """Download url to path and return the sha256 of the downloaded file."""
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as file:
        while chunk := response.read(1 << 20):
            sha.update(chunk)
            file.write(chunk)
    return sha.hexdigest()


def main(target):
    for rel_path, expected, urls in MODELS:
        path = os.path.join(target, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for url in urls:
            try:
                digest = download(url, path)
            except OSError as error:
                print(f"{url}: download failed ({error})")
                continue
            if digest == expected:
                print(f"{rel_path}: OK ({url})")
                break
            print(f"{url}: checksum mismatch ({digest})")
        else:
            sys.exit(f"Could not obtain {rel_path} with sha256 {expected}")


if __name__ == "__main__":
    main(sys.argv[1])
