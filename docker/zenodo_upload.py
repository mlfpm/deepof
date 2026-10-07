"""Upload a DeepOF container image archive to Zenodo as a DRAFT. Publishing is left to a human.

Environment variables:
    ZENODO_TOKEN            personal access token with the scopes deposit:write and deposit:actions
    ZENODO_URL              default https://zenodo.org (use https://sandbox.zenodo.org for tests)
    ZENODO_DEPOSITION_ID    optional: id of a published record of an earlier image. The upload then becomes a
                            new version of that record (all versions share one concept DOI).

Usage: python zenodo_upload.py <archive.tar.gz> <deepof version> <image reference with digest>
"""

import datetime
import hashlib
import json
import os
import sys
import urllib.request

ZENODO_URL = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
TOKEN = os.environ["ZENODO_TOKEN"]
METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenodo_metadata.json")

DESCRIPTION = """<p>Self-contained container image of <a href="https://gitlab.mpcdf.mpg.de/lucasmir/deepof">DeepOF</a>
{version}, including all dependencies in the exact versions used for this release (CUDA build of PyTorch,
also runs without GPU) and the pretrained models. Running it requires no access to package indices or download
servers.</p>
<p>Image: <code>{image}</code><br>
Archive sha256: <code>{sha256}</code></p>
<p><b>Docker</b> (Linux, Windows with Docker Desktop, macOS):</p>
<pre>docker load -i {archive}
docker run -it --rm -v /path/to/data:/data deepof:{version}
docker run -it --rm --gpus all -v /path/to/data:/data deepof:{version}   # NVIDIA GPU</pre>
<p><b>Apptainer / Singularity</b> (e.g. HPC clusters):</p>
<pre>gunzip {archive}
apptainer build deepof_{version}.sif docker-archive://{archive_tar}
apptainer shell --nv deepof_{version}.sif</pre>
<p>Jupyter: <code>docker run -it --rm -p 8888:8888 -v /path/to/data:/data deepof:{version} jupyter lab --ip=0.0.0.0
--no-browser --allow-root</code></p>
<p>GPU support requires an NVIDIA GPU of the Turing generation or newer (compute capability 7.5+) and a driver
supporting CUDA 12.8. On older GPUs, pass <code>device="cpu"</code> to the training functions.</p>"""


def request(method, url, body=None, data=None, length=None, content_type="application/json"):
    """Send an authenticated request to Zenodo and return the decoded JSON response (or None)."""
    headers = {"Authorization": f"Bearer {TOKEN}"}
    if body is not None:
        data = json.dumps(body).encode()
    if data is not None:
        headers["Content-Type"] = content_type
    if length is not None:
        headers["Content-Length"] = str(length)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=6 * 3600) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {url} failed: {error.code} {error.read().decode(errors='replace')}")
    return json.loads(content) if content else None


def sha256sum(path):
    sha = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 24):
            sha.update(chunk)
    return sha.hexdigest()


def create_draft():
    """Create a new draft, or a new version of ZENODO_DEPOSITION_ID without the files of the old version."""
    previous = os.environ.get("ZENODO_DEPOSITION_ID")
    if not previous:
        return request("POST", f"{ZENODO_URL}/api/deposit/depositions", body={})

    response = request("POST", f"{ZENODO_URL}/api/deposit/depositions/{previous}/actions/newversion")
    draft = request("GET", response["links"].get("latest_draft", response["links"]["self"]))
    for file in request("GET", draft["links"]["files"]) or []:
        request("DELETE", file["links"]["self"])
    return draft


def main(archive, version, image):
    print(f"Computing checksum of {archive} ...")
    checksum = sha256sum(archive)
    name = os.path.basename(archive)

    draft = create_draft()
    print(f"Uploading {name} to draft {draft['id']} ...")
    with open(archive, "rb") as file:
        request("PUT", f"{draft['links']['bucket']}/{name}", data=file, length=os.path.getsize(archive),
                content_type="application/octet-stream")

    with open(METADATA_FILE) as file:
        metadata = json.loads(file.read().replace("{version}", version))
    metadata["version"] = version
    metadata["publication_date"] = datetime.date.today().isoformat()
    metadata["description"] = DESCRIPTION.format(
        version=version, image=image, sha256=checksum, archive=name, archive_tar=name.removesuffix(".gz")
    )
    request("PUT", draft["links"]["self"], body={"metadata": metadata})

    print(f"Draft ready for review (NOT published): {draft['links']['html']}")
    print(f"sha256 {checksum}  {name}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
