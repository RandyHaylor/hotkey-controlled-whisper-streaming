"""
Download the sherpa-onnx streaming dictation models into <repo>/models/.

Fetches (from k2-fsa GitHub releases) and lays out:
    models/sherpa-zipformer-en-20m/   (encoder/decoder/joiner *.int8.onnx + tokens.txt)
    models/sherpa-online-punct-en/    (model.int8.onnx + bpe.vocab)

These are the smallest/fastest CPU streaming English ASR + an online
punctuation/truecasing model. Only the int8 component files are kept to keep
the bundle small (~50 MB total). They're LFS-tracked via .gitattributes.

Usage:
    python3 download_sherpa_models_to_local_models_directory.py
    python3 download_sherpa_models_to_local_models_directory.py asr
    python3 download_sherpa_models_to_local_models_directory.py punct
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LOCAL_MODELS_PARENT_DIRECTORY = Path(__file__).parent.resolve() / "models"

ASR_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
)
ASR_EXTRACTED_DIR_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
ASR_LOCAL_DIR_NAME = "sherpa-zipformer-en-20m"
ASR_FILES_TO_KEEP = (
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
)

PUNCT_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
    "sherpa-onnx-online-punct-en-2024-08-06.tar.bz2"
)
PUNCT_EXTRACTED_DIR_NAME = "sherpa-onnx-online-punct-en-2024-08-06"
PUNCT_LOCAL_DIR_NAME = "sherpa-online-punct-en"
PUNCT_FILES_TO_KEEP = ("model.int8.onnx", "bpe.vocab")


def _download_extract_and_copy(release_url, extracted_dir_name, local_dir_name,
                               files_to_keep):
    destination_directory = LOCAL_MODELS_PARENT_DIRECTORY / local_dir_name
    destination_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "model.tar.bz2"
        print(f"[sherpa-download] fetching {release_url} ...", flush=True)
        urllib.request.urlretrieve(release_url, archive_path)
        print("[sherpa-download] extracting ...", flush=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(temp_dir)
        extracted_root = Path(temp_dir) / extracted_dir_name
        for filename in files_to_keep:
            source = extracted_root / filename
            if not source.is_file():
                raise SystemExit(f"expected file missing in archive: {source}")
            shutil.copyfile(source, destination_directory / filename)
            print(f"[sherpa-download] -> {destination_directory / filename}", flush=True)


def main():
    requested = sys.argv[1:] or ["asr", "punct"]
    for which in requested:
        if which == "asr":
            _download_extract_and_copy(
                ASR_RELEASE_URL, ASR_EXTRACTED_DIR_NAME, ASR_LOCAL_DIR_NAME,
                ASR_FILES_TO_KEEP,
            )
        elif which == "punct":
            _download_extract_and_copy(
                PUNCT_RELEASE_URL, PUNCT_EXTRACTED_DIR_NAME, PUNCT_LOCAL_DIR_NAME,
                PUNCT_FILES_TO_KEEP,
            )
        else:
            raise SystemExit(f"unknown target '{which}' (use: asr, punct)")
    print("[sherpa-download] done.", flush=True)


if __name__ == "__main__":
    main()
