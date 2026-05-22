"""
Download the sherpa-onnx streaming dictation models into <repo>/models/.

Fetches (from k2-fsa GitHub releases) and lays out one local directory per
model. Only the int8 component files (+ tokens) are kept to keep the bundle
small. The .onnx weights are LFS-tracked via .gitattributes.

Available targets (each maps to one local models/ subdirectory):

    asr            sherpa-zipformer-en-20m            (~50 MB, smallest/fastest)
    asr-2023-06-26 sherpa-zipformer-en-2023-06-26     (~70 MB, accurate tier)
    asr-2023-06-21 sherpa-zipformer-en-2023-06-21     (~180 MB, LibriSpeech+GigaSpeech)
    punct          sherpa-online-punct-en             (punctuation + truecasing companion)

Usage:
    python3 download_sherpa_models_to_local_models_directory.py            # default bundle (asr + punct)
    python3 download_sherpa_models_to_local_models_directory.py all        # every target above
    python3 download_sherpa_models_to_local_models_directory.py asr-2023-06-26 punct
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LOCAL_MODELS_PARENT_DIRECTORY = Path(__file__).parent.resolve() / "models"

K2_RELEASE_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


class SherpaModelDownloadSpec:
    """One downloadable model: its release tarball and the files to keep."""

    def __init__(self, release_url, extracted_dir_name, local_dir_name,
                 files_to_keep):
        self.release_url = release_url
        self.extracted_dir_name = extracted_dir_name
        self.local_dir_name = local_dir_name
        self.files_to_keep = files_to_keep


# target name -> spec. Each ASR model ships DIFFERENT component filenames
# (epoch/avg/chunk vary by model), so the kept-file list is per-model.
SHERPA_MODEL_DOWNLOAD_SPECS_BY_TARGET = {
    "asr": SherpaModelDownloadSpec(
        release_url=f"{K2_RELEASE_BASE_URL}/asr-models/"
                    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        extracted_dir_name="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        local_dir_name="sherpa-zipformer-en-20m",
        files_to_keep=(
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ),
    ),
    "asr-2023-06-26": SherpaModelDownloadSpec(
        release_url=f"{K2_RELEASE_BASE_URL}/asr-models/"
                    "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        extracted_dir_name="sherpa-onnx-streaming-zipformer-en-2023-06-26",
        local_dir_name="sherpa-zipformer-en-2023-06-26",
        files_to_keep=(
            "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
            "tokens.txt",
        ),
    ),
    "asr-2023-06-21": SherpaModelDownloadSpec(
        release_url=f"{K2_RELEASE_BASE_URL}/asr-models/"
                    "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        extracted_dir_name="sherpa-onnx-streaming-zipformer-en-2023-06-21",
        local_dir_name="sherpa-zipformer-en-2023-06-21",
        files_to_keep=(
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ),
    ),
    "punct": SherpaModelDownloadSpec(
        release_url=f"{K2_RELEASE_BASE_URL}/punctuation-models/"
                    "sherpa-onnx-online-punct-en-2024-08-06.tar.bz2",
        extracted_dir_name="sherpa-onnx-online-punct-en-2024-08-06",
        local_dir_name="sherpa-online-punct-en",
        files_to_keep=("model.int8.onnx", "bpe.vocab"),
    ),
}

# What `main()` downloads when given no arguments (the committed bundle).
DEFAULT_TARGETS = ("asr", "punct")


def _download_extract_and_copy(spec: SherpaModelDownloadSpec):
    destination_directory = LOCAL_MODELS_PARENT_DIRECTORY / spec.local_dir_name
    destination_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "model.tar.bz2"
        print(f"[sherpa-download] fetching {spec.release_url} ...", flush=True)
        urllib.request.urlretrieve(spec.release_url, archive_path)
        print("[sherpa-download] extracting ...", flush=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(temp_dir)
        extracted_root = Path(temp_dir) / spec.extracted_dir_name
        for filename in spec.files_to_keep:
            source = extracted_root / filename
            if not source.is_file():
                raise SystemExit(f"expected file missing in archive: {source}")
            shutil.copyfile(source, destination_directory / filename)
            print(f"[sherpa-download] -> {destination_directory / filename}", flush=True)


def main():
    requested = sys.argv[1:] or list(DEFAULT_TARGETS)
    if requested == ["all"]:
        requested = list(SHERPA_MODEL_DOWNLOAD_SPECS_BY_TARGET.keys())
    for target in requested:
        spec = SHERPA_MODEL_DOWNLOAD_SPECS_BY_TARGET.get(target)
        if spec is None:
            valid = ", ".join(SHERPA_MODEL_DOWNLOAD_SPECS_BY_TARGET.keys())
            raise SystemExit(f"unknown target '{target}' (use one of: {valid}, all)")
        _download_extract_and_copy(spec)
    print("[sherpa-download] done.", flush=True)


if __name__ == "__main__":
    main()
