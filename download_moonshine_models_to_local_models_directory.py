"""
Download Moonshine streaming model weights into <repo>/models/ using
Moonshine's own official downloader (moonshine_voice.get_model_for_language).

Produces, for offline use by moonshine_streaming_server.py:

    models/moonshine-tiny-streaming/   (encoder.ort, decoder_kv.ort,
    models/moonshine-small-streaming/   cross_kv.ort, adapter.ort,
                                        frontend.ort, streaming_config.json,
                                        tokenizer.bin, ...)

These are English streaming models (the real-time, CPU-optimized family).
They're LFS-tracked via .gitattributes (models/moonshine-*-streaming/*.ort),
matching how the Whisper tiny/base weights are bundled.

Usage:
    python3 download_moonshine_models_to_local_models_directory.py
    python3 download_moonshine_models_to_local_models_directory.py tiny
    python3 download_moonshine_models_to_local_models_directory.py tiny small
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# Our local model-directory name -> (upstream ModelArch attribute name).
# Keep in sync with MOONSHINE_STREAMING_MODELS in moonshine_streaming_backend.py.
MOONSHINE_STREAMING_VARIANTS = {
    "tiny": ("moonshine-tiny-streaming", "TINY_STREAMING"),
    "small": ("moonshine-small-streaming", "SMALL_STREAMING"),
}
LOCAL_MODELS_PARENT_DIRECTORY = Path(__file__).parent.resolve() / "models"


def download_one_streaming_variant_into_local_models_directory(variant_short_name):
    try:
        from moonshine_voice import ModelArch, get_model_for_language
    except ImportError as import_error:
        raise SystemExit(
            "moonshine-voice is required. Install with: pip install moonshine-voice"
        ) from import_error

    local_directory_name, model_arch_attribute_name = MOONSHINE_STREAMING_VARIANTS[
        variant_short_name
    ]
    model_arch = getattr(ModelArch, model_arch_attribute_name)

    print(
        f"[moonshine-download] fetching {local_directory_name} "
        f"({model_arch_attribute_name}) via moonshine-voice ...",
        flush=True,
    )
    # get_model_for_language downloads the component .ort files into the
    # moonshine-voice cache and returns the directory holding them.
    upstream_model_directory, _ = get_model_for_language("en", model_arch)
    upstream_model_directory = Path(upstream_model_directory)

    local_destination_directory = (
        LOCAL_MODELS_PARENT_DIRECTORY / local_directory_name
    )
    local_destination_directory.mkdir(parents=True, exist_ok=True)

    copied_file_count = 0
    for component_file_path in upstream_model_directory.iterdir():
        if not component_file_path.is_file():
            continue
        # Skip filelock artifacts (zero-byte *.lock) the downloader leaves
        # in its cache; only the real model components belong in our tree.
        if component_file_path.suffix == ".lock":
            continue
        shutil.copyfile(
            component_file_path,
            local_destination_directory / component_file_path.name,
        )
        copied_file_count += 1

    print(
        f"[moonshine-download] -> {local_destination_directory} "
        f"({copied_file_count} component files)",
        flush=True,
    )


def main():
    requested_variants = sys.argv[1:] or list(MOONSHINE_STREAMING_VARIANTS.keys())
    for requested_variant in requested_variants:
        if requested_variant not in MOONSHINE_STREAMING_VARIANTS:
            raise SystemExit(
                f"Unknown variant '{requested_variant}'. Available: "
                f"{', '.join(MOONSHINE_STREAMING_VARIANTS.keys())}"
            )
        download_one_streaming_variant_into_local_models_directory(requested_variant)
    print("[moonshine-download] done.", flush=True)


if __name__ == "__main__":
    main()
