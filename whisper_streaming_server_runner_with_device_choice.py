"""
Thin wrapper around whisper_streaming/whisper_online_server.py that lets
us pick CPU vs GPU at launch time without modifying the submodule.

The submodule's whisper_online.py hardcodes device="cuda" inside
FasterWhisperASR.load_model. We monkey-patch that one method before
importing/invoking the server, so a single env var controls it:

    WHISPER_DEVICE=cuda  (default)  -> device="cuda",  compute_type="float16"
    WHISPER_DEVICE=cpu               -> device="cpu",   compute_type="int8"

Everything else passes through unchanged. All command-line flags accepted
by whisper_online_server.py are accepted here too.
"""

import ctypes
import glob
import os
import runpy
import sys
from pathlib import Path


def preload_pip_installed_nvidia_shared_libraries_for_gpu_mode():
    """
    CTranslate2 (used by faster-whisper) calls dlopen('libcudnn_ops.so.9')
    at transcribe time. When cuDNN is only available via the pip wheel
    `nvidia-cudnn-cu12` (the common case for users without system CUDA),
    that path is not on the dynamic linker search path and the dlopen
    fails -> server aborts mid-transcription with "Unable to load any of
    libcudnn_ops.so". We preload all libcudnn*/libcublas* with RTLD_GLOBAL
    so subsequent dlopens by CTranslate2 resolve. CPU mode doesn't need
    this — skip the work.
    """
    if os.environ.get("WHISPER_DEVICE", "cuda").strip().lower() == "cpu":
        return
    try:
        import nvidia.cudnn
        import nvidia.cublas
    except ImportError:
        return  # No pip-cudnn/cublas; either system CUDA or no GPU.
    nvidia_pip_package_lib_dirs = [
        os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"),
        os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib"),
    ]
    for nvidia_lib_dir in nvidia_pip_package_lib_dirs:
        if not os.path.isdir(nvidia_lib_dir):
            continue
        for shared_object_path in sorted(
            glob.glob(os.path.join(nvidia_lib_dir, "lib*.so*"))
        ):
            try:
                ctypes.CDLL(shared_object_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


preload_pip_installed_nvidia_shared_libraries_for_gpu_mode()


def determine_whisper_device_choice_and_compute_type():
    requested_device = os.environ.get("WHISPER_DEVICE", "cuda").strip().lower()
    if requested_device == "cpu":
        return "cpu", "int8"
    return "cuda", "float16"


def install_load_model_monkeypatch_for_chosen_device():
    """Patch FasterWhisperASR.load_model so it builds a WhisperModel with
    our chosen device + compute_type instead of the hardcoded cuda/float16."""
    chosen_device, chosen_compute_type = (
        determine_whisper_device_choice_and_compute_type()
    )

    # Make whisper_streaming importable as a package.
    repo_root_directory = Path(__file__).parent.resolve()
    whisper_streaming_directory = repo_root_directory / "whisper_streaming"
    sys.path.insert(0, str(whisper_streaming_directory))

    # Now import the module to access FasterWhisperASR before monkey-patching.
    import whisper_online as whisper_online_module

    def patched_load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        if model_dir is not None:
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")
        return WhisperModel(
            model_size_or_path,
            device=chosen_device,
            compute_type=chosen_compute_type,
            download_root=cache_dir,
        )

    whisper_online_module.FasterWhisperASR.load_model = patched_load_model
    print(
        f"[wrapper] patched FasterWhisperASR.load_model -> "
        f"device={chosen_device}, compute_type={chosen_compute_type}",
        flush=True,
    )


def run_whisper_streaming_server_main_module():
    """Execute whisper_online_server.py exactly as if it were invoked
    directly, but with our patched FasterWhisperASR in place."""
    repo_root_directory = Path(__file__).parent.resolve()
    server_script_path = (
        repo_root_directory / "whisper_streaming" / "whisper_online_server.py"
    )
    # Pass through all our argv to whisper_online_server (its argparse
    # reads from sys.argv).
    sys.argv[0] = str(server_script_path)
    runpy.run_path(str(server_script_path), run_name="__main__")


def main():
    install_load_model_monkeypatch_for_chosen_device()
    run_whisper_streaming_server_main_module()


if __name__ == "__main__":
    main()
