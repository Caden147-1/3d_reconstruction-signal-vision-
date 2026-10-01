"""Isolated adapter for the official VideoScene one-step video_gen entry point.

Uses an observed COLMAP render instead of NoPoSplat's inferred scene prior.
The pretrained diffusion model is unchanged; this is not the paper's complete
training/evaluation pipeline. No generated video is treated as an observation.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time

from process_control import managed_process


CODE_REVISION = "de23c631fdb0663e5e732317c4642f38e4bd295f"
WEIGHT_REVISION = "cf6075132cb3d7e1b42b6e7e382748b424387f19"
WEIGHT_FILES = {
    "model.safetensors": (16077338768, "fb1eed6a728ecb92a21f4ec1d0b691542e482c2cfbba75e3f105eddfeda6fb02"),
    "prompt_embeds.pt": (1852602, "5fc44a5ac3667abb849b812349196197a8dcfe6c44f631dac3bee0028d732b1d"),
    "CogvideoX-Interpolation/vae/diffusion_pytorch_model.safetensors": (431221142, "bd47d57ad948ff80da0af0cb2e4dcdef65073aba59bccfd383ada9a7d1c02024"),
    "CogvideoX-Interpolation/transformer/diffusion_pytorch_model.safetensors": (11250292088, "43dcb0b051f30458eda7f13b3513a0b5c9ffa0a290709b3ccc98de30132518af"),
}
CONFIG_FILES = {
    "CogvideoX-Interpolation/transformer/config.json": (802, "c99b9c27aba51f362f6195652eedb2576fa6cc5b9fceaedc00028a4a1679f1d2"),
    "CogvideoX-Interpolation/vae/config.json": (839, "836132e6da4015a5c1a3c19beba3fd639ad15879293ea4c63745006576fe6f91"),
    "CogvideoX-Interpolation/scheduler/scheduler_config.json": (482, "247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5"),
}


def isolated_python(repo):
    return repo / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def readiness(repo, python):
    weights = repo / "checkpoints/checkpoints"
    missing = []
    for path in (python, repo / "videoscene/pipeline.py"):
        if not path.is_file():
            missing.append(str(path))
    for name, (size, _) in WEIGHT_FILES.items():
        path = weights / name
        if not path.is_file() or path.stat().st_size != size:
            missing.append(str(path))
    for name in CONFIG_FILES:
        path = weights / name
        try:
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                raise ValueError("Not a model config")
        except (OSError, ValueError):
            missing.append(str(path))
    return missing


def runtime_readiness(repo, python, timeout=90):
    """Probe the isolated interpreter without loading the multi-GB checkpoints."""
    command = [str(python), str(Path(__file__).resolve()), "--repo", str(repo.resolve()), "--check-runtime"]
    try:
        with managed_process(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace") as process:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode:
                return {"status": "runtime_unavailable", "error": (stderr or stdout)[-4000:].strip() or f"Exit code {process.returncode}"}
        report = json.loads(stdout)
        if not isinstance(report, dict) or report.get("status") != "runtime_ready_not_inference_tested":
            raise ValueError("Unexpected VideoScene runtime probe response")
        return report
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"status": "runtime_unavailable", "error": str(exc)}


def runtime_preflight(repo, python):
    report = runtime_readiness(repo, python)
    if report["status"] != "runtime_ready_not_inference_tested":
        print(f"[error] VideoScene runtime preflight failed: {report['error']}", flush=True)
        print("[VideoScene] Check the isolated environment and CUDA driver; See README.md for the isolated environment requirements and asset layout.", flush=True)
        return False
    print(f"[VideoScene] Runtime ready: {report['gpu']}; CUDA/BF16/imports/attention checked, full inference not tested.", flush=True)
    return True


def environment_preflight(repo: Path, python: Path | None = None) -> bool:
    """Check required assets and the isolated runtime before a reconstruction reset."""
    python = python if python is not None else isolated_python(repo)
    missing = readiness(repo, python)
    if missing:
        print_readiness_failure(repo, python, missing)
        return False
    return runtime_preflight(repo, python)


def check_official_code(repo):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != CODE_REVISION or subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "videoscene"], cwd=repo):
        raise ValueError("VideoScene code differs from the pinned official implementation")


def probe_runtime(repo):
    import cv2
    import decord
    import imageio
    import numpy
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from safetensors import safe_open
    from torch.nn.attention import sdpa_kernel, SDPBackend

    check_official_code(repo)
    sys.path.insert(0, str(repo))
    from videoscene.pipeline import VideoScenePipeline
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("VideoScene requires a working BF16-capable CUDA GPU")
    value = torch.zeros((1, 2, 16, 64), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        result = torch.nn.functional.scaled_dot_product_attention(value, value, value)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Non-finite output from CUDA attention preflight")
    torch.cuda.synchronize()
    return {"status": "runtime_ready_not_inference_tested", "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "bf16": True,
            "inference_tested": False}


def print_readiness_failure(repo, python, missing):
    print("[error] VideoScene assets are not ready; diffusion was not run and reconstruction was not reset.", flush=True)
    print(f"[VideoScene] Isolated interpreter: {python}", flush=True)
    for value in missing:
        path = Path(value)
        reason = "missing"
        try:
            if path.is_file():
                with path.open("rb") as handle:
                    pointer = handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
                reason = "Git LFS pointer, not downloaded content" if pointer else "invalid or incomplete file"
        except OSError as exc:
            reason = f"unreadable: {exc}"
        print(f"  - {path}: {reason}", flush=True)
    print(f"[VideoScene] Prepare the pinned code at {repo} and model assets at {repo / 'checkpoints/checkpoints'}; see README.md.", flush=True)


def verify_weights(weights):
    import hashlib
    for name, (size, expected) in {**WEIGHT_FILES, **CONFIG_FILES}.items():
        path = weights / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b""):
                digest.update(block)
        if path.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"VideoScene official weight hash mismatch: {name}")


def transformer_weight_sources(expected, base, distilled):
    """Mirror upstream's full base + partial distilled override, with strict coverage."""
    if set(base) != set(expected):
        raise ValueError("Official base transformer does not completely match its config")
    if not distilled or not set(distilled).issubset(expected):
        raise ValueError("Unexpected or empty VideoScene distilled transformer weights")
    for source in (base, distilled):
        for name, shape in source.items():
            if tuple(shape) != tuple(expected[name]):
                raise ValueError(f"VideoScene weight shape mismatch: {name}")
    return {name: "distilled" if name in distilled else "base" for name in expected}


def load_pipeline(repo):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from safetensors import safe_open

    check_official_code(repo)
    sys.path.insert(0, str(repo))
    from videoscene.pipeline import VideoScenePipeline
    weights = repo / "checkpoints/checkpoints"
    verify_weights(weights)
    base = weights / "CogvideoX-Interpolation"
    config = CogVideoXTransformer3DModel.load_config(str(base / "transformer"), local_files_only=True)
    # Upstream loads the interpolation backbone, then a partial distilled state
    # dict. Materialize the final selection once, without two full CPU copies.
    with init_empty_weights():
        transformer = CogVideoXTransformer3DModel.from_config(config)
    expected = {name: tensor.shape for name, tensor in transformer.state_dict().items()}
    with safe_open(str(base / "transformer/diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as backbone, \
         safe_open(str(weights / "model.safetensors"), framework="pt", device="cpu") as distilled:
        sources = transformer_weight_sources(expected,
                   {name: backbone.get_slice(name).get_shape() for name in backbone.keys()},
                   {name: distilled.get_slice(name).get_shape() for name in distilled.keys()})
        for name, source in sources.items():
            value = (distilled if source == "distilled" else backbone).get_tensor(name)
            set_module_tensor_to_device(transformer, name, "cpu", value=value, dtype=torch.bfloat16)
    del expected
    transformer.eval()
    vae = AutoencoderKLCogVideoX.from_pretrained(str(base / "vae"), torch_dtype=torch.bfloat16, local_files_only=True)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(str(base / "scheduler"), local_files_only=True)
    pipe = VideoScenePipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()
    prompt = torch.load(weights / "prompt_embeds.pt", map_location="cpu", weights_only=True)
    if not isinstance(prompt, torch.Tensor) or prompt.ndim != 3 or not torch.isfinite(prompt).all():
        raise ValueError("Invalid pretrained prompt embedding")
    return pipe, prompt


def export_frames(frames, output):
    if len(frames) != 49 or any(frame.size != (720, 480) for frame in frames):
        raise ValueError("Missing or incorrectly sized VideoScene output frames")
    output.mkdir(exist_ok=False)
    for index, frame in enumerate(frames):
        frame.convert("RGB").save(output / f"{index:03d}.png", format="PNG")


def generate(repo, job, seed):
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("VideoScene requires a BF16-capable CUDA GPU")
    free, total = torch.cuda.mem_get_info()
    budget = min(9*1024**3, free-1536*1024**2)
    if budget < 4*1024**3:
        raise RuntimeError("Insufficient free GPU memory for the guarded VideoScene trial")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.manual_seed(seed)
    start = time.monotonic()
    pipe, prompt = load_pipeline(repo)
    import videoscene.pipeline as official
    with np.load(job / "prior.npz", allow_pickle=False) as data:
        frames = data["rgb"]
        endpoints = data["endpoints"]
    if frames.shape != (49, 480, 720, 3) or frames.dtype != np.uint8 or endpoints.shape != (2, 480, 720, 3):
        raise ValueError("Prior must contain exactly 49 calibrated 720x480 frames")
    # Bypass the upstream decord resize/loop, keeping pixel-to-camera calibration
    # exact. All actual one-step diffusion inference uses upstream video_gen.
    previous_reader = official.process_video
    previous_export = official.export_to_video
    official.process_video = lambda _: torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    # Geometry validation consumes lossless frames, not an MP4 round trip.
    official.export_to_video = lambda images, *a, **kw: export_frames(images, job / "generated")
    try:
        # Do not fall back to quadratic-memory math attention on Windows builds
        # without FlashAttention; the installed efficient CUDA kernel is supported.
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            pipe.video_gen(str(job / "prediction_tmp.mp4"), endpoints.astype(np.float32)/255., prompt, noise_timestep=460)
    finally:
        official.process_video = previous_reader
        official.export_to_video = previous_export
    if len(list((job / "generated").glob("*.png"))) != 49:
        raise ValueError("VideoScene did not export exactly 49 frames")
    return {"status": "generated", "model": "VideoScene", "code_revision": CODE_REVISION,
            "weights_revision": WEIGHT_REVISION, "seed": seed, "frames": 49,
            "elapsed_s": time.monotonic()-start, "allocator_budget_mib": budget/1024**2,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
            "offload": "sequential_cpu", "prior": "observed_colmap_not_noposplat",
            "output_format": "lossless_png",
            "interpretation": "Synthetic view proposals, not recovered measurements"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--check-runtime", action="store_true", help="Check imports and a tiny CUDA/BF16 attention operation; no model loading")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if not args.check_runtime and args.job is None:
        parser.error("--job is required unless --check-runtime is used")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.repo.resolve() / ".cache/huggingface")
    os.environ["TORCH_HOME"] = str(args.repo.resolve() / ".cache/torch")
    if args.check_runtime:
        print(json.dumps(probe_runtime(args.repo.resolve())), flush=True)
        return
    result = generate(args.repo.resolve(), args.job.resolve(), args.seed)
    (args.job / "inference.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
