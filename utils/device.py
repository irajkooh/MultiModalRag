"""
Device detection and selection utility.
Priority: CUDA → MPS (Apple Silicon) → CPU

Usage:
    from utils.device import get_device, device_info
    device = get_device()          # e.g. "mps", "cuda", "cpu"
    print(device_info())           # human-readable summary
"""
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> str:
    """
    Detect the best available torch device.
    Respects the TORCH_DEVICE env var to force a specific device.
    
    Returns one of: "cuda", "mps", "cpu"
    """
    # Allow explicit override
    forced = os.environ.get("TORCH_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        logger.info(f"Device forced via TORCH_DEVICE env var: {forced}")
        return forced

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Auto-detected device: {device}")
        return device

    except ImportError:
        logger.warning("torch not importable — falling back to CPU")
        return "cpu"


@lru_cache(maxsize=1)
def device_info() -> dict:
    """
    Return a dict with device name, label, and extra GPU info where available.
    Cached so detection only runs once per process.
    """
    device = get_device()
    info = {"device": device, "label": device.upper()}

    try:
        import torch

        if device == "cuda":
            idx = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            info["vram_gb"] = round(total / 1024 ** 3, 1)
            info["label"] = f"CUDA — {info['gpu_name']} ({info['vram_gb']} GB)"

        elif device == "mps":
            # Try to get Apple chip name
            try:
                import subprocess
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                chip = "Apple Silicon"
            info["label"] = f"MPS — {chip}"

        else:
            info["label"] = "CPU"

    except Exception as e:
        logger.debug(f"device_info detail error: {e}")

    logger.info(f"Device info: {info['label']}")
