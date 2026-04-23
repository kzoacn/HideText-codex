from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
GHOSTEXT_ROOT = SCRIPT_DIR.parent
OPTIONAL_PAPER_ROOT = GHOSTEXT_ROOT.parent / "Ghostext-paper"

FULL_COVER_TEXT_POLICY = (
    "per-run JSONL releases full cover text; summary bundles retain hashes and aggregate metrics"
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sh(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return "unknown"
    return out.strip() or "unknown"


def maybe_git_head(repo: Path | None) -> str:
    if repo is None or not repo.exists():
        return "unknown"
    return sh(["git", "-C", str(repo), "rev-parse", "HEAD"])


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def sha256_hex(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def parse_cmake_set_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    pattern = re.compile(r"""^\s*set\(([A-Za-z0-9_]+)\s+(?:"([^"]*)"|([^\s\)]+))\)\s*$""")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        values[match.group(1)] = match.group(2) or match.group(3) or ""
    return values


def read_llama_cpp_build_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "llama_cpp_python_version": "unknown",
        "llama_cpp_commit_sha": "unknown",
        "llama_cpp_build_number": "unknown",
        "ggml_build_commit": "unknown",
        "ggml_build_number": "unknown",
        "ggml_version": "unknown",
        "llama_system_info": "unknown",
    }
    try:
        import llama_cpp
    except Exception:
        return info

    try:
        info["llama_cpp_python_version"] = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        pass

    package_root = Path(inspect.getfile(llama_cpp)).resolve().parent.parent
    llama_cmake = package_root / "lib" / "cmake" / "llama" / "llama-config.cmake"
    ggml_cmake = package_root / "lib" / "cmake" / "ggml" / "ggml-config.cmake"
    llama_vars = parse_cmake_set_file(llama_cmake)
    ggml_vars = parse_cmake_set_file(ggml_cmake)
    info["llama_cpp_commit_sha"] = llama_vars.get("LLAMA_BUILD_COMMIT", info["llama_cpp_commit_sha"])
    info["llama_cpp_build_number"] = llama_vars.get(
        "LLAMA_BUILD_NUMBER",
        info["llama_cpp_build_number"],
    )
    info["ggml_build_commit"] = ggml_vars.get("GGML_BUILD_COMMIT", info["ggml_build_commit"])
    info["ggml_build_number"] = ggml_vars.get("GGML_BUILD_NUMBER", info["ggml_build_number"])
    info["ggml_version"] = ggml_vars.get("GGML_VERSION", info["ggml_version"])

    try:
        system_info = llama_cpp.llama_cpp.llama_print_system_info()
        if isinstance(system_info, bytes):
            info["llama_system_info"] = system_info.decode("utf-8", errors="replace").strip()
        else:
            info["llama_system_info"] = str(system_info).strip()
    except Exception:
        pass
    return info


def read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, value = line.split(":", 1)
                return value.strip()
    output = sh(["lscpu"])
    for line in output.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def read_total_ram_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0.0
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("MemTotal:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            break
        return float(parts[1]) / (1024.0 * 1024.0)
    return 0.0


def build_runtime_info() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_timestamp(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": read_cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_total_gib": round(read_total_ram_gib(), 2),
        "ghostext_git_head": maybe_git_head(GHOSTEXT_ROOT),
        "paper_git_head": maybe_git_head(OPTIONAL_PAPER_ROOT),
        **read_llama_cpp_build_info(),
    }


def extract_model_provenance(
    *,
    model_path: Path,
    resolved_model_source: str,
    backend_metadata: dict[str, Any],
    llama_metadata: dict[str, Any],
) -> dict[str, Any]:
    base_models: list[dict[str, Any]] = []
    raw_count = llama_metadata.get("general.base_model.count", 0)
    try:
        base_model_count = int(raw_count)
    except Exception:
        base_model_count = 0
    for index in range(base_model_count):
        prefix = f"general.base_model.{index}."
        item = {
            "name": llama_metadata.get(prefix + "name"),
            "organization": llama_metadata.get(prefix + "organization"),
            "repo_url": llama_metadata.get(prefix + "repo_url"),
        }
        if any(value not in (None, "") for value in item.values()):
            base_models.append(item)

    return {
        "resolved_model_path": str(model_path),
        "resolved_model_source": resolved_model_source,
        "file_size_bytes": model_path.stat().st_size,
        "backend_metadata": backend_metadata,
        "gguf_metadata": {
            "general.architecture": llama_metadata.get("general.architecture"),
            "general.name": llama_metadata.get("general.name"),
            "general.basename": llama_metadata.get("general.basename"),
            "general.size_label": llama_metadata.get("general.size_label"),
            "general.license": llama_metadata.get("general.license"),
            "general.file_type": llama_metadata.get("general.file_type"),
            "tokenizer.ggml.model": llama_metadata.get("tokenizer.ggml.model"),
            "tokenizer.ggml.pre": llama_metadata.get("tokenizer.ggml.pre"),
        },
        "upstream_provenance": {
            "canonical_model_label": llama_metadata.get("general.name"),
            "base_models": base_models,
        },
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    iterations: int = 10000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        estimates.append(statistic(sample))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = estimates[int(alpha * (len(estimates) - 1))]
    hi = estimates[int((1.0 - alpha) * (len(estimates) - 1))]
    return (lo, hi)
