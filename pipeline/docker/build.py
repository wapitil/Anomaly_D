from __future__ import annotations

import subprocess
from pathlib import Path

# Docker/OpenExplorer: compile ONNX to RDK X5 .bin.
# Input: runs/res_640/onnx/res_640.onnx and calibration images.
# Output: runs/res_640/model_output/res_640.bin.

CONFIG = Path("pipeline/docker/res_640.yaml")
MODEL_TYPE = "onnx"


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit(f"config not found: {CONFIG}")
    subprocess.run(
        ["hb_mapper", "makertbin", "--config", str(CONFIG), "--model-type", MODEL_TYPE],
        check=True,
    )


if __name__ == "__main__":
    main()
