"""Validated command-line entry point for antibody-epitope inference."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "weights/VJ_tokenizer.pkl",
    "weights/PSSM_repr/AAIMax_PSSM_repr.npy",
    "weights/PSSM_repr/AAIMin_PSSM_repr.npy",
    "weights/model/fold0.ckpt",
    "weights/model/fold1.ckpt",
    "weights/model/fold2.ckpt",
    "weights/model/fold3.ckpt",
    "weights/model/fold4.ckpt",
    "weights/ablang_weight/model-weight-heavy",
    "Epitope_lib/epitope_new.csv",
    "Epitope_lib/Epitopelib_pdb",
    "Epitope_lib/Epitope_Feature/epitopelib_DSSP",
    "Epitope_lib/Epitope_Feature/epitopelib_LLM",
    "Epitope_lib/Epitope_Feature/epitopelib_PSSM",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one antibody against every epitope in Epitope_lib."
    )
    parser.add_argument("--name", required=True, help="Antibody name, for example ab1")
    return parser.parse_args()


def validate(name: str) -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    for path in (f"oas_fasta/{name}.fasta", f"oas_VJ/{name}_VJ.csv"):
        if not (ROOT / path).exists():
            missing.append(path)

    pssm = ROOT / "raw_pssm" / f"{name}_AbH.pssm"
    if not pssm.exists():
        missing.append(str(pssm.relative_to(ROOT)))

    if shutil.which("mkdssp") is None and not (ROOT / "mkdssp").exists():
        missing.append("mkdssp executable")

    if missing:
        print("Missing inference resources:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    args = parse_args()
    validate(args.name)
    command = [sys.executable, str(ROOT / "prediction_Abh.py"), args.name]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
