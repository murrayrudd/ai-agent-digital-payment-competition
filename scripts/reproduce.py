"""Verify the published archives, then run the study in a fresh temporary directory."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]

def verify(archive):
    for line in archive.read("SHA256SUMS").decode().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(archive.read(name)).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monetary", action="store_true", help="refit the monetary specifications and resample 1,000 times")
    parser.add_argument("--scenarios", action="store_true", help="repeat the nine 1,000-resample scenario fits")
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with ZipFile(ROOT / "development/instrument-development-v1.0.0.zip") as archive:
        verify(archive)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with ZipFile(ROOT / "study/replication-v1.0.0.zip") as archive:
            verify(archive)
            archive.extractall(work)
        subprocess.run([sys.executable, str(work / "main/validate_design.py")], env=env, check=True, cwd=work)
        subprocess.run([sys.executable, str(work / "refit_predictions.py")], env=env, check=True, cwd=work)
        check = [sys.executable, str(work / "check_study.py")]
        if args.monetary:
            check.append("--monetary")
        subprocess.run(check, env=env, check=True, cwd=work)
        if args.scenarios:
            subprocess.run([sys.executable, str(work / "scenario/scenario_analysis.py"), "--replicates", "1000"], env=env, check=True, cwd=work)
        print("Both publication archives passed their checksums.")

if __name__ == "__main__":
    main()
