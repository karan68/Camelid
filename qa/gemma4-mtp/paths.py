"""Path resolution for the Gemma 4 MTP reference harness.

Every path here comes from the environment or from this file's own location,
never from a hardcoded operator path. The scripts beside this one originally
pinned a per-agent-session scratch directory and an absolute home path, which
made them unrunnable the moment that session ended and leaked the operator's
home directory into a public repository (`scripts/check-public-scrub.sh`
rejects both forms).

Overrides:
  GEMMA4_MTP_ASSISTANT  HF directory of the assistant checkpoint
                        (default: <repo>/models/gemma-4-26B-A4B-it-assistant)
  GEMMA4_MTP_WORKDIR    scratch holding oracle inputs and captured stages
                        (default: <this directory>/work)
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# `mtp_inputs` lives beside this file. Running a script by path already puts its
# own directory on sys.path, but importing this module from elsewhere should work
# too, so make it explicit rather than positional.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def assistant_dir() -> Path:
    """HF directory of the MTP assistant checkpoint."""
    return Path(
        os.environ.get(
            "GEMMA4_MTP_ASSISTANT", str(REPO / "models" / "gemma-4-26B-A4B-it-assistant")
        )
    )


def workdir() -> Path:
    """Scratch directory for oracle inputs and captured intermediates."""
    path = Path(os.environ.get("GEMMA4_MTP_WORKDIR", str(HERE / "work")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def oracle_npz() -> Path:
    """The committed oracle's `.npz`, which the caller must stage into the workdir.

    Fails loudly rather than returning a path that does not exist: a silent
    `FileNotFoundError` three frames deep reads like a code defect, when the real
    cause is simply that the bundle was never staged.
    """
    path = workdir() / "oracle.npz"
    if not path.exists():
        raise SystemExit(
            f"oracle.npz not found at {path}\n"
            "Stage it from qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle/, "
            "or point GEMMA4_MTP_WORKDIR at the directory that holds it."
        )
    return path
