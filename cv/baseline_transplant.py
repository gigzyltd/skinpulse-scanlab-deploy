"""Bind baseline normalisation into this process for the sensitivity harness.

Commit f879755 has no sensitivity harness. The baseline column therefore runs
the harness in this tree and replaces one function: ``normalise``. Provenance
strings that name that function are stamped from the same commit. No other
module is replaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from scanlab.cv.support import ScanLabError
from scanlab.cv.versions import ILLUMINATION_VERSION, NORMALISATION_VERSION, PIPELINE_VERSION

BASELINE_COMMIT = "f87975571a1ea57a964ba6968f799973cbe2b659"
BASELINE_NORMALISE_REPO_PATH = "services/scanlab/cv/baseline/normalise_f879755.py"
BASELINE_NORMALISE_GIT_PATH = "services/scanlab/cv/normalise.py"
CANDIDATE_NORMALISE_REPO_PATH = "services/scanlab/cv/normalise.py"

# Strings from f879755 services/scanlab/cv/versions.py. The CV package does not
# shell out to git; tests compare these literals to that blob when the clone has it.
BASELINE_PROVENANCE = {
    "PIPELINE_VERSION": "scanlab-cv-0.2.1",
    "NORMALISATION_VERSION": "colour-illum-v1",
    "ILLUMINATION_VERSION": "masked-gaussian-v1",
}

_BASELINE_FILE = Path(__file__).resolve().parent / "baseline" / "normalise_f879755.py"
_CANDIDATE_FILE = Path(__file__).resolve().parent / "normalise.py"


def candidate_transplant_record() -> dict:
    """Provenance for a run that did not replace any module."""

    return {
        "used": False,
        "role": "candidate",
        "source_commit": None,
        "transplanted_files": [],
        "provenance_files": [],
        "normalise_file": CANDIDATE_NORMALISE_REPO_PATH,
        "normalise_source_id": _content_id(_CANDIDATE_FILE),
        "pipeline_version": PIPELINE_VERSION,
        "normalisation_version": NORMALISATION_VERSION,
        "illumination_version": ILLUMINATION_VERSION,
        "other_modules_replaced": [],
        "note": (
            "No module transplant. scanlab.cv.pipeline.normalise is the normalise.py "
            "imported with this process. Provenance strings are the in-process versions "
            f"module ({PIPELINE_VERSION}, {NORMALISATION_VERSION}, {ILLUMINATION_VERSION})."
        ),
    }


def apply_baseline_transplant() -> dict:
    """Load baseline normalise.py and bind it over the pipeline function.

    Also binds the three baseline version strings onto the pipeline and the
    sensitivity harness so the report names masked-gaussian-v1 and
    scanlab-cv-0.2.1. The in-process versions module is left as imported.
    """

    module = _load_baseline_normalise()
    import scanlab.cv.pipeline as pipeline
    import scanlab.cv.sensitivity as sensitivity

    pipeline.normalise = module.normalise
    for target in (pipeline, sensitivity):
        target.PIPELINE_VERSION = BASELINE_PROVENANCE["PIPELINE_VERSION"]
        target.NORMALISATION_VERSION = BASELINE_PROVENANCE["NORMALISATION_VERSION"]
        target.ILLUMINATION_VERSION = BASELINE_PROVENANCE["ILLUMINATION_VERSION"]
    return {
        "used": True,
        "role": "baseline",
        "source_commit": BASELINE_COMMIT,
        "transplanted_files": [BASELINE_NORMALISE_GIT_PATH],
        "provenance_files": ["services/scanlab/cv/versions.py"],
        "vendored_file": BASELINE_NORMALISE_REPO_PATH,
        "normalise_file": BASELINE_NORMALISE_GIT_PATH,
        "normalise_source_id": _content_id(_BASELINE_FILE),
        "pipeline_version": BASELINE_PROVENANCE["PIPELINE_VERSION"],
        "normalisation_version": BASELINE_PROVENANCE["NORMALISATION_VERSION"],
        "illumination_version": BASELINE_PROVENANCE["ILLUMINATION_VERSION"],
        "other_modules_replaced": [],
        "note": (
            "Module transplant used. "
            f"{BASELINE_NORMALISE_REPO_PATH} is the exact {BASELINE_NORMALISE_GIT_PATH} "
            f"blob from commit {BASELINE_COMMIT}. It is loaded in-process and bound over "
            "scanlab.cv.pipeline.normalise. PIPELINE_VERSION, NORMALISATION_VERSION, and "
            "ILLUMINATION_VERSION on the pipeline and this harness are the strings from "
            "that commit's versions.py (scanlab-cv-0.2.1, colour-illum-v1, "
            "masked-gaussian-v1). The transplanted file still imports COLOUR_IDENTITY and "
            "COLOUR_NEUTRAL_VERSION from the in-process versions module; those two strings "
            "are unchanged at that commit. Metrics, quality gates, synthetic cases, the "
            "geometric backend, and the LIMITATIONS tuple were not replaced."
        ),
    }


def _load_baseline_normalise():
    if not _BASELINE_FILE.is_file():
        raise ScanLabError("baseline_normalise_missing", "The baseline normalise.py file is missing.")
    source = _BASELINE_FILE.read_text(encoding="utf-8")
    if "ILLUMINATION_LUMA_ANCHOR" in source:
        raise ScanLabError(
            "baseline_normalise_invalid",
            "The baseline normalise.py file contains the luma anchor.",
        )
    name = "scanlab_baseline_normalise_f879755"
    spec = importlib.util.spec_from_file_location(name, _BASELINE_FILE)
    if spec is None or spec.loader is None:
        raise ScanLabError("baseline_normalise_invalid", "The baseline normalise.py file could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "normalise", None)):
        raise ScanLabError("baseline_normalise_invalid", "The baseline normalise.py file has no normalise function.")
    return module


def _content_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
