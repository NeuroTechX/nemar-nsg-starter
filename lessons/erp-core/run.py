#!/usr/bin/env python3
"""Run MNE-BIDS-Pipeline on one ERP CORE task, then measure the component.

This is the NSG entry point. It uses only the standard library until the
pinned analysis packages are installed, so it starts on the stock
"Python on Expanse" tool (Python 3.11.4, read-only environment).

    python run.py                       # P3, all 40 participants
    python run.py --subjects 001,002    # a quick subset
    python run.py --dataset-root data/ERP_CORE --skip-install   # local run in a venv
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text(encoding="utf-8"))
# prepare.py writes job.json so a portal job (which takes no arguments) runs what was packaged.
JOB = json.loads((HERE / "job.json").read_text(encoding="utf-8")) if (HERE / "job.json").is_file() else {}
PIPELINE_VERSION = SOURCE["pipeline"]["version"]
OSF_FILES = "https://files.osf.io/v1/resources/9f5w7/providers/osfstorage"


def scratch():
    """Node-local scratch. NSG returns the whole job directory, so keep bulk data out of it."""
    return Path(os.environ.get("TMPDIR") or "/tmp") / "erpcore"


def read_manifest(task, subjects):
    with (HERE / "manifests" / f"{task}.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    every = sorted({row["path"].split("/")[0][4:] for row in rows if row["path"].startswith("sub-")})
    chosen = every if subjects == "all" else subjects.split(",")
    missing = set(chosen) - set(every)
    if missing:
        raise ValueError(f"No such ERP CORE participants: {sorted(missing)}")
    keep = [r for r in rows if not r["path"].startswith("sub-") or r["path"][4:7] in chosen]
    return chosen, keep


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, rows):
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
        if path.stat().st_size != int(row["size"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"{path} differs from the OSF file pinned in manifests/")


def download(root, rows, attempts=6):
    """Fetch the pinned OSF files that are not already present and correct."""
    for row in rows:
        path = root / row["path"]
        if path.is_file() and path.stat().st_size == int(row["size"]) and sha256(path) == row["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{OSF_FILES}/{row['osf_file_id']}?version={row['osf_version']}"
        partial = path.with_name(path.name + ".part")
        for attempt in range(attempts):
            try:
                with urlopen(url, timeout=120) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out, 1 << 20)
                if sha256(partial) != row["sha256"]:
                    raise ValueError(f"checksum mismatch for {row['path']}")
                partial.replace(path)
                break
            except (OSError, URLError, ValueError) as exc:
                wait = 15 * (attempt + 1)
                print(f"download retry {attempt + 1} for {row['path']} in {wait} s: {exc}", flush=True)
                time.sleep(wait)
        else:
            raise ValueError(f"Could not download {row['path']} from OSF")


def locate_data(explicit, rows):
    if explicit is not None:
        return explicit, "local"
    bundled = HERE / "data" / "ERP_CORE"
    if bundled.is_dir():
        return bundled, "bundle"
    target = scratch() / "ERP_CORE"
    started = time.monotonic()
    download(target, rows)
    print(f"Downloaded and verified {len(rows)} files in {time.monotonic() - started:.0f} s", flush=True)
    return target, "OSF download"


def install(deps):
    """Install the pinned packages into node-local scratch.

    The tool's own environment is read-only and has older numpy and MNE. Installing every
    pinned package with --ignore-installed and running the analysis in a fresh interpreter
    with PYTHONPATH keeps the old copies from shadowing the new ones.
    """
    stamp = deps / f".mne-bids-pipeline-{PIPELINE_VERSION}"
    if stamp.exists():
        return
    command = [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--ignore-installed",
               "--no-warn-script-location", "--target", str(deps), "-r", str(HERE / "requirements.txt")]
    wheels = HERE / "wheels"
    if wheels.is_dir():  # bundled with `prepare.py zip --wheels` for a job without internet
        command[4:4] = ["--no-index", "--find-links", str(wheels)]
    started = time.monotonic()
    subprocess.run(command, check=True)
    stamp.touch()
    print(f"Installed the pinned packages in {time.monotonic() - started:.0f} s", flush=True)


def child_env(deps, extra):
    env = dict(os.environ, **extra, MPLBACKEND="Agg", PYTHONNOUSERSITE="1")
    if deps is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(deps), env.get("PYTHONPATH")]))
        env["PATH"] = os.pathsep.join([str(deps / "bin"), env.get("PATH", "")])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=JOB.get("task", "P3"), choices=SOURCE["supported_tasks"])
    parser.add_argument("--subjects", default=JOB.get("subjects", "all"), help="'all' or comma-separated labels such as 001,002")
    parser.add_argument("--dataset-root", type=Path, help="an existing copy of the ERP CORE BIDS folder")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--deriv-root", type=Path, help="pipeline derivatives (default: node-local scratch)")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="participants processed in parallel (8 was tested with 12 GB)")
    parser.add_argument("--skip-install", action="store_true",
                        help="use the packages already importable by this Python (local venv)")
    parser.add_argument("--keep-bundle", action="store_true",
                        help="keep bundled data and wheels in the job directory (NSG would send them back)")
    parser.add_argument("--preflight", action="store_true", help="check the inputs and stop")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        subjects, rows = read_manifest(args.task, args.subjects)
        root, location = locate_data(args.dataset_root, rows)
        verify(root, rows)
        print(f"Verified {len(rows)} input files for {len(subjects)} participants ({location})", flush=True)
        if args.preflight:
            return
        deps = None
        if not args.skip_install:
            deps = scratch() / "deps"
            install(deps)
        deriv = args.deriv_root or scratch() / "derivatives"
        env = child_env(deps, {
            "ERPCORE_TASK": args.task, "ERPCORE_BIDS_ROOT": str(root), "ERPCORE_DERIV_ROOT": str(deriv),
            "ERPCORE_SUBJECTS": ",".join(subjects), "ERPCORE_N_JOBS": str(args.jobs),
        })
        pipeline_started = time.monotonic()
        subprocess.run([sys.executable, "-c", "from mne_bids_pipeline._main import main; main()",
                        "--config", str(HERE / "config_erpcore.py"), "--steps", "init,preprocessing,sensor"],
                       check=True, env=env)
        pipeline_s = time.monotonic() - pipeline_started
        subprocess.run([sys.executable, str(HERE / "summarize.py"), "--task", args.task,
                        "--bids-root", str(root), "--deriv-root", str(deriv), "--output", str(args.output),
                        "--subjects", ",".join(subjects)], check=True, env=env)
        # Keep the small outputs: the group report and each participant's averaged ERPs.
        # Per-participant reports (about 12 MB each) and epochs stay in scratch.
        for path in deriv.glob(f"sub-average/ses-{args.task}/eeg/*_report.html"):
            (args.output / "reports").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.output / "reports" / path.name)
        for path in deriv.glob(f"sub-*/ses-{args.task}/eeg/sub-*_task-{args.task}_ave.fif"):
            (args.output / "evoked").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.output / "evoked" / path.name)
        manifest_path = args.output / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["job"] = {
            "input_location": location, "host": socket.gethostname(), "launcher_python": platform.python_version(),
            "cpu_count": os.cpu_count(), "jobs": args.jobs, "pipeline_seconds": round(pipeline_s),
            "total_seconds": round(time.monotonic() - started),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if JOB and not args.keep_bundle:  # only inside a ZIP built by prepare.py
            # NSG returns the whole job directory, so drop bundled inputs before the job ends.
            for bundled in (HERE / "data", HERE / "wheels"):
                if bundled.is_dir() and (bundled.name == "wheels" or location == "bundle"):
                    shutil.rmtree(bundled)
        print(f"Done in {time.monotonic() - started:.0f} s. Results are in {args.output}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERP CORE job could not finish: {exc}. See the lesson README for setup.")


if __name__ == "__main__":
    main()
