#!/usr/bin/env python3
"""Download pinned ERP CORE files or build a portal-ready NSG input ZIP."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from run import HERE, SOURCE, download, read_manifest, verify

TOP = "erp_core"  # the NSG "subdirectory" setting must match this name
CODE = ("run.py", "summarize.py", "config_erpcore.py", "source.json", "requirements.txt",
        "reference/Mean_Amplitude_Diff_Waves_P3.txt")


def fetch(root, task, subjects):
    chosen, rows = read_manifest(task, subjects)
    download(root, rows)
    print(f"Verified {len(rows)} files for {len(chosen)} participants under {root}")


def wheels_for_nsg(directory):
    """Download the pinned wheels for NSG's Python 3.11 on x86-64 Linux."""
    subprocess.run([sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "--only-binary", ":all:",
                    "--python-version", "3.11", "--platform", "manylinux_2_28_x86_64",
                    "--platform", "manylinux_2_17_x86_64", "--platform", "manylinux2014_x86_64",
                    "--platform", "any", "--dest", str(directory), "-r", str(HERE / "requirements.txt")],
                   check=True)
    return sorted(directory.glob("*.whl"))


def package(output, task, subjects, data, wheels):
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name in CODE:
            archive.write(HERE / name, f"{TOP}/{name}")
        archive.writestr(f"{TOP}/job.json", json.dumps({"task": task, "subjects": subjects}) + "\n")
        for manifest in sorted((HERE / "manifests").glob("*.tsv")):
            archive.write(manifest, f"{TOP}/manifests/{manifest.name}")
        if wheels:
            with TemporaryDirectory() as temporary:
                for wheel in wheels_for_nsg(Path(temporary)):
                    archive.write(wheel, f"{TOP}/wheels/{wheel.name}", compress_type=ZIP_STORED)
        if data is not None:
            _, rows = read_manifest(task, subjects)
            verify(data, rows)
            for row in rows:
                archive.write(data / row["path"], f"{TOP}/data/ERP_CORE/{row['path']}", compress_type=ZIP_STORED)
    print(f"Created {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="download pinned files for one task from OSF")
    fetch_command.add_argument("dataset_root", type=Path, help="destination, for example data/ERP_CORE")
    zip_command = commands.add_parser("zip", help="build an NSG upload")
    zip_command.add_argument("output", type=Path)
    zip_command.add_argument("--include-data", type=Path, metavar="ROOT",
                             help="bundle verified files from a local copy (several GB for 40 participants)")
    zip_command.add_argument("--wheels", action="store_true",
                             help="bundle the pinned Python wheels for a job without internet access")
    for command in (fetch_command, zip_command):
        command.add_argument("--task", default="P3", choices=SOURCE["supported_tasks"])
        command.add_argument("--subjects", default="all", help="'all' or labels such as 001,002")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch(args.dataset_root, args.task, args.subjects)
        else:
            package(args.output, args.task, args.subjects, args.include_data, args.wheels)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
