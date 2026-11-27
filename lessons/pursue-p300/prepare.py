#!/usr/bin/env python3
"""Fetch the pinned public recording or build a portal-ready NSG input ZIP."""

import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

from run import HERE, SOURCE, verify_dataset, verify_file


def fetch(root):
    dataset = SOURCE["openneuro_id"]
    revision = SOURCE["git_revision"]
    for relative, expected in SOURCE["sha256"].items():
        destination = root / relative
        if destination.exists():
            verify_file(destination, expected)
            continue
        if relative == SOURCE["recording"]:
            url = f"https://s3.amazonaws.com/openneuro.org/{dataset}/{relative}"
        else:
            url = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset}/{revision}/{relative}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            download = Path(temporary) / destination.name
            with urlopen(url, timeout=60) as response, download.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify_file(download, expected)
            # Publish only files which match the pinned snapshot.
            download.replace(destination)
    print(f"Verified the pinned recording and its sidecars under {root}")


def package(output, data):
    if data is not None:
        verify_dataset(data)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for path in ("run.py", "source.json"):
            archive.write(HERE / path, f"p300_erp/{path}")
        if data is not None:
            for relative in SOURCE["sha256"]:
                archive.write(data / relative, f"p300_erp/data/{SOURCE['openneuro_id']}/{relative}")
    print(f"Created {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_command = commands.add_parser("fetch", help="download only the pinned recording and sidecars")
    fetch_command.add_argument("dataset_root", type=Path, help="destination directory for ds007056")
    zip_command = commands.add_parser("zip", help="build an NSG upload")
    zip_command.add_argument("output", type=Path)
    zip_command.add_argument("--include-data", type=Path, help="bundle a verified local ds007056 subset")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch(args.dataset_root)
        else:
            package(args.output, args.include_data)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
