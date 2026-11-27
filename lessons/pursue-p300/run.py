#!/usr/bin/env python3
"""Compute target and standard ERPs from one pinned PURSUE EEG recording."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text(encoding="utf-8"))
CONDITIONS = {"target": 1, "standard": 2}
LABELS = {"Rare_Target": "target", "Frequent_NonTarget": "standard"}
EPOCH = (-0.2, 0.8)
BASELINE = (-0.2, 0.0)
FILTER_HZ = (0.5, 30.0)
REJECT_UV = 150.0
WINDOW_MS = (300, 600)


def verify_file(path, expected):
    if not path.is_file():
        raise ValueError(f"Missing input or unmaterialized git-annex file: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path} differs from the pinned OpenNeuro snapshot")


def verify_dataset(root):
    for relative, expected in SOURCE["sha256"].items():
        verify_file(root / relative, expected)


def dataset_location(explicit):
    if explicit is not None:
        return explicit, "local"
    bundled = HERE / "data" / SOURCE["openneuro_id"]
    if bundled.is_dir():
        return bundled, "bundle"
    if nemar_path := os.environ.get("NEMARPATH"):
        return Path(nemar_path) / SOURCE["openneuro_id"], "NEMARPATH"
    raise ValueError("Set NEMARPATH, bundle the pinned files, or pass --dataset-root")


def read_stimuli(root):
    event_dict = json.loads((root / SOURCE["event_dictionary"]).read_text(encoding="utf-8"))
    levels = event_dict["value"]["Levels"]
    with (root / SOURCE["events"]).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return [
            (int(row["sample"]), LABELS[levels[row["value"]]])
            for row in rows
            if levels.get(row["value"]) in LABELS
        ]


def write_tsv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def analyze(root, input_location, out, channel, stimuli, counts):
    # The CPU Python on Expanse environment supplies these packages.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import scipy

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_eeglab(root / SOURCE["recording"], preload=True, eog="auto")
    raw.pick("eeg")
    if channel not in raw.ch_names or channel in raw.info["bads"]:
        raise ValueError(f"Choose an available scalp EEG channel: {', '.join(raw.ch_names)}")

    for annotation in raw.annotations:
        if annotation["description"].casefold() == "boundary":
            # Make the discontinuity visible to both filtering and epoch rejection.
            raw.annotations.rename({annotation["description"]: "BAD boundary"})
            break
    raw.filter(*FILTER_HZ, skip_by_annotation=("edge", "bad_acq_skip", "bad boundary"))
    events = np.array([(sample + raw.first_samp, 0, CONDITIONS[label]) for sample, label in stimuli])
    epochs = mne.Epochs(
        raw, events, event_id=CONDITIONS, tmin=EPOCH[0], tmax=EPOCH[1], baseline=BASELINE,
        reject={"eeg": REJECT_UV * 1e-6}, preload=True, reject_by_annotation=True,
    )
    kept = {label: len(epochs[label]) for label in CONDITIONS}
    if min(kept.values()) == 0:
        raise ValueError(f"No retained epochs in at least one condition: {kept}")
    # Screen high-amplitude trials in the recording's reference before re-referencing.
    epochs.set_eeg_reference("average")

    electrode = epochs.ch_names.index(channel)
    target, standard = (
        epochs[label].average().data[electrode] * 1e6 for label in CONDITIONS
    )
    times_ms = epochs.times * 1000
    difference = target - standard
    in_window = (times_ms >= WINDOW_MS[0]) & (times_ms <= WINDOW_MS[1])
    drop_reasons = {
        label: dict(sorted(Counter(
            reason for (_, condition), reasons in zip(stimuli, epochs.drop_log, strict=True)
            if condition == label for reason in reasons
        ).items()))
        for label in CONDITIONS
    }

    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "trial-counts.tsv", ("condition", "found", "retained", "dropped"), (
        (label, counts[label], kept[label], counts[label] - kept[label]) for label in CONDITIONS
    ))
    write_tsv(out / "erp.tsv", ("time_ms", "target_uv", "standard_uv", "target_minus_standard_uv"), (
        (f"{t:g}", f"{a:.6f}", f"{b:.6f}", f"{diff:.6f}")
        for t, a, b, diff in zip(times_ms, target, standard, difference, strict=True)
    ))

    fig, (ax, contrast) = plt.subplots(
        2, 1, sharex=True, height_ratios=(2, 1), figsize=(8, 5.5), layout="constrained",
    )
    for axis in (ax, contrast):
        axis.axvspan(*WINDOW_MS, color="#E3EBF4")
        axis.axvline(0, color="#505050", linewidth=0.8)
        axis.axhline(0, color="#505050", linewidth=0.6)
    ax.plot(times_ms, target, color="#2351A0", label=f"Target (n={kept['target']})")
    ax.plot(times_ms, standard, color="#A24A1D", linestyle="--", label=f"Standard (n={kept['standard']})")
    ax.set(ylabel="Amplitude (µV)", title=f"PURSUE visual oddball · sub-{SOURCE['subject']} · {channel}")
    ax.legend(frameon=False, loc="best")
    contrast.plot(times_ms, difference, color="#553480")
    contrast.set(xlabel="Time from stimulus onset (ms)", ylabel="Difference (µV)",
                 title="Target minus standard · 300–600 ms shaded")
    fig.savefig(out / "erp.png", dpi=130)
    plt.close(fig)

    manifest = {
        "dataset": SOURCE,
        "input_location": input_location,
        "environment": {
            "python": platform.python_version(), "mne": mne.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
        "analysis": {
            "channel": channel, "scalp_eeg_channels": len(raw.ch_names),
            "sample_rate_hz": raw.info["sfreq"], "erp_reference": "average of scalp EEG channels",
            "rejection_reference": "as recorded",
            "epoch_s": EPOCH, "baseline_s": BASELINE, "filter_hz": FILTER_HZ,
            "reject_eeg_peak_to_peak_uv": REJECT_UV, "measurement_window_ms": WINDOW_MS,
        },
        "found": dict(counts),
        "retained": kept,
        "drop_reasons": drop_reasons,
        "target_minus_standard_mean_uv": round(float(difference[in_window].mean()), 6),
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote ERP output to {out} ({kept['target']} target, {kept['standard']} standard epochs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="path to the directory ds007056")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--channel", default="Pz")
    parser.add_argument("--preflight", action="store_true", help="check dataset files and count stimuli")
    args = parser.parse_args()
    try:
        root, input_location = dataset_location(args.dataset_root)
        verify_dataset(root)
        stimuli = read_stimuli(root)
        counts = Counter(label for _, label in stimuli)
        if not all(counts[label] for label in CONDITIONS):
            raise ValueError(f"Expected both stimulus conditions, found {dict(counts)}")
        if args.preflight:
            print(json.dumps({"dataset": SOURCE["openneuro_id"], "input_location": input_location,
                              "verified_stimuli": dict(counts)}, indent=2))
        else:
            analyze(root, input_location, args.output, args.channel, stimuli, counts)
    except (OSError, ValueError, ImportError) as exc:
        sys.exit(f"P300 job could not finish: {exc}. See the lesson README for setup and bundling.")


if __name__ == "__main__":
    main()
