#!/usr/bin/env python3
"""Measure one ERP CORE component from MNE-BIDS-Pipeline output and compare it with the paper.

run.py calls this after the pipeline. It needs MNE-Python, NumPy and Matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text(encoding="utf-8"))
RT_WINDOW_S = (0.2, 1.0)
# Approximate versions of the paper's participant exclusion rules (Kappenman et al. 2021, 2.5).
MAX_ARTIFACT_RATE = 0.25
MIN_ACCURACY = 0.75
MIN_RETAINED_PER_CONDITION = 0.5


def behaviour(events_tsv, spec):
    """Count stimuli, correct responses and usable trials per condition from events.tsv."""
    rename = spec["rename_events"]
    labels = {full: name for name, full in spec["conditions"].items()}
    with events_tsv.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream, delimiter="\t")]
    counts = {name: {"stimuli": 0, "correct": 0, "usable": 0} for name in spec["conditions"]}
    for i, row in enumerate(rows):
        renamed = rename.get(f"{row['trial_type']}/{row['value']}")
        if renamed not in labels:
            continue
        onset = float(row["onset"])
        first = next((r for r in rows[i + 1:] if r["trial_type"] == "response"
                      and 0 < float(r["onset"]) - onset <= 1.5), None)
        tally = counts[labels[renamed]]
        tally["stimuli"] += 1
        if first is not None and rename.get(f"response/{first['value']}") == "response/correct":
            tally["correct"] += 1
            if RT_WINDOW_S[0] <= float(first["onset"]) - onset <= RT_WINDOW_S[1]:
                tally["usable"] += 1
    return counts


def published_subject_values(spec):
    """Per-participant difference-wave amplitudes from the ERP CORE measurement file."""
    info = spec["published"].get("subject_values")
    if not info:
        return {}
    values = {}
    for line in (HERE / info["file"]).read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            amplitude, erpset = line.split()
            values[f"{int(erpset.split('_')[0]):03d}"] = float(amplitude)
    return values


def window_mean(evoked, channel, window_ms, delay_ms=0):
    picked = evoked.copy().pick([channel])
    times = picked.times * 1000 - delay_ms  # time from the stimulus on the screen
    mask = (times >= window_ms[0]) & (times <= window_ms[1])
    return float(picked.data[0, mask].mean() * 1e6)


def summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"n": int(len(values)), "mean_uv": round(float(values.mean()), 3), "sd_uv": None, "dz": None}
    sd = float(values.std(ddof=1))
    return {"n": int(len(values)), "mean_uv": round(float(values.mean()), 3),
            "sd_uv": round(sd, 3), "dz": round(float(values.mean()) / sd, 3)}


def compare_subjects(rows):
    """Agreement with the paper's own value for each participant both analyses kept."""
    both = [r for r in rows if r["paper_difference_uv"] != "" and not r["excluded"]]
    if len(both) < 3:
        return {"n": len(both)}
    ours = np.array([r["difference_uv"] for r in both])
    paper = np.array([r["paper_difference_uv"] for r in both])
    return {"n": len(both), "pearson_r": round(float(np.corrcoef(ours, paper)[0, 1]), 3),
            "mean_difference_uv": round(float((ours - paper).mean()), 3),
            "mean_absolute_difference_uv": round(float(np.abs(ours - paper).mean()), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="P3")
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", required=True)
    args = parser.parse_args()
    mne.set_log_level("WARNING")

    task, spec = args.task, SOURCE["components"][args.task]
    channel, window = spec["channel"], spec["window_ms"]
    # Events mark the trigger. The paper shifted them by the measured monitor delay, so
    # times here are relative to when the stimulus appeared on the screen.
    delay = spec.get("monitor_delay_ms", 0)
    first, second = spec["contrast"]
    paper_values = published_subject_values(spec)
    paper_excluded = {k for k in spec["published"].get("excluded_subjects", {}) if k != "source"}
    rows, waves, missing = [], {}, []
    for subject in args.subjects.split(","):
        name = f"sub-{subject}_ses-{task}_task-{task}"
        ave = args.deriv_root / f"sub-{subject}" / f"ses-{task}" / "eeg" / f"{name}_ave.fif"
        if not ave.is_file():
            missing.append(subject)
            continue
        by_comment = {e.comment: e for e in mne.read_evokeds(ave)}
        # The pipeline labels each evoked with the full event name, e.g. "stimulus/target".
        evokeds = {c: by_comment.get(full, by_comment.get(c)) for c, full in spec["conditions"].items()}
        counts = behaviour(args.bids_root / f"sub-{subject}" / f"ses-{task}" / "eeg" / f"{name}_events.tsv", spec)
        kept = {c: evokeds[c].nave for c in spec["conditions"]}
        stimuli = sum(c["stimuli"] for c in counts.values())
        usable = sum(c["usable"] for c in counts.values())
        accuracy = sum(c["correct"] for c in counts.values()) / stimuli
        artifact_rate = 1 - sum(kept.values()) / usable
        reasons = []
        if artifact_rate > MAX_ARTIFACT_RATE:
            reasons.append(f"artifacts {artifact_rate:.0%}")
        if accuracy < MIN_ACCURACY:
            reasons.append(f"accuracy {accuracy:.0%}")
        for condition, tally in counts.items():
            if kept[condition] < MIN_RETAINED_PER_CONDITION * tally["stimuli"]:
                reasons.append(f"{condition} retained {kept[condition]}/{tally['stimuli']}")
        amplitude = {c: window_mean(evokeds[c], channel, window, delay) for c in spec["conditions"]}
        difference = mne.combine_evoked([evokeds[first], evokeds[second]], weights=[1, -1])
        waves[subject] = {c: evokeds[c].copy().pick([channel]).data[0] * 1e6 for c in spec["conditions"]}
        waves[subject]["difference"] = difference.copy().pick([channel]).data[0] * 1e6
        times_ms = evokeds[first].times * 1000 - delay
        rows.append({
            "subject": subject, "excluded": bool(reasons), "exclusion_reasons": "; ".join(reasons),
            "accuracy": round(accuracy, 3), "artifact_rate": round(artifact_rate, 3),
            **{f"{c}_stimuli": counts[c]["stimuli"] for c in spec["conditions"]},
            **{f"{c}_usable": counts[c]["usable"] for c in spec["conditions"]},
            **{f"{c}_retained": kept[c] for c in spec["conditions"]},
            **{f"{c}_uv": round(amplitude[c], 3) for c in spec["conditions"]},
            "difference_uv": round(window_mean(difference, channel, window, delay), 3),
            "paper_excluded": subject in paper_excluded,
            "paper_difference_uv": paper_values.get(subject, ""),
        })
    if not rows:
        raise SystemExit("No per-participant ERPs were found in the pipeline output")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "subject-amplitudes.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    included = [r["subject"] for r in rows if not r["excluded"]] or [r["subject"] for r in rows]
    traces = {k: np.mean([waves[s][k] for s in included], axis=0) for k in [*spec["conditions"], "difference"]}
    sem = np.std([waves[s]["difference"] for s in included], axis=0, ddof=1) / np.sqrt(len(included)) \
        if len(included) > 1 else np.zeros_like(times_ms)
    with (out / "grand-average.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["time_ms", *[f"{k}_uv" for k in traces], "difference_sem_uv"])
        for i, t in enumerate(times_ms):
            writer.writerow([f"{t:g}", *[f"{traces[k][i]:.4f}" for k in traces], f"{sem[i]:.4f}"])

    published = spec["published"]
    ours_included = summary([r["difference_uv"] for r in rows if r["subject"] in included])
    ours_all = summary([r["difference_uv"] for r in rows])
    comparison = {
        "component": task, "channel": channel, "window_ms": window, "contrast": f"{first} minus {second}",
        "published": published, "this_run_after_exclusions": ours_included, "this_run_all_participants": ours_all,
        "excluded": {r["subject"]: r["exclusion_reasons"] for r in rows if r["excluded"]},
        "paper_excluded": sorted(paper_excluded & {r["subject"] for r in rows}),
        "per_participant_vs_paper": compare_subjects(rows),
        "missing_pipeline_output": missing,
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(3, 1), height_ratios=(2, 1))
    ax, contrast, dots = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    contrast.sharex(ax)
    for axis in (ax, contrast):
        axis.axvspan(*window, color="#E3EBF4")
        axis.axvline(0, color="#505050", linewidth=0.8)
        axis.axhline(0, color="#505050", linewidth=0.6)
    styles = ({"color": "#2351A0"}, {"color": "#A24A1D", "linestyle": "--"})
    for (condition, style) in zip(spec["conditions"], styles):
        ax.plot(times_ms, traces[condition], label=condition, **style)
    ax.set(ylabel="Amplitude (µV)",
           title=f"ERP CORE {task} · grand average of {len(included)} participants · {channel}")
    ax.legend(frameon=False, loc="upper left")
    contrast.fill_between(times_ms, traces["difference"] - sem, traces["difference"] + sem, color="#D9CCE8")
    contrast.plot(times_ms, traces["difference"], color="#553480")
    contrast.set(xlabel=f"Time from stimulus onset (ms, triggers shifted {delay} ms)", ylabel="Difference (µV)",
                 title=f"{first} minus {second} ± 1 SEM · {window[0]}–{window[1]} ms shaded")
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(rows))
    values = np.array([r["difference_uv"] for r in rows])
    excluded = np.array([r["excluded"] for r in rows])
    dots.scatter(0 + jitter[~excluded], values[~excluded], s=14, color="#553480", label="included")
    dots.scatter(0 + jitter[excluded], values[excluded], s=14, facecolor="none", edgecolor="#553480", label="excluded")
    dots.errorbar([0.35], [ours_included["mean_uv"]], yerr=[ours_included["sd_uv"] or 0], fmt="o", color="#553480", capsize=4)
    dots.errorbar([0.7], [published["difference_mean_amplitude_uv"]], yerr=[published["difference_sd_uv"]],
                  fmt="s", color="#505050", capsize=4)
    dots.set_xticks([0, 0.35, 0.7], ["each\nparticipant", f"this run\nN={ours_included['n']}",
                                     f"paper\nN={published['n']}"])
    dots.set_xlim(-0.25, 0.95)
    dots.axhline(0, color="#505050", linewidth=0.6)
    dots.set(ylabel=f"Mean amplitude {window[0]}–{window[1]} ms (µV)", title="Mean ± SD")
    dots.legend(frameon=False, loc="lower right", fontsize="small")
    fig.savefig(out / "grand-average.png", dpi=130)
    plt.close(fig)

    both = [r for r in rows if r["paper_difference_uv"] != ""]
    if both:
        fig, ax = plt.subplots(figsize=(4.8, 4.6), layout="constrained")
        paper = np.array([r["paper_difference_uv"] for r in both])
        ours = np.array([r["difference_uv"] for r in both])
        kept = np.array([not r["excluded"] for r in both])
        low, high = min(paper.min(), ours.min()) - 1, max(paper.max(), ours.max()) + 1
        ax.plot([low, high], [low, high], color="#505050", linewidth=0.8, linestyle=":")
        ax.scatter(paper[kept], ours[kept], s=16, color="#553480", label="included here")
        ax.scatter(paper[~kept], ours[~kept], s=16, facecolor="none", edgecolor="#553480", label="excluded here")
        ax.set(xlabel="Paper, ERPLAB (µV)", ylabel="This run, MNE-BIDS-Pipeline (µV)",
               title=f"{task} difference at {channel}, one dot per participant", xlim=(low, high), ylim=(low, high))
        ax.legend(frameon=False, loc="upper left", fontsize="small")
        fig.savefig(out / "vs-paper.png", dpi=130)
        plt.close(fig)

    import mne_bids_pipeline
    import scipy
    manifest = {
        "dataset": {k: SOURCE[k] for k in ("dataset", "paper_doi", "data_doi", "osf_bids_component", "license")},
        "task": task, "participants_requested": len(args.subjects.split(",")),
        "participants_processed": len(rows), "participants_included": len(included),
        "environment": {"python": platform.python_version(), "mne_bids_pipeline": mne_bids_pipeline.__version__,
                        "mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                        "matplotlib": matplotlib.__version__},
        "measurement": {"channel": channel, "window_ms": window, "rt_window_s": RT_WINDOW_S,
                        "monitor_delay_ms": delay,
                        "exclusion": {"max_artifact_rate": MAX_ARTIFACT_RATE, "min_accuracy": MIN_ACCURACY,
                                      "min_retained_per_condition": MIN_RETAINED_PER_CONDITION}},
        "result": comparison,
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{task}: {ours_included['n']} participants, {first} minus {second} at {channel} "
          f"{window[0]}-{window[1]} ms = {ours_included['mean_uv']} µV (SD {ours_included['sd_uv']}); "
          f"paper {published['difference_mean_amplitude_uv']} µV (SD {published['difference_sd_uv']}, "
          f"N={published['n']})")


if __name__ == "__main__":
    main()
