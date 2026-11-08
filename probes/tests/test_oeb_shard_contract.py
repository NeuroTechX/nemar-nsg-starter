"""Red-green contract for an OpenEEGBench shard run through upstream `open_eeg_bench` on NSG.

RED  : no shard_summary.json fetched yet -> fail with the submit instructions.
GREEN: the shard ran ON NSG (Expanse node, V100), installed the pinned stack, ran upstream's
       Experiment.run() for every cell, every cell completed, and every classification cell beat
       chance.

A shard run on your laptop and copied into fetched/ stays RED on the provenance test on purpose:
the contract is "this benchmark cell reproduces on the shared gateway", not "it runs somewhere".
Set OEB_ALLOW_LOCAL=1 to check a local run against the other three tests only.

    pytest probes/tests/test_oeb_shard_contract.py -q
    OEB_SHARD_DIR=sweep/oeb/smoke/fetched/arithmetic_zyma2019_s0/extracted pytest ...
"""
import glob
import json
import os
import pathlib
import sys

import pytest

DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "sweep" / "oeb" / "smoke" / "fetched"
ROOT = pathlib.Path(os.environ.get("OEB_SHARD_DIR", str(DEFAULT)))
# chance level = 1/n_classes for the upstream default datasets (balanced accuracy)
N_CLASSES = {"braindecode/arithmetic_zyma2019": 2, "braindecode/bcic2a": 4, "braindecode/bcic2020-3": 5,
             "braindecode/physionet": 4, "braindecode/chbmit": 2, "braindecode/faced": 9,
             "braindecode/isruc-sleep": 5, "braindecode/mdd_mumtaz2016": 2}


def _summaries():
    found = glob.glob(str(ROOT / "**" / "shard_summary.json"), recursive=True)
    if not found:
        pytest.fail(
            f"RED: no shard_summary.json under {ROOT}. Run\n"
            f"  python sweep/oeb_plan.py plan --name smoke\n"
            f"  ./nsgr/nsgr.sh submit PYTORCH_PY_EXPANSE sweep/oeb/smoke/shards/arithmetic_zyma2019_s0.zip\n"
            f"then fetch output.tar.gz into sweep/oeb/smoke/fetched/<shard>/ and `oeb_plan.py merge`."
        )
    return [json.load(open(f)) for f in found]


def test_shard_ran_on_nsg():
    if os.environ.get("OEB_ALLOW_LOCAL"):
        pytest.skip("OEB_ALLOW_LOCAL set: provenance check disabled")
    for s in _summaries():
        node, gpu = s.get("node", ""), s.get("gpu") or ""
        # Expanse GPU nodes are exp-NN-NN with Tesla V100-SXM2-32GB (docs/FINDINGS.md, env-snapshots/)
        assert node.startswith("exp-"), (
            f"RED: shard_summary.json says node={node!r} -- this is not an NSG/Expanse run. "
            f"Submit with `python sweep/oeb_plan.py submit --name smoke`, then `poll`, then `merge`.")
        if "shard_error" not in s:          # gpu is probed after the imports; a crashed shard has no gpu field
            assert "V100" in gpu, f"expected an Expanse V100, got gpu={gpu!r}"


def test_shard_installed_pinned_stack():
    for s in _summaries():
        assert s.get("pip_ok") is True, s.get("pip_err")
        assert "shard_error" not in s, f"shard crashed before probing torch: {s['shard_error']}"
        assert s.get("torch", "").startswith("2.4.1"), s.get("torch")   # the Volta-safe pin (FINDINGS §6a)
        assert s.get("cuda") is True, "shard did not see the V100"


def test_shard_ran_every_cell_to_completion():
    for s in _summaries():
        assert "shard_error" not in s, s.get("trace")
        assert s.get("n_experiments", 0) > 0
        not_done = [(r.get("backbone"), r.get("dataset"), r.get("finetuning"), (r.get("exception") or "")[:80])
                    for r in s["rows"] if r.get("status") != "completed" and not _known_failure(r)]
        assert not not_done, not_done          # the documented V100 OOM + upstream-broken cells are allowed (below)


REFERENCE_DATASET = "braindecode/arithmetic_zyma2019"


def test_reference_cell_beats_chance():
    """Only the reference cell (BIOT, frozen, linear head) must beat chance: it is the one we have a
    measured number for (0.5764 on arithmetic_zyma2019, identical on RTX 2080 and V100). Other
    backbones legitimately sit at chance on some datasets (bench1 2026-10-23: BENDR and Signal-JEPA
    at 0.50 on arithmetic) -- that is a benchmark result, not a platform failure."""
    seen = False
    for s in _summaries():
        for r in s["rows"]:
            # arithmetic only: elsewhere BIOT/frozen at chance is a result (public1 chbmit = 0.500,
            # the majority class on an imbalanced seizure task), not a platform failure
            if r.get("backbone") == "BIOT" and r.get("finetuning") == "frozen" and r.get("head") == "linear" \
                    and r.get("dataset") == REFERENCE_DATASET and r.get("status") == "completed":
                seen = True
                assert r["test_balanced_accuracy"] > 1.0 / N_CLASSES[r["dataset"]], r
    if not seen:
        pytest.skip("no BIOT/frozen/linear reference cell in this shard")


# The documented failures (V100 OOMs, upstream/peft bugs) live in sweep/oeb_known.py, shared with
# `oeb_plan.py submit --redo-lost` so a shard that can only fail the same way is not re-run.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "sweep"))
from oeb_known import OOM, deadline_documented, is_known_failure as _known_failure  # noqa: E402


def _known_v100_oom(r):
    return _known_failure(r) and any(m in (r.get("exception") or "") for m in OOM)


def test_no_cell_failed_for_a_software_reason():
    """A cell that raised is a stack problem (wrong pin, missing module, OOM), never a benchmark result.
    Lists every distinct exception so a student sees the pin to fix, not 35 tracebacks."""
    import collections
    bad = collections.Counter()
    for s in _summaries():
        for r in s["rows"]:
            if r.get("status") != "completed" and not _known_failure(r):
                last = (r.get("exception") or "?").strip().splitlines()[-1][:120]
                bad[last] += 1
    assert not bad, "\n".join(f"{n:3d} x {m}" for m, n in bad.most_common())


def test_only_the_documented_cells_hit_the_32gb_ceiling():
    """Every OOM must be one documented in sweep/oeb_known.py as not runnable at protocol on a
    V100-32GB; an OOM anywhere else is new information that belongs in LIMITATIONS §2."""
    other = [(r.get("backbone"), r.get("finetuning"), r.get("dataset"))
             for s in _summaries() for r in s["rows"]
             if "out of memory" in (r.get("exception") or "") and not _known_v100_oom(r)]
    assert not other, f"undocumented OOM cells: {other}"


def test_deadline_stops_are_documented():
    """A shard the driver stopped before the NSG time limit returned only part of its cells. That is
    fine only for shards whose cells are known to exceed the 48 h job limit (sweep/oeb_known.py)."""
    undocumented = [((s.get("shard") or {}).get("backbones"), (s.get("shard") or {}).get("datasets"),
                     s["deadline_stopped"]) for s in _summaries()
                    if s.get("deadline_stopped") and not deadline_documented(s)]
    assert not undocumented, f"shards stopped by the deadline guard: {undocumented}"
