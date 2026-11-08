"""Known, documented OpenEEGBench cell failures (docs/FINDINGS.md §9b, LIMITATIONS.md §2).

One definition shared by the shard contract (probes/tests/test_oeb_shard_contract.py), which allows
these cells, and `oeb_plan.py submit --redo-lost`, which does not re-run a shard whose only
failures are listed here: re-running it would fail the same way. Every entry matches on the
exception text, so a different failure in the same cell is still new information.
"""

OOM = ("out of memory", "OutOfMemoryError")
ANY = None

# (backbone substring, strategies or ANY, dataset or ANY, exception substrings)
KNOWN = [
    # EEGPT with gradients through the backbone needs > 32 GB at upstream's protocol (batch 64): it
    # OOMs on the first backward on every V100-32GB (FINDINGS §8b/§8c).
    ("EEGPT", {"lora", "ia3", "adalora", "dora", "oft", "full", "full_finetune", "two_stages"}, ANY, OOM),
    # isruc_sleep is the largest public dataset: EEGPT OOMs even frozen there (public1, 2026-11-07).
    ("EEGPT", ANY, "braindecode/isruc-sleep", OOM),
    # LaBraM OOMs at every strategy, frozen included, on the three big datasets (isruc asks for
    # 35 GiB in one allocation) (public1, 2026-11-07).
    ("LaBraM", ANY, "braindecode/chbmit", OOM),
    ("LaBraM", ANY, "braindecode/faced", OOM),
    ("LaBraM", ANY, "braindecode/isruc-sleep", OOM),
    # REVE + DoRA on FACED; every other REVE cell fits (public1, 2026-10-31).
    ("REVE", {"dora"}, "braindecode/faced", OOM),
    # open-eeg-bench passes LaBraM's top-level nn.Parameter `temporal_embedding` as LoRA-family
    # target_parameters: peft 0.21.2 rejects top-level parameters, 0.17-0.18 reject the ParamWrapper
    # with lora_dropout=0.1 (upstream's default) -> no released peft runs it.
    ("LaBraM", {"lora", "dora", "adalora"}, ANY, ("Targeting an nn.Parameter on the top-level module",)),
    # peft OFT unfolds every 4-D input as if it fed a Conv2d; CBraMod's spectral_proj and REVE's
    # to_patch_embedding are Linear layers fed (B, C, patch, feat) -> empty unfold -> reshape fails.
    ("CBraMod", {"oft"}, ANY, ("is invalid for input of size 0",)),
    ("REVE", {"oft"}, ANY, ("is invalid for input of size 0",)),
]

# Shards whose cells are longer than the 48 h NSG limit allows in one job: the driver's deadline
# guard stops them and returns the finished cells (Signal-JEPA on isruc_sleep: up to 17.5 h/cell).
KNOWN_DEADLINE = {("signal_jepa", "isruc_sleep")}     # (shard.json backbone, dataset)


def is_known_failure(row):
    exc = row.get("exception") or ""
    bb = row.get("backbone") or ""
    return any(b in bb and (s is ANY or row.get("finetuning") in s)
               and (d is ANY or row.get("dataset") == d) and any(m in exc for m in msgs)
               for b, s, d, msgs in KNOWN)


def only_known_failures(summary):
    """True when a shard has rows and every non-completed row is a documented failure."""
    rows = summary.get("rows") or []
    return bool(rows) and all(r.get("status") == "completed" or is_known_failure(r) for r in rows)


def deadline_documented(summary):
    cfg = summary.get("shard") or {}
    return all((b, d) in KNOWN_DEADLINE for b in cfg.get("backbones", []) for d in cfg.get("datasets", []))
