"""MNE-BIDS-Pipeline configuration for one ERP CORE task.

Adapted from the ERP CORE example that ships with MNE-BIDS-Pipeline 1.10.1
(mne_bids_pipeline/tests/configs/config_ERP_CORE.py, BSD-3-Clause). The changes
follow the published ERP CORE methods more closely (Kappenman et al., 2021,
https://doi.org/10.1016/j.neuroimage.2020.117465): all 40 participants, 256 Hz,
0.1 Hz high-pass, no low-pass, reference and epoch windows from Table 1, and only
trials with a correct response between 200 and 1000 ms.

run.py sets the environment variables read below. You can also run it by hand:

    ERPCORE_BIDS_ROOT=data/ERP_CORE ERPCORE_DERIV_ROOT=derivatives \
    ERPCORE_TASK=P3 mne_bids_pipeline --config config_erpcore.py --steps preprocessing,sensor
"""

import json
import os
from pathlib import Path

import mne

_SOURCE = json.loads((Path(__file__).resolve().parent / "source.json").read_text(encoding="utf-8"))
task = os.environ.get("ERPCORE_TASK", "P3")
if task not in _SOURCE["supported_tasks"]:
    raise RuntimeError(f"Task {task} is not configured yet. Supported: {_SOURCE['supported_tasks']}")
_SPEC = _SOURCE["components"][task]

bids_root = os.environ["ERPCORE_BIDS_ROOT"]
deriv_root = os.environ["ERPCORE_DERIV_ROOT"]
sessions = [task]
subjects = os.environ.get("ERPCORE_SUBJECTS", "all").split(",")
if subjects == ["all"]:
    subjects = "all"

ch_types = ["eeg"]
interactive = False
read_raw_bids_verbose = "error"  # the .set files name their .fdt with an older file name

# ERP CORE recorded 30 scalp channels plus HEOG_left, HEOG_right and VEOG_lower.
# The paper's bipolar EOG: HEOG = left minus right, VEOG = lower minus FP2.
eeg_template_montage = mne.channels.make_standard_montage("standard_1005")
eeg_bipolar_channels = {
    "HEOG": ("HEOG_left", "HEOG_right"),
    "VEOG": ("VEOG_lower", "FP2"),
}
drop_channels = ["HEOG_left", "HEOG_right", "VEOG_lower"]
eog_channels = ["HEOG", "VEOG"]
eeg_reference = _SPEC["reference"]

raw_resample_sfreq = 256  # the paper downsampled to 256 Hz
l_freq = 0.1  # the paper used a 0.1 Hz high-pass
h_freq = None  # mean amplitudes in the paper were measured without a low-pass
notch_freq = None

find_breaks = True
min_break_duration = 10
t_break_annot_start_after_previous_event = 3.0
t_break_annot_stop_before_next_event = 1.5

# Eye blinks and eye movements: ICA on a 1 Hz high-passed copy, components matched to EOG.
spatial_filter = "ica"
ica_algorithm = "picard-extended_infomax"
ica_l_freq = 1.0
ica_reject = {"eeg": 350e-6, "eog": 500e-6}  # only for fitting ICA, as in the MNE example
ica_n_components = 0.999
ica_max_iterations = 1000
ica_decim = 2
ica_eog_threshold = 2.0  # as in the MNE example; at the default 3.0 some participants had no blink component found

# The paper set a voltage threshold for each participant. autoreject's global
# threshold is the closest automated equivalent.
reject = "autoreject_global"

rename_events = _SPEC["rename_events"]
on_rename_missing_events = "ignore"
epochs_tmin, epochs_tmax = _SPEC["epoch_s"]
baseline = tuple(_SPEC["baseline_s"])

# Keep correct trials with a response from 200 to 1000 ms, as in the paper.
# Trials removed here appear as "metadata query" in the epochs drop log.
epochs_metadata_tmin = 0.0
epochs_metadata_tmax = 1.5
epochs_metadata_keep_first = ["response"]
epochs_metadata_query = "first_response == 'correct' and response >= 0.2 and response <= 1.0"
conditions = list(_SPEC["conditions"])
contrasts = [tuple(_SPEC["contrast"])]

# Sensor-space ERPs only. Decoding and source estimates are not part of this lesson.
decode = False
run_source_estimation = False
process_raw_clean = False

n_jobs = int(os.environ.get("ERPCORE_N_JOBS", "4"))
parallel_backend = "loky"
