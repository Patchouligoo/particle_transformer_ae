"""
h5 skims -> flat DataFrame, one row per event.

Same conventions as the HAXAD demonstrator (event weight, 105-145 GeV diphoton mass window, legacy
column names such as jet1_pt / el1_pt / mu1_pt), except that the FIRST N events of each file are
taken instead of a random sample: the files carry no kinematic ordering, and a contiguous read is
cheap and reproducible.
"""
import logging

import numpy as np
import pandas as pd

from configs.file_dict import process_dict

logger = logging.getLogger(__name__)

MEASUREMENT_REGION_MIN = 105000.0  # MeV
MEASUREMENT_REGION_MAX = 145000.0
LUMINOSITY = 470.0  # same constant as the event weight of the HAXAD demonstrator

# Columns read from skimmed.h5, with the names stored in the file (renamed by rename_columns afterwards).
OBJECT_COLUMNS = (
    [f"sel_photon{i}_{f}" for i in (1, 2) for f in ("pt", "eta", "phi")]
    + [f"sel_jet{i}_{f}" for i in (1, 2, 3, 4) for f in ("pt", "eta", "phi", "btag")]
    + [f"sel_fatjet{i}_{f}" for i in (1, 2) for f in ("pt", "eta", "phi", "m", "tau32", "tau43")]
    + [f"sel_electron{i}_{f}" for i in (1, 2) for f in ("pt", "eta", "phi")]
    + [f"sel_muon{i}_{f}" for i in (1, 2) for f in ("pt", "eta", "phi")]
    + ["met_pt", "met_phi"]
)
EVENT_COLUMNS = ["diphoton_mass", "diphoton_pt", "diphoton_delta_R"]
WEIGHT_COLUMNS = ["event_weight", "pythia_xsec [fb]", "pythia_filter_efficiency", "sumw_presel"]
columns = OBJECT_COLUMNS + EVENT_COLUMNS + WEIGHT_COLUMNS


def rename_columns(dataframe):
    """sel_photon1_pt -> photon1_pt, sel_electron1_pt -> el1_pt, sel_muon2_phi -> mu2_phi; others unchanged."""

    def legacy_name(column):
        return column.replace("sel_", "").replace("electron", "el").replace("muon", "mu")

    return dataframe.rename(columns=legacy_name)


def _load_first_events(file_path, num_events):
    """Read the file from the start, in chunks, until `num_events` events inside the mass window are
    collected. Returns (dataframe, rows_read, rows_in_file), where rows_read is the file position of
    the last kept event + 1: the weight rescaling needs it."""
    frames, kept = [], 0
    with pd.HDFStore(file_path, mode="r") as store:
        key = store.keys()[0]  # one table per file: /events
        rows_in_file = store.get_storer(key).nrows
        chunk_rows = int(np.clip(2 * num_events, 10_000, 500_000))  # a read materializes all ~200 columns
        for start in range(0, rows_in_file, chunk_rows):
            chunk = store.select(key, start=start, stop=min(start + chunk_rows, rows_in_file), columns=columns)
            in_window = chunk["diphoton_mass"].between(MEASUREMENT_REGION_MIN, MEASUREMENT_REGION_MAX)
            positions = np.flatnonzero(in_window)[: num_events - kept]
            frames.append(chunk.iloc[positions])
            kept += len(positions)
            if kept == num_events:
                return pd.concat(frames, ignore_index=True), start + positions[-1] + 1, rows_in_file
    logger.warning(f"{file_path}: only {kept} events in the mass window, {num_events} were requested")
    return pd.concat(frames, ignore_index=True), rows_in_file, rows_in_file


def process_delphes_events(process_name, num_events_per_process=100_000):
    """The first `num_events_per_process` events of `process_name` inside the diphoton mass window.

    Returns a float32 DataFrame with one row per event: the object columns (photon1_pt, jet3_btag,
    el1_eta, met_phi, ...; NaN where the object does not exist), diphoton_mass / diphoton_pt /
    diphoton_delta_R, the event weight (luminosity x cross section, rescaled so that the loaded
    events still represent the whole file) and process_id (see configs/file_dict.py).
    """
    process = process_dict[process_name]
    dataframe, rows_read, rows_in_file = _load_first_events(process["file"], num_events_per_process)
    dataframe["event_weight"] = (
        LUMINOSITY
        * dataframe["pythia_xsec [fb]"]
        * dataframe["event_weight"]
        * dataframe["pythia_filter_efficiency"].replace([np.inf, -np.inf], 1)
        / dataframe["sumw_presel"]
        * (rows_in_file / rows_read)
    )
    dataframe["process_id"] = process["process_id"]
    dataframe = dataframe.drop(columns=["pythia_xsec [fb]", "pythia_filter_efficiency", "sumw_presel"])
    return rename_columns(dataframe.astype("float32"))
