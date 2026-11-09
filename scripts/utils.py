from datetime import datetime
import os
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
import brian2 as b


# getTimestamp
# Returns a previously determined timestamp, or the timestamp of the current point in time
# - refresh [optional]: if True, forcibly retrieves a new timestamp; else, only returns a new timestamp if no previous one is known
# - return: timestamp in the format YY-MM-DD_HH-MM-SS
def getTimestamp():
    return datetime.now().strftime("%y-%m-%d_%H-%M-%S")


# getFormattedTime
# Returns a string indicating the hours, minutes, and seconds of a time given in seconds
# - time_el: time in seconds
# - return: formatted time string
def getFormattedTime(time_el):
    if time_el < 1:
        time_el_str = "<1 s"
    elif time_el < 60:
        time_el_str = str(time_el) + " s"
    elif time_el < 3600:
        time_el_str = str(time_el // 60) + " m " + str(time_el % 60) + " s"
    else:
        time_el_str = (
            str(time_el // 3600)
            + " h "
            + str((time_el % 3600) // 60)
            + " m "
            + str((time_el % 3600) % 60)
            + " s"
        )
    return time_el_str


# getDataPath
# Consumes a general description for the simulation and a file description, and returns
# a path to a timestamped file in the output directory;  if no file description is provided,
# returns the path to the output directory
# - sim_description [optional]: general description for the simulation
# - file [optional]: specific name and extension for the currently considered file
# - return: path to a file in the output directory
def getDataPath(timestamp, sim_description, file=""):
    out_path = "simulations\\sim_" + timestamp

    if sim_description != "":
        out_path = out_path + "_" + sim_description
    if file == "":
        return out_path

    return out_path, os.path.join(out_path, file)


def make_output_file(out_dir: Path, timestamp: str, filename: str) -> Path:
    return f"out_dir/{timestamp}_{filename}"


# completeProt
# protocol after adding keys that are missing. This allows to leave out unnecessary keys
# when defining protocols in the JSON config file.
# - prot: stimulation protocol, possibly incomplete
# - return: complete stimulation protocol
def completeProt(prot):

    if prot.get("scheme") is None:
        prot["scheme"] = ""

    if prot.get("time_start") is None:
        prot["time_start"] = 0

    # if prot.get("duration") is None:
    # 	prot["duration"] = 0

    if prot.get("freq") is None:
        prot["freq"] = 0

    if prot.get("N_stim") is None:
        prot["N_stim"] = 0

    if prot.get("I_0") is None:
        prot["I_0"] = 0

    if prot.get("sigma_WN") is None:
        prot["sigma_WN"] = 0

    if (
        prot.get("explicit_input") is None
        or prot["explicit_input"].get("receivers") is None
        or prot["explicit_input"].get("stim_times") is None
    ):
        prot["explicit_input"] = {"receivers": [], "stim_times": []}

    return prot


def setup_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("simulation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if run twice in same session
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # file handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# protFormatted
# Takes a complete stimulation protocol dictionary and creates a formatted string for
# human-readable output.
# - prot: stimulation protocol
# - return: formatted string with information about the protocol
def protFormatted(prot):
    if not prot["scheme"]:
        fmstr = "none"
    elif prot["scheme"] == "EXPLICIT":
        fmstr = f"{prot['scheme']}, pulses at {prot['explicit_input']['stim_times']} ms, to neurons {prot['explicit_input']['receivers']}"
    elif prot["scheme"] == "STET":
        fmstr = f"{prot['scheme']} (Poisson) to neurons {prot['explicit_input']['receivers']}, starting at {prot['time_start']} s"
    elif prot["I_0"] != 0:
        fmstr = f"{prot['scheme']} (OU), I_0 = {prot['I_0']} nA, sigma_WN = {prot['sigma_WN']} nA s^1/2"
    # elif prot['duration'] != 0:
    # 	fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s, lasting for {prot['duration']} s"
    else:
        fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s"

    return fmstr


# get_explicit_stimulus
# Check and retrieve the specification of an explicit stimulus input (explicitly specified pulses for
# stimulation to defined receiving neurons, implemented below via SpikeGeneratorGroup)
# - prot: stimulation protocol dictionary (including the key "explicit_input")
# - return: dictionary of receiving neurons and stimulation pulse times if explicit protocol is given, `None` if not
def get_explicit_stimulus(prot):
    # Check scheme
    if prot["scheme"] != "EXPLICIT":
        return None
    # Check abundance of stimulation
    if (
        len(prot["explicit_input"]["receivers"]) <= 0
        or len(prot["explicit_input"]["stim_times"]) <= 0
    ):
        return None
    return prot["explicit_input"]


def getSubpopulations(N_pop, core, p_r):
    core_recall = core[: int(np.floor(float(p_r) * core.shape[0]))]
    core_norecall = core[int(np.floor(float(p_r) * core.shape[0])) :]
    control = np.delete(np.arange(N_pop), core)
    return core_recall, core_norecall, control


def mean_population_rate_in_window(rate_monitor, t0_s, width_ms=500):
    t_s = np.asarray(rate_monitor.t / b.second)
    r_hz = np.asarray(
        rate_monitor.smooth_rate(window="gaussian", width=width_ms * b.ms) / b.Hz
    )
    idx = np.argmin(np.abs(t_s - (t0_s + 0.1)))
    return r_hz[idx]


def mean_rate_in_window(rate_monitor, t_start_ms, t_end_ms):
    t_ms = np.array(rate_monitor.t / b.ms)
    rate_hz = np.array(rate_monitor.rate / b.Hz)
    mask = (t_ms >= t_start_ms * 1000) & (t_ms <= t_end_ms * 1000)
    if not np.any(mask):
        return np.nan
    return np.mean(rate_hz[mask])


def compute_Q_from_rate_monitors(
    rate_mon_as,
    rate_mon_ans,
    rate_mon_ctrl,
    t0_s,
    smoothing=True,
):
    if smoothing:
        v_as = mean_population_rate_in_window(rate_mon_as, t0_s)
        v_ans = mean_population_rate_in_window(rate_mon_ans, t0_s)
        v_ctrl = mean_population_rate_in_window(rate_mon_ctrl, t0_s)
    else:
        v_as = mean_rate_in_window(rate_mon_as, t0_s, t0_s + 0.1)
        v_ans = mean_rate_in_window(rate_mon_ans, t0_s, t0_s + 0.1)
        v_ctrl = mean_rate_in_window(rate_mon_ctrl, t0_s, t0_s + 0.1)
    Q = (v_ans - v_ctrl) / v_as
    return (
        v_as,
        v_ans,
        v_ctrl,
        Q,
    )


def sample_ids(ids, k):
    if len(ids) <= k:
        return ids
    return np.random.choice(ids, size=k, replace=False)


def make_learning_monitors(
    neur_pop_exc,
    neur_pop_inh,
    pattern1_neurons,
    control_synapse_ids,
    pattern1_synapse_ids,
    pattern1_outgoing_synapse_ids,
    syn_exc_exc,
    max_neuron,
    max_syn,
):
    pattern1_neurons = np.array(pattern1_neurons)
    record_pattern1 = sample_ids(pattern1_synapse_ids, max_syn)
    record_pattern1_out = sample_ids(pattern1_outgoing_synapse_ids, max_syn)
    record_control = sample_ids(control_synapse_ids, max_syn)
    return {
        "spike_mon_exc": b.SpikeMonitor(neur_pop_exc),
        "spike_mon_inh": b.SpikeMonitor(neur_pop_inh),
        "syn_state_mon_pattern1": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern1
        ),
        "syn_state_mon_pattern1_outgoing": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern1_out
        ),
        "syn_state_mon_control": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_control
        ),
        "rate_mon_exc": b.PopulationRateMonitor(neur_pop_exc),
        "rate_mon_inh": b.PopulationRateMonitor(neur_pop_inh),
    }


def make_recall_monitors(
    assembly_ids,
    neur_pop_exc,
    neur_pop_inh,
    pattern1_synapse_ids,
    pattern2_synapse_ids,
    pattern1_outgoing_synapse_ids,
    pattern2_outgoing_synapse_ids,
    syn_exc_exc,
    max_neuron,
    max_syn,
):
    assembly_ids = np.array(assembly_ids)
    record_pattern1 = sample_ids(pattern1_synapse_ids, max_syn)
    record_pattern1_out = sample_ids(pattern1_outgoing_synapse_ids, max_syn)
    record_pattern2 = sample_ids(pattern2_synapse_ids, max_syn)
    record_pattern2_out = sample_ids(pattern2_outgoing_synapse_ids, max_syn)

    return {
        "spike_mon_exc": b.SpikeMonitor(neur_pop_exc),
        "spike_mon_inh": b.SpikeMonitor(neur_pop_inh),
        "syn_state_mon_pattern1": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern1
        ),
        "syn_state_mon_pattern1_outgoing": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern1_out
        ),
        "syn_state_mon_pattern2": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern2
        ),
        "syn_state_mon_pattern2_outgoing": b.StateMonitor(
            syn_exc_exc, ["Ca", "h", "z"], record=record_pattern2_out
        ),
        "rate_mon_exc": b.PopulationRateMonitor(neur_pop_exc),
        "rate_mon_inh": b.PopulationRateMonitor(neur_pop_inh),
    }


def mean_rate_from_spike_monitor(spike_monitor, neuron_ids, t0_s, window_s=0.5):
    spike_times = np.asarray(spike_monitor.t / b.second)
    spike_ids = np.asarray(spike_monitor.i)
    neuron_ids = np.asarray(neuron_ids)

    n_neurons = len(neuron_ids)
    if n_neurons == 0:
        return np.nan

    mask = np.isin(spike_ids, neuron_ids)
    st = spike_times[mask]

    half = window_s / 2.0
    n_spikes = np.sum((st >= t0_s - half) & (st < t0_s + half))

    return n_spikes / (n_neurons * window_s)


def compute_Q_from_spike_monitor(
    spike_monitor, as_ids, ans_ids, ctrl_ids, t0_s, window_s=0.5
):
    v_as = mean_rate_from_spike_monitor(spike_monitor, as_ids, t0_s + 0.1, window_s)
    v_ans = mean_rate_from_spike_monitor(spike_monitor, ans_ids, t0_s + 0.1, window_s)
    v_ctrl = mean_rate_from_spike_monitor(spike_monitor, ctrl_ids, t0_s + 0.1, window_s)

    Q = (v_ans - v_ctrl) / v_as if v_as > 0 else np.nan
    return v_as, v_ans, v_ctrl, Q


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
