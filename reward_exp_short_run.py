#!/bin/python3

# Brian simulation of memory consolidation in recurrent spiking neural networks, consisting of leaky integrate-and-fire neurons that are
# connected via current-based, plastic synapses (early phase based on calcium dynamics and late phase based on synaptic tagging and capture).
# Reproduces the results of Luboeinski and Tetzlaff, Commun. Biol., 2021 (https://doi.org/10.1038/s42003-021-01778-y).

# (c) Jannik Luboeinski 2022-2024
# Contact: mail[at]jlubo.net

import brian2 as b
from brian2.units import *
import numpy as np
import json
import os
import time
from datetime import datetime
import argparse
import pickle

from scripts.functions import *
from scripts.utils import *
from scripts.patterns import *

# b.set_device("cpp_standalone")


#####################################
# Runs simulation of a recurrent neural network with consolidation dynamics
# - config: configuration of model and simulation parameters (as a dictionary from JSON format)
# - neuron_observables: list of the neuronal observables to be monitored
# - synapse_observables: list of the synaptic observables to be monitored
def run(config):
    ###################################################
    ######## Set Parameters ###########################
    ###################################################

    minute = 60 * second
    hour = 60 * minute

    delta_t = config["simulation"]["dt"] * msecond
    b.defaultclock.dt = delta_t

    s_desc = config["simulation"][
        "short_description"
    ]  # short description of the simulation

    # Setup Logger
    # timestamp, logger, out_path, log_path = setup_logger()
    timestamp = getTimestamp()
    out_path, log_path = getDataPath(timestamp, sim_description=s_desc, file="log.txt")
    q_values_file = getDataPath(timestamp, sim_description=s_desc, file="q_values.csv")[
        1
    ]
    os.makedirs(out_path, exist_ok=True)
    logger = setup_logger(log_path)

    neuron_ids = np.array(
        config["simulation"]["sample_neuron_ids"]
    )  # identifier(s) of the neuron(s) to be probed ([]: none)

    # Protocol parameters
    learn_prot1 = completeProt(
        config["simulation"]["learn_protocol1"]
    )  # protocol for learning stimulation as a dictionary with the keys "time_start" (starting time),
    # "scheme" (scheme of pulses), and "freq" (stimulation frequency)

    bg_prot = completeProt(
        config["simulation"]["bg_protocol"]
    )  # protocol for background input

    N_CA = config["populations"]["N_CA"]  # 150, Core assembly
    p_r = config["populations"][
        "p_r"
    ]  # 0.5, Fraction of neurons activated for recall phase

    nC = 1e-9 * b.coulomb
    w_stim = 0.420075 * nC

    # network parameters
    N_exc = config["populations"]["N_exc"]  # 1600
    N_inh = config["populations"]["N_inh"]  # 400
    N_tot = N_exc + N_inh  # 2000
    tau_syn = config["synapses"]["tau_syn"] * msecond  # 5 ms
    t_syn_delay = config["synapses"]["t_ax_delay"] * msecond  # 3 ms

    patterns = [[], []]
    patterns[0] = np.arange(150)
    patterns[1] = np.arange(112, 262)

    # Weights
    # The weights are in the units of volts, since they are simply added to the membrane potential
    # The late-phase weights z are unitless
    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"] * mvolt  # 4.20075
    w_ei = config["populations"]["w_ei"] * h_0  # 2*h0
    w_ie = config["populations"]["w_ie"] * h_0  # 4*h0
    w_ii = config["populations"]["w_ii"] * h_0  # 4*h0

    # Probability of connection between two neurons
    p_c = config["populations"]["p_c"]  # 0.1

    # Neuromodulator level
    nm_init = config["neuron"]["nm"]

    # plasticity parameters
    t_Ca_delay = (
        config["synapses"]["syn_exc_calcium_plasticity"]["t_Ca_delay"] * msecond
    )  # 18.8, Delay of postsynaptic calcium influx after presynaptic spike

    # Contributions of pre and post spikes to calcium concentration, adjusted down to 60 percentage (for in vivo)
    Ca_pre = config["synapses"]["syn_exc_calcium_plasticity"]["Ca_pre"]  # 0.6
    Ca_post = config["synapses"]["syn_exc_calcium_plasticity"]["Ca_post"]  # 0.1655
    tau_Ca = 0.0488 * second  # Calcium time constant, 48.8 milisecond
    tau_h = 688.4 * second  # Early-phase time constant, 11.47 minutes
    tau_pro_c = 1 * hour  # 1 * hour, Protein time constant
    tau_z = 1 * hour  # Late-phase weight time constant
    z_max = 1  # Saturation level on late-phase weight. Can max be h_0
    gamma_p = 1645.6  # Potentiation rate
    gamma_d = 313.1  #  Depression rate
    theta_p = config["synapses"]["syn_exc_calcium_plasticity"][
        "theta_p"
    ]  # 3, Calcium threshold for potentiation
    theta_d = config["synapses"]["syn_exc_calcium_plasticity"][
        "theta_d"
    ]  # 1.2, Calcium threshold for depression
    nu_th_LTP = 0 * hertz
    nu_th_LTD = 0 * hertz
    sigma_pl = (
        config["synapses"]["syn_exc_calcium_plasticity"]["sigma_pl"] * mvolt
    )  # 2.90436, Early-phase weight dynamic, SD of noise
    alpha_c = 1  # Protein synthesis rate
    theta_tag_p = (
        config["synapses"]["syn_exc_calcium_plasticity"]["theta_tag"] * mvolt
    )  # 0.840149, Tagging threshold
    theta_tag_d = (
        config["synapses"]["syn_exc_calcium_plasticity"]["theta_tag"] * mvolt
    )  # 0.840149, Tagging threshold

    # neuron parameters
    R_mem = config["neuron"]["R_leak"] * Mohm  # 10.0
    C_mem = config["neuron"]["C_mem"] * farad  # 1e-9
    tau_mem = R_mem * C_mem  # membrane time constant
    V_rev = config["neuron"]["V_rev"] * mvolt  # -65.0
    t_ref = (
        config["neuron"]["t_ref"] * msecond + delta_t
    )  # 2.0, refractory period, add one time step to comply with other implementations
    V_reset = config["neuron"]["V_reset"] * mvolt  # -70.0
    V_th = config["neuron"]["V_th"] * mvolt  # -55.0
    V_spike = 35 * mvolt
    tau_OU = tau_syn  # Time constant for OU noise process
    w_spike = (
        100 * (V_th - V_reset) * np.exp(delta_t / tau_mem)
    )  # value that is sufficently large to cause spike in postsyn. neuron

    ##########################################################################
    # Save config, and start logging
    ##########################################################################

    json.dump(
        config,
        open(getDataPath(timestamp, s_desc, file="config.json")[1], "w"),
        indent="\t",
    )

    # output of key parameters
    logger.info(f"""
    Brian network simulation {getTimestamp()} (Brian version: {b.__version__})
    |
    Description: {config['simulation']['description']} s
    Protocol: L1
    Population parameters: N_exc = {config['populations']['N_exc']}, N_inh = {config['populations']['N_inh']}
    Learning protocol 1: {protFormatted(learn_prot1)}
    Background input: {protFormatted(bg_prot)}
    Starting Neuromodulator level: {config['neuron']['nm']}
    |
    """.strip())

    for run_idx in range(config["simulation"]["n_iter"]):

        seed = config["simulation"]["start_seed"] + run_idx
        np.random.seed(seed)
        b.seed(seed)

        logger.info(f"Run with seed: {seed}")

        ##########################################################################
        # Set background Input
        ##########################################################################

        I_0, sigma_wn = set_background_current(bg_prot)

        ##########################################################################
        # set up the neuron populations of the network
        ##########################################################################
        neur_pop_exc = b.NeuronGroup(
            N_exc,
            get_model("neuron_full"),
            threshold="V > V_th",
            reset="V=V_reset",
            refractory="t_ref",
            method="heun",
        )
        neur_pop_exc.V = V_rev  # initialize membrane potential at reversal potential
        neur_pop_exc.V_psp = 0 * mvolt  # initialize postsynaptic potential at zero
        neur_pop_exc.I_bg = I_0  # initialize background current at mean value
        neur_pop_exc.nm = nm_init

        neur_pop_inh = b.NeuronGroup(
            N_inh,
            get_model("neuron_simple"),
            threshold="V > V_th",
            reset="V=V_reset",
            refractory="t_ref",
            method="heun",
        )
        neur_pop_inh.V = V_rev  # initialize membrane potential at reversal potential
        neur_pop_inh.V_psp = 0 * mvolt  # initialize postsynaptic potential at zero
        neur_pop_inh.I_bg = I_0  # initialize background current at mean value

        ##########################################################################
        # Set up the stimulus groups
        ##########################################################################

        # learning
        neur_pop_ou_stim_learn1 = make_stimulus_group(N_CA, learn_prot1, 1e9, h_0)

        ##########################################################################
        # Set up the synapse populations
        ##########################################################################
        syn_exc_exc = b.Synapses(
            neur_pop_exc,
            neur_pop_exc,
            model=get_model("plastic_synapse"),
            on_pre=get_model("plastic_synapse_pre"),
            on_post=get_model("plastic_synapse_post"),
            method="heun",
        )  # model of synapses within excitatory population ('neur_pop_exc')
        syn_exc_inh = b.Synapses(
            neur_pop_exc,
            neur_pop_inh,
            model=get_model("static_synapse"),
            on_pre=get_model("static_synapse_pre"),
            method="heun",
        )  # model of synapses from excitatory to inhibitory neurons
        syn_inh_exc = b.Synapses(
            neur_pop_inh,
            neur_pop_exc,
            model=get_model("inhibitory_stdp_synapse"),
            on_pre=get_model("inhibitory_stdp_pre"),
            on_post=get_model("inhibitory_stdp_post"),
            method="heun",
        )  # model of synapses from inhibitory to excitatory neurons
        syn_inh_inh = b.Synapses(
            neur_pop_inh,
            neur_pop_inh,
            model=get_model("static_synapse"),
            on_pre=get_model("static_synapse_pre"),
            method="heun",
        )  # model of synapses within inhibitory population ('neur_pop_inh')

        # Synapses from stimulus groups to exc population
        syn_ou_stim_learn1 = b.Synapses(
            neur_pop_ou_stim_learn1, neur_pop_exc, model=get_model("stim_input_0")
        )

        ##########################################################################
        # Create the connections that provide input to the network
        ##########################################################################
        # Here is where we connect the two patterns
        for j_ in range(N_CA):  # input to all neurons in the assembly core
            syn_ou_stim_learn1.connect(
                i=j_, j=patterns[0][j_]
            )  # connect neurons in `neur_pop_ou_stim_learn` one-by-one to the first

        ##########################################################################
        # Create random connections
        ##########################################################################
        logger.info(f"Creating connections with probability p_c = {p_c}.")
        syn_exc_exc.connect(condition="i != j", p=p_c)
        if N_inh > 0:
            syn_exc_inh.connect(condition="i != j", p=p_c)
            syn_inh_exc.connect(condition="i != j", p=p_c)
            syn_inh_inh.connect(condition="i != j", p=p_c)

        # Set initial values for synapse-related variables (is not allowed to be done earlier)
        syn_exc_exc.h = h_0
        syn_exc_exc.z = 0
        syn_exc_exc.Ca = 0
        syn_exc_exc.pre_voltage.delay = t_syn_delay
        syn_exc_exc.pre_calcium.delay = t_Ca_delay

        syn_exc_inh.w = w_ei
        syn_inh_exc.w = -w_ie
        syn_inh_exc.pre_voltage.delay = t_syn_delay
        syn_inh_inh.w = -w_ii
        syn_exc_inh.pre_voltage.delay = t_syn_delay
        syn_inh_exc.pre_voltage.delay = t_syn_delay
        syn_inh_inh.pre_voltage.delay = t_syn_delay

        syn_inh_exc.Apre = 0
        syn_inh_exc.Apost = 0

        # Initial experimental choices; tune for your network.
        syn_inh_exc.tau_istdp = 20 * b.ms
        syn_inh_exc.rho_target = 3 * b.Hz
        syn_inh_exc.eta_istdp = 0.01 * b.mV
        syn_inh_exc.w_max_istdp = 2 * w_ie
        syn_inh_exc.plasticity_on = 0

        ie_pre = np.asarray(syn_inh_exc.i[:], dtype=int)
        ie_post = np.asarray(syn_inh_exc.j[:], dtype=int)
        ie_initial_mV = np.array(syn_inh_exc.w[:] / b.mV)
        ie_targets_pattern1 = np.isin(ie_post, patterns[0])
        recording_rng = np.random.default_rng(seed + 42)

        def sample_synapses(ids, maximum=100):
            return np.sort(
                recording_rng.choice(ids, size=min(maximum, len(ids)), replace=False)
            )

        ie_record_ids = np.concatenate(
            [
                sample_synapses(np.flatnonzero(ie_targets_pattern1)),
                sample_synapses(np.flatnonzero(~ie_targets_pattern1)),
            ]
        )

        neuron_ids_exc = neuron_ids[neuron_ids < N_exc]
        neuron_ids_inh = neuron_ids[neuron_ids >= N_exc] - N_exc
        conn_i = np.array(syn_exc_exc.i[:])  # presynaptic neuron ids
        conn_j = np.array(syn_exc_exc.j[:])  # postsynaptic neuron ids

        # Synapse in pattern1
        pattern1_neurons = np.array(patterns[0])
        pattern1_synapse_ids = np.where(
            np.isin(conn_i, pattern1_neurons) & np.isin(conn_j, pattern1_neurons)
        )[0]
        pattern1_outgoing_synapse_ids = np.where(  # outgoing synapses from pattern1
            np.isin(conn_i, pattern1_neurons) & ~np.isin(conn_j, pattern1_neurons)
        )[0]

        all_pattern_neurons = pattern1_neurons
        control_synapse_ids = np.where(
            ~np.isin(conn_i, all_pattern_neurons)
            & ~np.isin(conn_j, all_pattern_neurons)
        )[0]
        ctrl_ids = np.setdiff1d(np.arange(N_exc), all_pattern_neurons)

        ########################################################################
        ##################### Learning Pattern 1 Run ###########################
        ########################################################################

        logger.info(f"")
        logger.info(f"Learning-1...")

        t_0 = time.time()

        # Learning Phase
        # Store the network for learning and recall
        net_base = b.Network(
            neur_pop_exc,
            neur_pop_inh,
            neur_pop_ou_stim_learn1,
            syn_exc_exc,
            syn_exc_inh,
            syn_inh_inh,
            syn_inh_exc,
            syn_ou_stim_learn1,
        )
        b.defaultclock.dt = delta_t
        neur_pop_ou_stim_learn1.t_start = learn_prot1["time_start"] * second

        # Monitors for recording pre consolidation time
        learning_mons1 = make_learning_monitors(
            neur_pop_exc,
            neur_pop_inh,
            pattern1_neurons,
            control_synapse_ids,
            pattern1_synapse_ids,
            pattern1_outgoing_synapse_ids,
            syn_exc_exc,
            max_neuron=50,
            max_syn=100,
        )
        learning_mons1["inh_weight_mon"] = b.StateMonitor(
            syn_inh_exc,
            "w",
            record=ie_record_ids,
            dt=10 * b.ms,
        )
        net_base.add(*learning_mons1.values())
        net_base.run(
            config["simulation"]["learning_pattern1_runtime"] * second, report="stdout"
        )
        # Include the exact final state in the recorded trajectories.
        learning_mons1["inh_weight_mon"].record_single_timestep()
        net_base.remove(*learning_mons1.values())

        ########################################################################
        # Store the data in files
        ########################################################################

        learning1_results = {
            "time_ms": np.array(learning_mons1["rate_mon_exc"].t / msecond),
            "pattern1_neurons": pattern1_neurons,
            "pattern1_synapses": {
                "Ca": np.array(learning_mons1["syn_state_mon_pattern1"].Ca),
                "h_mV": np.array(learning_mons1["syn_state_mon_pattern1"].h / mvolt),
                "z": np.array(learning_mons1["syn_state_mon_pattern1"].z),
            },
            "pattern1_outgoing_synapses": {
                "Ca": np.array(learning_mons1["syn_state_mon_pattern1_outgoing"].Ca),
                "h_mV": np.array(
                    learning_mons1["syn_state_mon_pattern1_outgoing"].h / mvolt
                ),
                "z": np.array(learning_mons1["syn_state_mon_pattern1_outgoing"].z),
            },
            "control_synapses": {
                "Ca": np.array(learning_mons1["syn_state_mon_control"].Ca),
                "h_mV": np.array(learning_mons1["syn_state_mon_control"].h / mvolt),
                "z": np.array(learning_mons1["syn_state_mon_control"].z),
            },
            "spikes_exc": {
                "t_ms": np.array(learning_mons1["spike_mon_exc"].t / msecond),
                "i": np.array(learning_mons1["spike_mon_exc"].i),
            },
            "spikes_inh": {
                "t_ms": np.array(learning_mons1["spike_mon_inh"].t / msecond),
                "i": np.array(learning_mons1["spike_mon_inh"].i),
            },
            "rates": {
                "t_ms": np.array(learning_mons1["rate_mon_exc"].t / msecond),
                "exc_Hz": np.array(learning_mons1["rate_mon_exc"].rate / b.Hz),
                "inh_Hz": np.array(learning_mons1["rate_mon_inh"].rate / b.Hz),
            },
        }

        ie_final_mV = np.array(syn_inh_exc.w[:] / b.mV)
        ie_mon = learning_mons1["inh_weight_mon"]
        learning1_results["inhibitory_synapses"] = {
            # All connections: matching entries refer to the same synapse.
            "pre_neuron": ie_pre,
            "post_neuron": ie_post,
            "targets_pattern1": ie_targets_pattern1,
            "initial_w_mV": ie_initial_mV,
            "final_w_mV": ie_final_mV,
            "delta_w_mV": ie_final_mV - ie_initial_mV,
            # Sampled trajectories: rows follow recorded_synapse_ids.
            "recorded_synapse_ids": ie_record_ids,
            "time_ms": np.array(ie_mon.t / b.ms),
            "w_mV": np.array(ie_mon.w / b.mV),
            # Preserve the parameters used for this run.
            "parameters": {
                "tau_istdp_ms": float(syn_inh_exc.tau_istdp[:] / b.ms),
                "rho_target_Hz": float(syn_inh_exc.rho_target[:] / b.Hz),
                "eta_istdp_mV": float(syn_inh_exc.eta_istdp[:] / b.mV),
                "w_max_istdp_mV": float(syn_inh_exc.w_max_istdp[:] / b.mV),
                "plasticity_on": float(syn_inh_exc.plasticity_on[:]),
            },
        }

        learning1_sim_data_path = getDataPath(
            timestamp, s_desc, file="learning1_results.pkl"
        )[1]

        with open(learning1_sim_data_path, "wb") as f:
            pickle.dump(learning1_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        del learning1_results

        json.dump(
            config,
            open(getDataPath(timestamp, s_desc, file="config.json")[1], "w"),
            indent="\t",
        )

        t_2 = time.time()


########################################################################
########################################################################
########################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Configuration of the simulation parameters (JSON file)",
    )

    args = parser.parse_args()

    config = json.load(open(args.config_file, "r"))

    run(config)
