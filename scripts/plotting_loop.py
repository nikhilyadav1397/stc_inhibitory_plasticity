import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams.update(
    {
        "font.size": 14,  # base font
        "axes.titlesize": 18,  # title
        "axes.labelsize": 16,  # x/y labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def compute_population_rate(spike_times, spike_ids, neuron_ids, bins):
    neuron_ids = np.asarray(neuron_ids)
    if len(neuron_ids) == 0:
        return np.zeros(len(bins) - 1)

    mask = np.isin(spike_ids, neuron_ids)
    counts, _ = np.histogram(spike_times[mask], bins=bins)
    bin_s = (bins[1] - bins[0]) / 1000.0
    return counts / (len(neuron_ids) * bin_s)


def compute_subpopulations(results, config, which_recall=1):
    pattern1 = np.asarray(results["pattern1_neurons"])
    pattern2 = np.asarray(results["pattern2_neurons"])
    recall1 = np.asarray(results["recall1_neurons"])
    recall2 = np.asarray(results["recall2_neurons"])

    if which_recall == 1:
        pattern = pattern1
        recall = recall1
    else:
        pattern = pattern2
        recall = recall2

    ans = np.setdiff1d(pattern, recall)
    all_patterns = np.union1d(pattern1, pattern2)
    ctrl = np.setdiff1d(np.arange(config["populations"]["N_exc"]), all_patterns)

    return recall, ans, ctrl


def load_pickle_if_exists(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _savefig(data_dir, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), dpi=200)
    plt.close()


def plotMonitorResults(config, data_dir):
    learning1_path = os.path.join(data_dir, "learning1_results.pkl")
    learning2_path = os.path.join(data_dir, "learning2_results.pkl")
    consolidation_path = os.path.join(data_dir, "consolidation_results.pkl")
    recall_pre_path = os.path.join(data_dir, "recall_pre_results.pkl")
    post_recall1_path = os.path.join(data_dir, "recall1_results.pkl")
    post_recall2_path = os.path.join(data_dir, "recall2_results.pkl")

    learning1 = load_pickle_if_exists(learning1_path)
    learning2 = load_pickle_if_exists(learning2_path)
    consolidation = load_pickle_if_exists(consolidation_path)
    recall_pre = load_pickle_if_exists(recall_pre_path)
    post_recall1 = load_pickle_if_exists(post_recall1_path)
    post_recall2 = load_pickle_if_exists(post_recall2_path)

    if learning1 is not None:
        plot_learning_results(learning1, config, data_dir, suffix="learning1")

    if learning2 is not None:
        plot_learning_results(learning2, config, data_dir, suffix="learning2")

    if consolidation is not None:
        plot_consolidation_results(consolidation, learning1, config, data_dir)
        del consolidation

    if recall_pre is not None:
        plot_recall_results(
            recall_pre, learning1, config, data_dir, suffix="recall_pre"
        )
        del recall_pre

    if post_recall1 is not None:
        plot_recall_results(
            post_recall1, learning1, config, data_dir, suffix="postrecall1"
        )
        del post_recall1

    if post_recall2 is not None:
        plot_recall_results(
            post_recall2, learning1, config, data_dir, suffix="postrecall2"
        )
        del post_recall2
    del learning1, learning2


def plot_learning_results(results, config, data_dir, suffix="learning"):
    t = results["time_ms"]

    spike_times_exc = np.asarray(results["spikes_exc"]["t_ms"])
    neurons_exc = np.asarray(results["spikes_exc"]["i"])
    spike_times_inh = np.asarray(results["spikes_inh"]["t_ms"])
    neurons_inh = np.asarray(results["spikes_inh"]["i"])

    # -------------------------------
    # Spike raster plot
    # -------------------------------
    N_exc_plot = 500
    N_inh_plot = 100

    spikes_per_neuron = []
    for n in range(N_exc_plot):
        spikes_per_neuron.append(spike_times_exc[neurons_exc == n])
    for n in range(N_inh_plot):
        spikes_per_neuron.append(spike_times_inh[neurons_inh == n])

    plt.figure(figsize=(10, 5))
    plt.title("Spike raster")
    colors = ["blue"] * N_exc_plot + ["red"] * N_inh_plot
    plt.eventplot(
        spikes_per_neuron,
        orientation="horizontal",
        lineoffsets=np.arange(len(spikes_per_neuron)),
        linelengths=0.8,
        colors=colors,
    )
    plt.axhline(N_exc_plot, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.grid(axis="x", alpha=0.2)
    _savefig(data_dir, f"{suffix}_spike_raster.png")

    # -------------------------------
    # Calcium in pattern1 synapses
    # -------------------------------
    Ca_pattern1 = results["pattern1_synapses"]["Ca"]
    Ca_pattern1_out = results["pattern1_outgoing_synapses"]["Ca"]

    plt.figure(figsize=(18, 5))
    plt.title("Calcium Ca for pattern1 synapses")

    Ca_mean = np.mean(Ca_pattern1, axis=0)
    Ca_mean_out = np.mean(Ca_pattern1_out, axis=0)
    Ca_std = np.std(Ca_pattern1, axis=0)
    Ca_std_out = np.std(Ca_pattern1_out, axis=0)

    plt.plot(t, Ca_mean, linewidth=2.5, color="blue", label="mean Ca assembly")
    plt.plot(t, Ca_mean_out, linewidth=2.5, color="orange", label="mean Ca out")
    plt.fill_between(t, Ca_mean - Ca_std, Ca_mean + Ca_std, color="blue", alpha=0.2)
    plt.fill_between(
        t, Ca_mean_out - Ca_std_out, Ca_mean_out + Ca_std_out, color="orange", alpha=0.2
    )

    theta_p = config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"]
    theta_d = config["synapses"]["syn_exc_calcium_plasticity"]["theta_d"]
    plt.axhline(theta_p, linestyle="--", color="red", label="theta_p")
    plt.axhline(theta_d, linestyle="--", color="black", label="theta_d")
    plt.xlabel("Time (ms)")
    plt.ylabel("Ca")
    plt.grid(alpha=0.2)
    plt.legend()
    # _savefig(data_dir, f"{suffix}_calcium_pattern1.png")

    # -------------------------------
    # Weights
    # -------------------------------

    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]
    h_pattern1 = results["pattern1_synapses"]["h_mV"]
    h_pattern1_out = results["pattern1_outgoing_synapses"]["h_mV"]
    h_pattern2 = results["pattern2_synapses"]["h_mV"]
    h_pattern2_out = results["pattern2_outgoing_synapses"]["h_mV"]
    h_control = results["control_synapses"]["h_mV"]

    plt.figure(figsize=(8, 6))
    plt.title(f"{suffix.upper()} Early-phase synaptic weights")

    # -------------------------------
    # Early phase h
    # -------------------------------

    # Means
    h_mean1 = np.mean(h_pattern1, axis=0)
    h_mean_out1 = np.mean(h_pattern1_out, axis=0)
    h_mean2 = np.mean(h_pattern2, axis=0)
    h_mean_out2 = np.mean(h_pattern2_out, axis=0)
    h_mean_ctrl = np.mean(h_control, axis=0)

    # Std
    h_std1 = np.std(h_pattern1, axis=0)
    h_std_out1 = np.std(h_pattern1_out, axis=0)
    h_std2 = np.std(h_pattern2, axis=0)
    h_std_out2 = np.std(h_pattern2_out, axis=0)
    h_std_ctrl = np.std(h_control, axis=0)

    plt.plot(t, h_mean1, linewidth=2, label="Pattern 1")
    plt.plot(t, h_mean_out1, linewidth=2, label="Pattern 1 outgoing")
    plt.plot(t, h_mean2, linewidth=2, label="Pattern 2")
    plt.plot(t, h_mean_out2, linewidth=2, label="Pattern 2 outgoing")
    plt.plot(t, h_mean_ctrl, linewidth=2, label="Control")

    plt.fill_between(t, h_mean1 - h_std1, h_mean1 + h_std1, alpha=0.18)
    plt.fill_between(t, h_mean_out1 - h_std_out1, h_mean_out1 + h_std_out1, alpha=0.18)
    plt.fill_between(t, h_mean2 - h_std2, h_mean2 + h_std2, alpha=0.18)
    plt.fill_between(t, h_mean_out2 - h_std_out2, h_mean_out2 + h_std_out2, alpha=0.18)
    plt.fill_between(
        t, h_mean_ctrl - h_std_ctrl, h_mean_ctrl + h_std_ctrl, color="gray", alpha=0.12
    )
    plt.axhline(h_0, linestyle="--", color="red", linewidth=2, label=r"$h_0$ baseline")
    for stim in [
        10000,
        10500,
        11000,  # first triplet
        20000,
        20500,
        21000,  # second triplet
        30000,
    ]:  # recall
        plt.axvline(stim, color="gray", alpha=0.12, lw=1)

    # Zoom to relevant learning period
    if suffix == "learning1":
        plt.xlim(4000, 7500)
    if suffix == "learning2":
        plt.xlim(14000, 17500)

    plt.xlabel("Time (ms)")
    plt.ylabel("h (mV)")
    plt.ylim(3.5, 8.5)
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, frameon=True)
    plt.tight_layout()
    _savefig(data_dir, f"fig1a_{suffix}.png")


def plot_consolidation_results(results, learning, config, data_dir):
    t = results["time_ms"]
    if "no_prerecall" in config["simulation"]["short_description"]:
        nm = config["neuron"]["nm"]
    else:
        nm = results.get("pre_consolidation_nm", None)
    pattern1_neurons = np.asarray(learning["pattern1_neurons"])
    pattern2_neurons = np.asarray(learning["pattern2_neurons"])

    # -------------------------------
    # sum_h_diff (assembly vs non-assembly)
    # -------------------------------
    sum_h_diff = results["neurons_exc"]["sum_h_diff_mV"]

    all_exc = np.arange(config["populations"]["N_exc"])
    control_neurons = np.setdiff1d(
        all_exc, np.union1d(pattern1_neurons, pattern2_neurons)
    )

    sum_h_diff_pattern1 = sum_h_diff[pattern1_neurons]
    sum_h_diff_pattern2 = sum_h_diff[pattern2_neurons]
    sum_h_diff_control = sum_h_diff[control_neurons]

    mean_pattern1 = np.mean(sum_h_diff_pattern1, axis=0)
    mean_pattern2 = np.mean(sum_h_diff_pattern2, axis=0)
    mean_control = np.mean(sum_h_diff_control, axis=0)
    std_pattern1 = np.std(sum_h_diff_pattern1, axis=0)
    std_pattern2 = np.std(sum_h_diff_pattern2, axis=0)
    std_control = np.std(sum_h_diff_control, axis=0)

    protein_threshold_init = 1 / (config["neuron"]["nm"] + 0.001)
    protein_threshold = 1 / (nm + 0.001)
    t_h = t / (1000 * 3600)
    cross_idx = np.where(mean_pattern1[1:] <= protein_threshold)[0]
    cross_t_h = t_h[cross_idx[0]] if len(cross_idx) > 0 else None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title("Consolidation: summed early-phase deviation")

    # mean traces
    ax.plot(t_h, mean_pattern1, linewidth=2, label="Pattern 1")
    ax.plot(t_h, mean_pattern2, linewidth=2, label="Pattern 2")
    ax.plot(t_h, mean_control, linewidth=2, color="tab:gray", label="Control")
    ax.fill_between(
        t_h,
        mean_pattern1 - std_pattern1,
        mean_pattern1 + std_pattern1,
        alpha=0.18,
    )
    ax.fill_between(
        t_h,
        mean_pattern2 - std_pattern2,
        mean_pattern2 + std_pattern2,
        alpha=0.18,
    )
    ax.fill_between(
        t_h,
        mean_control - std_control,
        mean_control + std_control,
        color="tab:gray",
        alpha=0.12,
    )
    ax.axhline(
        protein_threshold,
        linestyle="--",
        color="black",
        linewidth=2,
        label=rf"Protein threshold ($nm={nm:.3f}$)",
    )
    ax.axhline(
        protein_threshold_init,
        linestyle="--",
        color="black",
        linewidth=2,
        alpha=0.2,
        label=rf"Protein threshold Initial ($nm={config["neuron"]["nm"]:.3f}$)",
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.15)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("sum_h_diff (mV)")
    ax.grid(alpha=0.22)
    ax.set_xlim(t_h.min(), t_h.max())
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.8")
    plt.tight_layout()
    _savefig(data_dir, "fig2a.png")

    # -------------------------------
    # Protein (assembly vs non-assembly)
    # -------------------------------
    protein = results["neurons_exc"]["p"]

    all_exc = np.arange(config["populations"]["N_exc"])
    control_neurons = np.setdiff1d(
        all_exc, np.union1d(pattern1_neurons, pattern2_neurons)
    )
    protein_pattern1 = protein[pattern1_neurons]
    protein_pattern2 = protein[pattern2_neurons]
    protein_control = protein[control_neurons]

    mean_pattern1 = np.mean(protein_pattern1, axis=0)
    mean_pattern2 = np.mean(protein_pattern2, axis=0)
    std_pattern1 = np.std(protein_pattern1, axis=0)
    std_pattern2 = np.std(protein_pattern2, axis=0)
    mean_control = np.mean(protein_control, axis=0)
    std_control = np.std(protein_control, axis=0)

    t_h = t / (1000 * 3600)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title("Consolidation: protein amount")
    ax.plot(t_h, mean_pattern1, linewidth=2, label="Pattern 1")
    ax.plot(t_h, mean_pattern2, linewidth=2, label="Pattern 2")
    ax.plot(t_h, mean_control, linewidth=2, color="tab:gray", label="Control")
    ax.fill_between(
        t_h,
        mean_pattern1 - std_pattern1,
        mean_pattern1 + std_pattern1,
        alpha=0.18,
    )
    ax.fill_between(
        t_h,
        mean_pattern2 - std_pattern2,
        mean_pattern2 + std_pattern2,
        alpha=0.18,
    )
    ax.fill_between(
        t_h,
        mean_control - std_control,
        mean_control + std_control,
        color="tab:gray",
        alpha=0.12,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.15)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Protein")
    ax.set_xlim(t_h.min(), t_h.max())
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.8")
    plt.tight_layout()
    _savefig(data_dir, "fig2b.png")

    # -------------------------------
    # Weights (single plot: h and z*h0)
    # -------------------------------

    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]
    # Early phase h
    h_pattern1 = results["pattern1_synapses"]["h_mV"]
    h_pattern1_out = results["pattern1_outgoing_synapses"]["h_mV"]
    h_pattern2 = results["pattern2_synapses"]["h_mV"]
    h_pattern2_out = results["pattern2_outgoing_synapses"]["h_mV"]
    # Late phase contribution z*h0
    z_pattern1 = results["pattern1_synapses"]["z"] * h_0
    z_pattern1_out = results["pattern1_outgoing_synapses"]["z"] * h_0
    z_pattern2 = results["pattern2_synapses"]["z"] * h_0
    z_pattern2_out = results["pattern2_outgoing_synapses"]["z"] * h_0

    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]

    # Early phase h
    h_pattern1 = results["pattern1_synapses"]["h_mV"]
    h_pattern1_out = results["pattern1_outgoing_synapses"]["h_mV"]
    h_pattern2 = results["pattern2_synapses"]["h_mV"]
    h_pattern2_out = results["pattern2_outgoing_synapses"]["h_mV"]

    # Late phase contribution z*h0
    z_pattern1 = results["pattern1_synapses"]["z"] * h_0
    z_pattern1_out = results["pattern1_outgoing_synapses"]["z"] * h_0
    z_pattern2 = results["pattern2_synapses"]["z"] * h_0
    z_pattern2_out = results["pattern2_outgoing_synapses"]["z"] * h_0

    # Means
    h_mean1 = np.mean(h_pattern1, axis=0)
    h_mean_out1 = np.mean(h_pattern1_out, axis=0)
    h_mean2 = np.mean(h_pattern2, axis=0)
    h_mean_out2 = np.mean(h_pattern2_out, axis=0)

    z_mean1 = np.mean(z_pattern1, axis=0)
    z_mean_out1 = np.mean(z_pattern1_out, axis=0)
    z_mean2 = np.mean(z_pattern2, axis=0)
    z_mean_out2 = np.mean(z_pattern2_out, axis=0)

    # STDs
    h_std1 = np.std(h_pattern1, axis=0)
    h_std_out1 = np.std(h_pattern1_out, axis=0)
    h_std2 = np.std(h_pattern2, axis=0)
    h_std_out2 = np.std(h_pattern2_out, axis=0)

    z_std1 = np.std(z_pattern1, axis=0)
    z_std_out1 = np.std(z_pattern1_out, axis=0)
    z_std2 = np.std(z_pattern2, axis=0)
    z_std_out2 = np.std(z_pattern2_out, axis=0)

    t_h = t / (1000 * 3600)

    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_title("Consolidation: early- and late-phase weight components")

    # -------------------------------
    # Early phase h = solid
    # -------------------------------
    ax.plot(t_h, h_mean1, color="tab:blue", lw=1.5, label="Pattern 1: h")
    ax.plot(t_h, h_mean_out1, color="tab:orange", lw=1.5, label="Pattern 1 out: h")
    ax.plot(t_h, h_mean2, color="tab:green", lw=1.5, label="Pattern 2: h")
    ax.plot(t_h, h_mean_out2, color="tab:red", lw=1.5, label="Pattern 2 out: h")

    ax.fill_between(
        t_h, h_mean1 - h_std1, h_mean1 + h_std1, color="tab:blue", alpha=0.12
    )
    ax.fill_between(
        t_h,
        h_mean_out1 - h_std_out1,
        h_mean_out1 + h_std_out1,
        color="tab:orange",
        alpha=0.12,
    )
    ax.fill_between(
        t_h, h_mean2 - h_std2, h_mean2 + h_std2, color="tab:green", alpha=0.12
    )
    ax.fill_between(
        t_h,
        h_mean_out2 - h_std_out2,
        h_mean_out2 + h_std_out2,
        color="tab:red",
        alpha=0.12,
    )

    # -------------------------------
    # Late phase z*h0 = dashed
    # -------------------------------
    ax.plot(
        t_h, z_mean1, color="tab:blue", lw=2.0, ls="--", label=r"Pattern 1: $z h_0$"
    )
    ax.plot(
        t_h,
        z_mean_out1,
        color="tab:orange",
        lw=2.0,
        ls="--",
        label=r"Pattern 1 out: $z h_0$",
    )
    ax.plot(
        t_h, z_mean2, color="tab:green", lw=2.0, ls="--", label=r"Pattern 2: $z h_0$"
    )
    ax.plot(
        t_h,
        z_mean_out2,
        color="tab:red",
        lw=2.0,
        ls="--",
        label=r"Pattern 2 out: $z h_0$",
    )

    ax.fill_between(
        t_h, z_mean1 - z_std1, z_mean1 + z_std1, color="tab:blue", alpha=0.08
    )
    ax.fill_between(
        t_h,
        z_mean_out1 - z_std_out1,
        z_mean_out1 + z_std_out1,
        color="tab:orange",
        alpha=0.08,
    )
    ax.fill_between(
        t_h, z_mean2 - z_std2, z_mean2 + z_std2, color="tab:green", alpha=0.08
    )
    ax.fill_between(
        t_h,
        z_mean_out2 - z_std_out2,
        z_mean_out2 + z_std_out2,
        color="tab:red",
        alpha=0.08,
    )

    # -------------------------------
    # Baseline h0
    # -------------------------------
    ax.axhline(h_0, color="black", ls=":", lw=2, alpha=0.9, label=r"$h_0$")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Weight contribution (mV)")
    ax.set_xlim(t_h.min(), t_h.max())
    ax.grid(alpha=0.22)

    ax.legend(
        loc="upper right", ncol=2, frameon=True, facecolor="white", edgecolor="0.8"
    )

    plt.tight_layout()
    _savefig(data_dir, "fig2c.png")


def plot_recall_results(results, learning, config, data_dir, suffix="postrecall1"):
    t = results["time_ms"]

    spike_times_exc = np.asarray(results["spikes_exc"]["t_ms"])
    neurons_exc = np.asarray(results["spikes_exc"]["i"])
    spike_times_inh = np.asarray(results["spikes_inh"]["t_ms"])
    neurons_inh = np.asarray(results["spikes_inh"]["i"])

    # -------------------------------
    # Spike raster
    # -------------------------------
    N_exc_plot = 500
    N_inh_plot = 100

    spikes_per_neuron = []
    for n in range(N_exc_plot):
        spikes_per_neuron.append(spike_times_exc[neurons_exc == n])
    for n in range(N_inh_plot):
        spikes_per_neuron.append(spike_times_inh[neurons_inh == n])

    plt.figure(figsize=(10, 5))
    plt.title(f"Spike raster ({suffix})")
    colors = ["blue"] * N_exc_plot + ["red"] * N_inh_plot
    plt.eventplot(
        spikes_per_neuron,
        orientation="horizontal",
        lineoffsets=np.arange(len(spikes_per_neuron)),
        linelengths=0.8,
        colors=colors,
    )
    plt.axhline(N_exc_plot, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.grid(axis="x", alpha=0.2)
    _savefig(data_dir, f"{suffix}_spike_raster.png")

    # -------------------------------
    # Recall weights
    # -------------------------------

    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]

    if suffix == "postrecall1":
        h_pattern = results["pattern1_synapses"]["h_mV"]
        z_pattern = results["pattern1_synapses"]["z"]
        title_label = "Recall 1"
    if suffix == "postrecall2":
        h_pattern = results["pattern2_synapses"]["h_mV"]
        z_pattern = results["pattern2_synapses"]["z"]
        title_label = "Recall 2"
    if suffix == "recall_pre":
        h_pattern = results["pattern1_synapses"]["h_mV"]
        z_pattern = results["pattern1_synapses"]["z"]
        title_label = "Recall 1"

    # convert late phase to mV contribution
    z_pattern_mV = z_pattern * h_0
    w_pattern = h_pattern + z_pattern_mV

    # means
    h_mean = np.mean(h_pattern, axis=0)
    z_mean = np.mean(z_pattern_mV, axis=0)
    w_mean = np.mean(w_pattern, axis=0)

    # std
    h_std = np.std(h_pattern, axis=0)
    z_std = np.std(z_pattern_mV, axis=0)
    w_std = np.std(w_pattern, axis=0)

    t_plot = t / 1000

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_title(f"{title_label}: synaptic weight components")

    # early phase
    ax.plot(t_plot, h_mean, color="tab:blue", linewidth=2.2, label="Early phase h")
    ax.fill_between(
        t_plot,
        h_mean - h_std,
        h_mean + h_std,
        color="tab:blue",
        alpha=0.15,
    )

    # late phase
    ax.plot(
        t_plot,
        z_mean,
        color="tab:orange",
        linestyle="--",
        linewidth=2.2,
        label=r"Late phase $z h_0$",
    )
    ax.fill_between(
        t_plot,
        z_mean - z_std,
        z_mean + z_std,
        color="tab:orange",
        alpha=0.12,
    )

    # total weight
    ax.plot(
        t_plot,
        w_mean,
        color="tab:green",
        linewidth=2.4,
        label=r"Total weight $w = h + z h_0$",
    )
    ax.fill_between(
        t_plot,
        w_mean - w_std,
        w_mean + w_std,
        color="tab:green",
        alpha=0.10,
    )

    # baseline reference
    ax.axhline(
        h_0,
        color="black",
        linestyle=":",
        linewidth=1.8,
        alpha=0.9,
        label=r"Baseline $h_0$",
    )
    ax.set_xlabel("Simulation Time (seconds)")
    ax.set_ylabel("Weight (mV)")
    ax.grid(alpha=0.22)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="0.8")
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)

    plt.tight_layout()
    _savefig(data_dir, f"fig3a_{suffix}.png")
