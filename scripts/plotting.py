import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import os.path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t
import os
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf

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


def plot_learning_results(results, config, data_dir):
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
    plt.xlim(3000, 4200)
    plt.ylabel("Neuron index")
    plt.grid(axis="x", alpha=0.2)
    _savefig(data_dir, "learning_spike_raster.png")

    # -------------------------------
    # Calcium in pattern1 synapses
    # -------------------------------
    Ca_pattern1 = np.asarray(results["pattern1_synapses"]["Ca"])
    Ca_pattern1_out = np.asarray(results["pattern1_outgoing_synapses"]["Ca"])

    Ca_mean = np.mean(Ca_pattern1, axis=0)
    Ca_std = np.std(Ca_pattern1, axis=0)
    Ca_mean_out = np.mean(Ca_pattern1_out, axis=0)
    Ca_std_out = np.std(Ca_pattern1_out, axis=0)

    plasticity = config["synapses"]["syn_exc_calcium_plasticity"]
    theta_p = plasticity["theta_p"]
    theta_d = plasticity["theta_d"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, Ca_mean, linewidth=2.5, color="blue", label="Assembly")
    ax.fill_between(
        t,
        Ca_mean - Ca_std,
        Ca_mean + Ca_std,
        color="blue",
        alpha=0.2,
    )

    ax.plot(
        t,
        Ca_mean_out,
        linewidth=2.5,
        color="orange",
        label="Outgoing",
    )
    ax.fill_between(
        t,
        Ca_mean_out - Ca_std_out,
        Ca_mean_out + Ca_std_out,
        color="orange",
        alpha=0.2,
    )

    ax.axhline(
        theta_p,
        linestyle="--",
        color="red",
        linewidth=1.5,
        label=r"$\theta_p$",
    )
    ax.axhline(
        theta_d,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label=r"$\theta_d$",
    )

    ax.set(
        xlim=(3000, 4200),
        xlabel="Time (ms)",
        ylabel="Ca",
        title="Calcium during learning",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _savefig(data_dir, "learning_calcium_pattern1.png")

    # -------------------------------
    # Weights
    # -------------------------------

    h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"]
    h_pattern1 = results["pattern1_synapses"]["h_mV"]
    h_pattern1_out = results["pattern1_outgoing_synapses"]["h_mV"]
    h_control = results["control_synapses"]["h_mV"]

    plt.figure(figsize=(18, 6))
    plt.title("Early-phase synaptic weights")

    # Early phase h
    # Means
    h_mean1 = np.mean(h_pattern1, axis=0)
    h_mean_out1 = np.mean(h_pattern1_out, axis=0)
    h_mean_ctrl = np.mean(h_control, axis=0)

    # Std
    h_std1 = np.std(h_pattern1, axis=0)
    h_std_out1 = np.std(h_pattern1_out, axis=0)
    h_std_ctrl = np.std(h_control, axis=0)

    plt.plot(t, h_mean1, linewidth=2, label="Pattern 1")
    plt.plot(t, h_mean_out1, linewidth=2, label="Pattern 1 outgoing")
    plt.plot(t, h_mean_ctrl, linewidth=2, label="Control")

    plt.fill_between(t, h_mean1 - h_std1, h_mean1 + h_std1, alpha=0.18)
    plt.fill_between(t, h_mean_out1 - h_std_out1, h_mean_out1 + h_std_out1, alpha=0.18)
    plt.fill_between(
        t, h_mean_ctrl - h_std_ctrl, h_mean_ctrl + h_std_ctrl, color="gray", alpha=0.12
    )
    plt.axhline(h_0, linestyle="--", color="red", linewidth=2, label=r"$h_0$ baseline")
    for stim in [
        3000,
    ]:  # recall
        plt.axvline(stim, color="gray", alpha=0.12, lw=1)

    # Zoom to relevant learning period
    plt.xlim(3000, 4200)

    plt.xlabel("Time (ms)")
    plt.ylabel("h (mV)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, frameon=True)
    plt.tight_layout()
    _savefig(data_dir, "learning_weights_pattern1.png")

    # -------------------------------
    # Inhibitory synaptic weights: I → E
    # -------------------------------
    inh = results["inhibitory_synapses"]

    t_inh = np.asarray(inh["time_ms"])
    recorded_ids = np.asarray(inh["recorded_synapse_ids"], dtype=int)

    # Convert negative weights to positive inhibitory strength.
    strength = -np.asarray(inh["w_mV"])
    initial_strength = -np.asarray(inh["initial_w_mV"])
    final_strength = -np.asarray(inh["final_w_mV"])
    delta_strength = final_strength - initial_strength

    # Masks for all connections and for recorded trajectories.
    targets_assembly = np.asarray(inh["targets_pattern1"], dtype=bool)
    recorded_targets_assembly = targets_assembly[recorded_ids]

    groups = [
        ("I → Assembly", targets_assembly, "tab:blue"),
        ("I → Other excitatory", ~targets_assembly, "tab:orange"),
    ]

    # -------------------------------
    # Sampled weight trajectories
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, all_mask, color in groups:
        sample_mask = all_mask[recorded_ids]
        if not np.any(sample_mask):
            continue

        values = strength[sample_mask]
        mean = values.mean(axis=0)
        sd = values.std(axis=0)

        ax.plot(
            t_inh,
            mean,
            color=color,
            linewidth=2,
            label=f"{label} (n={values.shape[0]})",
        )
        ax.fill_between(
            t_inh,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.18,
        )

    stim_ms = config["simulation"]["learn_protocol1"]["time_start"] * 1000
    ax.axvline(
        stim_ms,
        color="gray",
        linestyle=":",
        label="Learning onset",
    )

    ax.set(
        xlabel="Time (ms)",
        ylabel=r"Inhibitory strength $-w$ (mV)",
        title="Inhibitory weights during learning: mean ± SD",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _savefig(data_dir, "learning_inhibitory_weights.png")

    # -------------------------------
    # Changes across all I → E synapses
    # -------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    # Shared bin edges make the populations comparable.
    bins = np.histogram_bin_edges(delta_strength, bins=50)

    for label, mask, color in groups:
        if not np.any(mask):
            continue

        ax.hist(
            delta_strength[mask],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel=r"Change in inhibitory strength $(-w_f)-(-w_i)$ (mV)",
        ylabel="Probability density",
        title="Inhibitory plasticity: positive = stronger inhibition",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _savefig(data_dir, "learning_inhibitory_weight_changes.png")


def remove_outliers(vals):
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)

    if len(vals) < 4:
        return vals

    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    return vals[(vals >= low) & (vals <= high)]


def mean_ci(x, alpha=0.05):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    m = np.mean(x)

    if n < 2:
        return m, m, m

    h = sem(x) * t.ppf(1 - alpha / 2, n - 1)
    return m, m - h, m + h


def p_to_stars(p):
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=12):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=1.2,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def get_p_from_recency_table(recency_df, exp, overlap, corrected=True):
    row = recency_df[
        (recency_df["experiment"] == exp) & (recency_df["overlap"] == overlap)
    ]

    if len(row) == 0:
        return np.nan

    col = "p_fdr" if corrected and "p_fdr" in row.columns else "p_uncorrected"
    return row.iloc[0][col]


def get_p_from_paired_table(paired_df, exp, overlap, comparison, corrected=True):
    row = paired_df[
        (paired_df["experiment"] == exp)
        & (paired_df["overlap"] == overlap)
        & (paired_df["comparison"] == comparison)
    ]

    if len(row) == 0:
        return np.nan

    col = "p_fdr" if corrected and "p_fdr" in row.columns else "p_uncorrected"
    return row.iloc[0][col]


def y_position_for_bracket(vals, pad_frac=0.08):
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)

    ymin = np.min(vals)
    ymax = np.max(vals)
    yrange = ymax - ymin

    if yrange == 0:
        yrange = max(abs(ymax), 0.01)

    return ymax + pad_frac * yrange, 0.04 * yrange
