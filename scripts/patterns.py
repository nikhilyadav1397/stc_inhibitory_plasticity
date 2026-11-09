import numpy as np


# We can generate two patterns that overlap (the common size)
def generate_overlap_patterns(total_neurons, common_size, unique_size, pattern_order):
    all_neurons = np.arange(total_neurons)
    common_neurons = np.random.choice(all_neurons, common_size, replace=False)
    remaining_neurons = np.setdiff1d(all_neurons, common_neurons)
    patterns = []
    used_unique = set()
    for _ in range(pattern_order):
        available = np.setdiff1d(remaining_neurons, np.array(list(used_unique)))
        unique_neurons = np.random.choice(available, unique_size, replace=False)
        used_unique.update(unique_neurons)
        pattern = np.sort(np.concatenate([common_neurons, unique_neurons]))
        patterns.append(pattern)
    pattern_unique = np.unique(np.concatenate(patterns))
    control_neurons = np.setdiff1d(all_neurons, pattern_unique)
    control_neurons = np.sort(control_neurons)
    return patterns, control_neurons


# Generate random patterns, means the patterns can be overlapping randomly
# pattern_order: number of patterns to be generated
def generate_random_patterns(total_neurons, pattern_size, pattern_order):
    all_neurons = np.arange(total_neurons)
    patterns = []
    for _ in range(pattern_order):
        pattern = np.sort(np.random.choice(all_neurons, pattern_size, replace=False))
        patterns.append(pattern)
    pattern_unique = np.unique(np.concatenate(patterns))
    control_neurons = np.setdiff1d(all_neurons, pattern_unique)
    return patterns, np.sort(control_neurons)


# generate_nonoverlap_patterns
def generate_nonoverlap_patterns(total_neurons, pattern_size, pattern_order):
    all_neurons = np.arange(total_neurons)
    np.random.shuffle(all_neurons)
    patterns = []
    for i in range(pattern_order):
        pattern = np.sort(all_neurons[i * pattern_size : (i + 1) * pattern_size])
        patterns.append(pattern)
    pattern_unique = np.unique(np.concatenate(patterns))
    control_neurons = np.setdiff1d(all_neurons, pattern_unique)
    control_neurons = np.sort(control_neurons)

    return patterns, control_neurons
