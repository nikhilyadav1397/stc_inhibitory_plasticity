from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from scripts.functions import *
from scripts.utils import *


@dataclass(frozen=True)
class Protocols:
    learn1: dict[str, Any]
    learn2: dict[str, Any]
    recall1: dict[str, Any]
    recall2: dict[str, Any]
    recall1_after: dict[str, Any]
    recall2_after: dict[str, Any]
    background: dict[str, Any]


@dataclass(frozen=True)
class PopulationLayout:
    n_exc: int
    n_inh: int
    n_total: int
    n_ca: int
    recall_fraction: float
    pattern_overlap: float
    n_overlap: int
    patterns: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class OutputContext:
    timestamp: str
    output_path: Path
    q_values_path: Path
    logger: Logger


def prepare_protocols(simulation_config: dict[str, Any]) -> Protocols:
    return Protocols(
        learn1=completeProt(simulation_config["learn1"]),
        learn2=completeProt(simulation_config["learn2"]),
        recall1=completeProt(simulation_config["recall1"]),
        recall2=completeProt(simulation_config["recall2"]),
        recall1_after=completeProt(simulation_config["recall1_after"]),
        recall2_after=completeProt(simulation_config["recall2_after"]),
        background=completeProt(simulation_config["bg"]),
    )


def prepare_population_layout(
    population_config: dict[str, Any],
) -> PopulationLayout:
    n_exc = population_config["N_exc"]
    n_inh = population_config["N_inh"]
    n_ca = population_config["N_CA"]
    recall_fraction = population_config["p_r"]
    pattern_overlap = population_config.get("pattern_overlap", 0.5)

    if n_exc <= 0:
        raise ValueError("N_exc must be positive.")

    if n_inh < 0:
        raise ValueError("N_inh cannot be negative.")

    if n_ca <= 0:
        raise ValueError("N_CA must be positive.")

    if not 0.0 <= recall_fraction <= 1.0:
        raise ValueError("p_r must lie between 0 and 1.")

    if not 0.0 <= pattern_overlap <= 1.0:
        raise ValueError("pattern_overlap must lie between 0 and 1.")

    n_overlap = round(pattern_overlap * n_ca)
    second_pattern_start = n_ca - n_overlap

    pattern1 = np.arange(n_ca)
    pattern2 = np.arange(
        second_pattern_start,
        second_pattern_start + n_ca,
    )

    if pattern2[-1] >= n_exc:
        raise ValueError(
            "The two cell assemblies do not fit within the excitatory population."
        )

    return PopulationLayout(
        n_exc=n_exc,
        n_inh=n_inh,
        n_total=n_exc + n_inh,
        n_ca=n_ca,
        recall_fraction=recall_fraction,
        pattern_overlap=pattern_overlap,
        n_overlap=n_overlap,
        patterns=(pattern1, pattern2),
    )


def prepare_output_context(
    short_description: str,
) -> OutputContext:
    timestamp = getTimestamp()

    output_path, log_path = getDataPath(
        timestamp,
        sim_description=short_description,
        file="log.txt",
    )

    q_values_path = getDataPath(
        timestamp,
        sim_description=short_description,
        file="q_values.csv",
    )[1]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    return OutputContext(
        timestamp=timestamp,
        output_path=output_path,
        q_values_path=Path(q_values_path),
        logger=setup_logger(log_path),
    )
