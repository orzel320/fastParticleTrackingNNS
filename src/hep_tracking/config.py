from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class TrackSimulationConfig:
    hits_per_track: int
    noise_ratio: float
    sigma_pos: float
    sigma_dir: float
    vertex_spread: float
    r_max: float = 100.0
    dir_scale: float = 60.0
    seed: int = 42

@dataclass
class DatasetConfig:
    name: str
    target_hits: int
    simulation_params: TrackSimulationConfig
    max_hits_per_event: int = 10_000

@dataclass
class KNNModelConfig:
    name: str
    model_factory: Callable[..., Any]
    model_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class ClassifierModelConfig:
    name: str
    model_factory: Callable[..., Any]
    model_kwargs: dict[str, Any] = field(default_factory=dict)