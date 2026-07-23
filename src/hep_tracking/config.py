"""Configuration data structures for track simulation, datasets, and models.[cite: 3]"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TrackSimulationConfig:
    """Configuration parameters for generating synthetic particle tracks and noise.[cite: 3]

    Attributes:
        hits_per_track: Number of detector hits generated per particle track.[cite: 3]
        noise_ratio: Ratio of noise hits relative to the total number of track hits.[cite: 3]
        sigma_pos: Standard deviation of the positional noise applied to track hits.[cite: 3]
        sigma_dir: Standard deviation of the directional variance.[cite: 3]
        vertex_spread: Maximum absolute coordinate for uniformly generated track origin vertices.[cite: 3]
        r_max: Maximum radius for the detector simulation. Defaults to 100.0.[cite: 3]
        dir_scale: Scaling factor applied to directional vectors. Defaults to 60.0.[cite: 3]
        seed: Random seed for generating reproducible tracks. Defaults to 42.[cite: 3]
    """
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
    """Configuration for generating a complete dataset consisting of multiple events.[cite: 3]

    Attributes:
        name: Unique identifier for the generated dataset.[cite: 3]
        target_hits: The total approximate number of hits desired across the entire dataset.[cite: 3]
        simulation_params: Track simulation configuration detailing how individual 
            events should be generated.[cite: 3]
        max_hits_per_event: Maximum number of hits allowed in a single generated event. 
            Defaults to 10_000.[cite: 3]
    """
    name: str
    target_hits: int
    simulation_params: TrackSimulationConfig
    max_hits_per_event: int = 10_000


@dataclass
class KNNModelConfig:
    """Configuration for instantiating and benchmarking a K-Nearest Neighbors model.[cite: 3]

    Attributes:
        name: Human-readable identifier for the model configuration.[cite: 3]
        model_factory: A callable, such as a class constructor, that creates 
            the model instance.[cite: 3]
        model_kwargs: Dictionary of keyword arguments to unpack and pass into 
            the model factory during instantiation. Defaults to an empty dictionary.[cite: 3]
    """
    name: str
    model_factory: Callable[..., Any]
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierModelConfig:
    """Configuration for instantiating a machine learning classifier.[cite: 3]

    Attributes:
        name: Human-readable identifier for the classifier configuration.[cite: 3]
        model_factory: A callable, such as a class constructor, that creates 
            the classifier instance.[cite: 3]
        model_kwargs: Dictionary of keyword arguments to unpack and pass into 
            the model factory during instantiation. Defaults to an empty dictionary.[cite: 3]
    """
    name: str
    model_factory: Callable[..., Any]
    model_kwargs: dict[str, Any] = field(default_factory=dict)