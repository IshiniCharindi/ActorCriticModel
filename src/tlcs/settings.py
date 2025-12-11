from pathlib import Path
from typing import Annotated, Any
import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
class TrainingSettings(BaseModel):
    # simulation
    gui: bool
    total_episodes: PositiveInt
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    green_duration: PositiveInt
    yellow_duration: PositiveInt
    turn_chance: Annotated[float, Field(ge=0, le=1)]
    # model
    num_layers: PositiveInt
    width_layers: PositiveInt
    learning_rate: PositiveFloat
    training_epochs: PositiveInt
    # agent
    gamma: Annotated[float, Field(ge=0, le=1)]
    beta_entropy: PositiveFloat
    lambda_gae: Annotated[float, Field(ge=0, le=1)]
    n_step_size: PositiveInt # Add this line
    # paths
    sumocfg_file: Path
class TestingSettings(BaseModel):
    gui: bool
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    episode_seed: int
    yellow_duration: PositiveInt
    green_duration: PositiveInt
    turn_chance: Annotated[float, Field(ge=0, le=1)]
    gamma: Annotated[float, Field(ge=0, le=1)]
    total_episodes: PositiveInt  # Added: Required in main.py for testing loop
    n_step_size: PositiveInt  # Added: Required for agent init
    learning_rate: PositiveFloat  # Added: Required for agent init
    beta_entropy: PositiveFloat  # Added: Required for agent init
    sumocfg_file: Path
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(
            f"Invalid YAML format in {path}; expected a mapping at the top level"
        )
    return data
def load_training_settings(settings_file: Path) -> TrainingSettings:
    return TrainingSettings.model_validate(load_yaml(settings_file))
def load_testing_settings(settings_file: Path) -> TestingSettings:
    return TestingSettings.model_validate(load_yaml(settings_file))