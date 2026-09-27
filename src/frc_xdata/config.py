"""Load and validate configuration from YAML files and the environment."""

import os
from pathlib import Path
from typing import TypeVar

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError

from frc_xdata.errors import ConfigError

API_KEY_VAR = "ROBOFLOW_API_KEY"

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against a pydantic model.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or does not
            match the model. The message names the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path} does not match {model.__name__}:\n{e}") from e


def roboflow_api_key() -> str:
    """Return the Roboflow API key from the environment or a .env file.

    The .env file is searched for from the current directory upward, so
    commands run from the repo root find the one there. A variable already
    set in the environment wins over the file.

    Raises:
        ConfigError: If ``ROBOFLOW_API_KEY`` is unset or empty.
    """
    # Without usecwd, python-dotenv searches upward from this module's file,
    # which ignores the working directory and misses the repo when the
    # package is installed into site-packages.
    load_dotenv(find_dotenv(usecwd=True))
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise ConfigError(f"{API_KEY_VAR} is not set. Copy .env.example to .env and fill it in.")
    return key
