"""Configuration loader and validator for YAML configs."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
from pydantic import ValidationError

from app.schemas import RoutesConfigFile, ModelsConfigFile, SettingsConfigFile


class ConfigurationError(Exception):
    """Raised when configuration files are missing, malformed, or fail validation."""
    pass


class AppConfig:
    """Strongly-typed container holding all validated configurations."""
    def __init__(
        self,
        routes_config: RoutesConfigFile,
        models_config: ModelsConfigFile,
        settings_config: SettingsConfigFile,
    ):
        self.routes_config = routes_config
        self.models_config = models_config
        self.settings_config = settings_config

    @property
    def routes(self):
        return self.routes_config.routes

    @property
    def models(self):
        return self.models_config.models

    @property
    def settings(self):
        return self.settings_config


def load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """Reads and parses a YAML file safely."""
    if not filepath.exists():
        raise ConfigurationError(f"Configuration file not found: '{filepath.resolve()}'")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                raise ConfigurationError(f"Configuration file '{filepath.name}' is empty.")
            if not isinstance(data, dict):
                raise ConfigurationError(
                    f"Configuration file '{filepath.name}' must contain a YAML mapping (dictionary)."
                )
            return data
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Malformed YAML in '{filepath.name}': {exc}") from exc


def load_configurations(
    config_dir: Path = Path("config"),
    routes_file: str = "routes.yaml",
    models_file: str = "models.yaml",
    settings_file: str = "settings.yaml",
) -> AppConfig:
    """Loads and validates routes.yaml, models.yaml, and settings.yaml.
    
    Fails fast with detailed errors if validation fails.
    """
    config_dir = Path(config_dir)
    routes_path = config_dir / routes_file
    models_path = config_dir / models_file
    settings_path = config_dir / settings_file

    # 1. Load routes
    raw_routes = load_yaml_file(routes_path)
    try:
        routes_config = RoutesConfigFile(**raw_routes)
    except ValidationError as exc:
        raise ConfigurationError(f"Validation failed for '{routes_path.name}':\n{exc}") from exc

    # 2. Load models
    raw_models = load_yaml_file(models_path)
    try:
        models_config = ModelsConfigFile(**raw_models)
    except ValidationError as exc:
        raise ConfigurationError(f"Validation failed for '{models_path.name}':\n{exc}") from exc

    # 3. Load settings (or use defaults if missing)
    if settings_path.exists():
        raw_settings = load_yaml_file(settings_path)
        try:
            settings_config = SettingsConfigFile(**raw_settings)
        except ValidationError as exc:
            raise ConfigurationError(f"Validation failed for '{settings_path.name}':\n{exc}") from exc
    else:
        settings_config = SettingsConfigFile()

    # 4. Consistency check: Ensure each route domain has at least one configured model
    configured_domains = {r.name for r in routes_config.routes}
    model_domains = set()
    for m in models_config.models:
        model_domains.update(m.domains)

    uncovered = configured_domains - model_domains
    if uncovered:
        raise ConfigurationError(
            f"The following domain(s) defined in '{routes_path.name}' have no eligible models "
            f"in '{models_path.name}': {sorted(list(uncovered))}. Every domain must be serviced by at least one model."
        )

    return AppConfig(
        routes_config=routes_config,
        models_config=models_config,
        settings_config=settings_config,
    )
