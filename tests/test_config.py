"""Unit tests for configuration loading and validation."""

import pytest
import yaml
from pathlib import Path

from app.config_loader import load_configurations, ConfigurationError
from app.schemas import RoutesConfigFile, ModelsConfigFile, SettingsConfigFile


def test_load_valid_configurations():
    """Ensure standard project config files load and validate properly."""
    config = load_configurations(config_dir=Path("config"))
    assert len(config.routes) >= 5
    assert len(config.models) >= 8
    assert config.settings.encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.settings.routing.default_selection_strategy == "quality"


def test_duplicate_route_name_rejection(tmp_path):
    """Ensure duplicate route names in routes.yaml raise a validation error."""
    routes_yaml = tmp_path / "routes.yaml"
    routes_yaml.write_text(
        yaml.dump({
            "routes": [
                {"name": "coding", "utterances": ["Write a python script"]},
                {"name": "coding", "utterances": ["Debug Java code"]},
            ]
        })
    )
    with pytest.raises(Exception):
        RoutesConfigFile(**yaml.safe_load(routes_yaml.read_text()))


def test_duplicate_model_rejection(tmp_path):
    """Ensure duplicate provider/model combinations in models.yaml raise a validation error."""
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        yaml.dump({
            "models": [
                {"provider": "OpenAI", "model_name": "gpt-4o", "domains": ["coding"]},
                {"provider": "OpenAI", "model_name": "gpt-4o", "domains": ["math"]},
            ]
        })
    )
    with pytest.raises(Exception):
        ModelsConfigFile(**yaml.safe_load(models_yaml.read_text()))


def test_missing_config_file(tmp_path):
    """Ensure missing required configuration file raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc:
        load_configurations(config_dir=tmp_path)
    assert "not found" in str(exc.value)


def test_malformed_yaml(tmp_path):
    """Ensure malformed YAML content raises ConfigurationError."""
    routes_yaml = tmp_path / "routes.yaml"
    routes_yaml.write_text("routes: [unclosed list")
    with pytest.raises(ConfigurationError) as exc:
        load_configurations(config_dir=tmp_path)
    assert "Malformed YAML" in str(exc.value)


def test_uncovered_domain_error(tmp_path):
    """Ensure an error is raised if a domain has no eligible configured models."""
    routes_yaml = tmp_path / "routes.yaml"
    routes_yaml.write_text(
        yaml.dump({
            "routes": [
                {"name": "unsupported_domain", "utterances": ["Utterance test"]},
            ]
        })
    )
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        yaml.dump({
            "models": [
                {"provider": "OpenAI", "model_name": "gpt-4o", "domains": ["other_domain"]},
            ]
        })
    )
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(yaml.dump({"routing": {"default_selection_strategy": "quality"}}))

    with pytest.raises(ConfigurationError) as exc:
        load_configurations(config_dir=tmp_path)
    assert "unsupported_domain" in str(exc.value)
