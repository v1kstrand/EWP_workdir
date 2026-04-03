from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Tracker:
    def log_parameters(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


@dataclass
class LocalTracker(Tracker):
    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.params_path = self.out_dir / "params.json"

    def log_parameters(self, params: dict[str, Any]) -> None:
        self.params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    def finish(self) -> None:
        return


class NullTracker(Tracker):
    def log_parameters(self, params: dict[str, Any]) -> None:
        return

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        return

    def finish(self) -> None:
        return


class CometTracker(Tracker):
    def __init__(
        self,
        project_name: str,
        api_key: str,
        out_dir: Path,
        workspace: str | None = None,
        experiment_key: str | None = None,
    ):
        import comet_ml

        workspace = workspace or os.environ.get("COMET_WORKSPACE")
        common_kwargs = dict(
            api_key=api_key,
            project_name=project_name,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            parse_args=False,
            auto_output_logging=False,
            log_env_details=False,
            log_git_metadata=False,
            log_git_patch=False,
            display_summary_level=0,
        )
        if workspace:
            common_kwargs["workspace"] = workspace
        if experiment_key:
            self.experiment = comet_ml.ExistingExperiment(
                experiment_key=experiment_key,
                **common_kwargs,
            )
            print(f"Comet attached: key={self.experiment.get_key()}")
        else:
            self.experiment = comet_ml.Experiment(**common_kwargs)
            print(f"Comet initialized: key={self.experiment.get_key()}")

        self.experiment.set_name(out_dir.name)
        experiment_url = None
        if hasattr(self.experiment, "url"):
            experiment_url = self.experiment.url
        elif hasattr(self.experiment, "get_url"):
            experiment_url = self.experiment.get_url()
        if experiment_url:
            print(f"Comet url: {experiment_url}")
        (out_dir / "comet.experiment_key").write_text(self.experiment.get_key() + "\n", encoding="utf-8")
        if experiment_url:
            (out_dir / "comet.url").write_text(experiment_url + "\n", encoding="utf-8")

    def log_parameters(self, params: dict[str, Any]) -> None:
        self.experiment.log_parameters(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.experiment.log_metrics(metrics, step=step)

    def finish(self) -> None:
        self.experiment.end()


def build_tracker(
    backend: str,
    out_dir: Path,
    project_name: str,
    api_key_env: str,
    workspace: str | None,
    experiment_key: str | None,
    disabled: bool,
) -> Tracker:
    if disabled:
        return LocalTracker(out_dir)

    api_key = os.environ.get(api_key_env)
    if backend == "comet_required":
        if not api_key:
            raise RuntimeError(f"Missing required Comet API key in env var {api_key_env}")
        return CometTracker(
            project_name=project_name,
            api_key=api_key,
            out_dir=out_dir,
            workspace=workspace,
            experiment_key=experiment_key,
        )

    if backend in {"comet_or_local", "comet_optional"}:
        if api_key:
            try:
                return CometTracker(
                    project_name=project_name,
                    api_key=api_key,
                    out_dir=out_dir,
                    workspace=workspace,
                    experiment_key=experiment_key,
                )
            except (ImportError, RuntimeError, OSError, ValueError):
                return LocalTracker(out_dir)
        return LocalTracker(out_dir)

    if backend == "local":
        return LocalTracker(out_dir)

    raise ValueError(f"Unknown tracker backend: {backend}")
