"""Region-derived generative candidate sampling (campaign acquisition upgrade).

Pool-based acquisition can only select recipes that already exist in a pool;
when the pool has few recipes near a sparse target's region, cycles stop
producing target-relevant chemistry. This module instead *generates*
candidates inside the observed target region: it derives a targeted-sampling
profile (material systems, binder bounds, w/b and age windows) from the
region-analysis artifacts (nonzero-row quantiles and per-system support) and
feeds it to the existing bounded-simplex recipe generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from inverse_gems.sampling import generate_recipe_rows, write_recipe_csv
from inverse_gems.utils import config_path, write_json

_BINDER_COLUMNS = [
    "meta__OPC",
    "meta__slag",
    "meta__fly_ash",
    "meta__metakaolin",
    "meta__silica_fume",
    "meta__limestone",
    "meta__gypsum",
]


def _quantile_lookup(quantiles: pd.DataFrame, column: str) -> dict[str, float] | None:
    subset = quantiles[(quantiles["subset"] == "nonzero") & (quantiles["column"] == column)]
    if subset.empty or not int(subset.iloc[0].get("count") or 0):
        return None
    return subset.iloc[0].to_dict()


def derive_region_sampling_profile(
    region_dir: str | Path,
    *,
    lower_quantile: str = "p10",
    upper_quantile: str = "p90",
    max_material_systems: int = 3,
) -> dict[str, Any]:
    """Build a targeted-sampling profile from target-region artifacts."""
    region = Path(region_dir)
    quantiles = pd.read_csv(region / "target_region_quantiles.csv")
    systems_frame = pd.read_csv(region / "target_region_by_material_system.csv")
    summary = json.loads((region / "target_region_summary.json").read_text(encoding="utf-8"))

    supported = systems_frame[systems_frame["nonzero_count"] > 0]
    systems = [str(s) for s in supported["material_system"].head(max_material_systems) if str(s)]
    if not systems:
        raise ValueError(
            f"Target region {region} has no material system with nonzero support; "
            "cannot derive a sampling profile."
        )

    bounds: dict[str, list[float]] = {}
    for column in _BINDER_COLUMNS:
        row = _quantile_lookup(quantiles, column)
        if row is None:
            continue
        low = max(0.0, float(row[lower_quantile]))
        high = float(row[upper_quantile])
        if high <= 0.0 or high < low:
            continue
        bounds[column.replace("meta__", "")] = [round(low, 3), round(high, 3)]

    profile: dict[str, Any] = {
        "description": (
            f"Auto-derived from target region of {summary.get('target_column')} "
            f"({summary.get('nonzero_count')} nonzero rows)."
        ),
        "material_systems": systems,
        "material_systems_sampling": "balanced",
        "bounds_overrides": {"default": bounds},
    }
    age_row = _quantile_lookup(quantiles, "meta__age_days")
    if age_row is not None:
        profile["age_sampling"] = {
            "mode": "log_uniform",
            "min": max(0.1, round(float(age_row[lower_quantile]), 3)),
            "max": max(0.2, round(float(age_row[upper_quantile]), 3)),
            "count": 1,
        }
    wb_row = _quantile_lookup(quantiles, "meta__w_b")
    if wb_row is not None:
        profile["water"] = {
            "mode": "wb_total",
            "w_b": [round(float(wb_row[lower_quantile]), 3), round(float(wb_row[upper_quantile]), 3)],
        }
    return profile


def generate_region_candidates(
    *,
    region_dir: str | Path,
    out: str | Path,
    n: int,
    seed: int = 42,
    sampling_config: str | Path | None = None,
    lower_quantile: str = "p10",
    upper_quantile: str = "p90",
) -> Path:
    """Generate ``n`` candidate recipes inside the target region; returns the CSV path."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = derive_region_sampling_profile(
        region_dir, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    profiles_path = out_dir / "derived_target_profile.yaml"
    profiles_path.write_text(
        yaml.safe_dump({"profiles": {"region_derived": profile}}, sort_keys=False),
        encoding="utf-8",
    )
    rows = generate_recipe_rows(
        config_path=sampling_config or config_path("sampling.yaml"),
        n=int(n),
        mode="mixed",
        seed=seed,
        target_profile="region_derived",
        target_profiles_path=profiles_path,
        recipe_id_prefix="region_gen",
    )
    recipes_path = out_dir / "region_generated_recipes.csv"
    write_recipe_csv(recipes_path, rows)
    write_json(
        out_dir / "region_generation_summary.json",
        {
            "region_dir": str(region_dir),
            "profile": profile,
            "n_requested": int(n),
            "n_generated": len(rows),
            "seed": seed,
            "recipes_csv": str(recipes_path),
        },
    )
    return recipes_path
