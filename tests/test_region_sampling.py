import numpy as np
import pandas as pd
import pytest
import yaml

from gemspilot.region_sampling import derive_region_sampling_profile, generate_region_candidates
from inverse_gems.target_region_analysis import write_target_region_analysis


def _region_model_table(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for index in range(60):
        in_region = index < 20
        opc = rng.uniform(45, 70) if in_region else rng.uniform(70, 95)
        fly_ash = rng.uniform(20, 45) if in_region else rng.uniform(0, 10)
        limestone = rng.uniform(0.2, 2.0) if in_region else rng.uniform(10, 25)
        rows.append(
            {
                "meta__recipe_id": f"r{index}",
                "meta__material_system": "OPC_fly_ash_limestone" if in_region else "OPC_limestone",
                "meta__age_days": rng.uniform(20, 120) if in_region else rng.uniform(1, 7),
                "meta__OPC": opc,
                "meta__slag": 0.0,
                "meta__fly_ash": fly_ash,
                "meta__metakaolin": 0.0,
                "meta__silica_fume": 0.0,
                "meta__limestone": limestone,
                "meta__gypsum": rng.uniform(0, 4),
                "meta__w_b": rng.uniform(0.38, 0.5),
                "meta__xgems_water_g": 42.0,
                "y__amount_hemicarbonate": rng.uniform(0.002, 0.01) if in_region else 0.0,
            }
        )
    path = tmp_path / "model_table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_region_profile_derivation_and_generation(tmp_path):
    table = _region_model_table(tmp_path)
    region_dir = write_target_region_analysis(
        model_table=table, target="hemicarbonate", out=tmp_path / "region"
    )
    profile = derive_region_sampling_profile(region_dir)
    assert profile["material_systems"] == ["OPC_fly_ash_limestone"]
    bounds = profile["bounds_overrides"]["default"]
    # bounds learned from the nonzero region, not the full table
    assert bounds["limestone"][1] < 5.0
    assert bounds["fly_ash"][0] > 10.0
    assert profile["age_sampling"]["min"] > 7.0

    recipes = generate_region_candidates(
        region_dir=region_dir, out=tmp_path / "gen", n=10, seed=3
    )
    frame = pd.read_csv(recipes)
    assert len(frame) == 10
    assert set(frame["material_system"]) == {"OPC_fly_ash_limestone"}
    assert (frame["limestone"] <= 5.0).all()
    assert (frame["fly_ash"] >= 10.0).all()
    assert (tmp_path / "gen" / "region_generation_summary.json").exists()


def test_region_profile_requires_nonzero_support(tmp_path):
    table = _region_model_table(tmp_path)
    frame = pd.read_csv(table)
    frame["y__amount_hemicarbonate"] = 0.0
    frame.to_csv(table, index=False)
    region_dir = write_target_region_analysis(
        model_table=table, target="hemicarbonate", out=tmp_path / "region0"
    )
    with pytest.raises(ValueError, match="no material system with nonzero support"):
        derive_region_sampling_profile(region_dir)
