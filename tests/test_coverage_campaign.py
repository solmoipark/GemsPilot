import json
from pathlib import Path

import pandas as pd
import pytest

import gemspilot.coverage_campaign as cc
from gemspilot.coverage_campaign import read_target_metric, run_coverage_campaign


def _seed_global_artifacts(db: Path, *, r2: float, nonzero: int) -> None:
    """Create the minimal global_chemistry artifacts the campaign reads."""
    chem_dir = db / "global_chemistry"
    surro_dir = chem_dir / "global_surrogate"
    surro_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(10):
        rows.append(
            {
                "meta__recipe_id": f"seed_{index}",
                "meta__material_system": "OPC_slag",
                "meta__age_days": 28.0,
                "meta__OPC": 70.0 - index,
                "meta__slag": 30.0 + index,
                "meta__w_b": 0.4,
                "x__OPC": 70.0 - index,
                "x__slag": 30.0 + index,
                "y__porosity": 0.30 + 0.001 * index,
                "y__amount_hemicarbonate": 0.001 if index < nonzero else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(chem_dir / "global_model_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "y__amount_hemicarbonate",
                "n_total": 10,
                "r2": r2,
                "rmse": 0.001,
                "full_nonzero_count": nonzero,
                "full_nonzero_fraction": nonzero / 10.0,
            }
        ]
    ).to_csv(surro_dir / "target_metrics.csv", index=False)


class FakeCycle:
    """Simulates the acquisition cycle: acquires rows and improves the metric."""

    def __init__(self, db: Path, acquires: list[int], r2_steps: list[float]):
        self.db = db
        self.acquires = acquires
        self.r2_steps = r2_steps
        self.calls = 0

    def __call__(self, *, out, max_candidates, priority_targets, target_region_table, **kwargs):
        index = self.calls
        self.calls += 1
        acquired = min(self.acquires[index], max_candidates)
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "global_acquisition_cycle_summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "stages": [{"name": "acquire_candidates", "status": "complete", "rows": acquired}],
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        _seed_global_artifacts(self.db, r2=self.r2_steps[index], nonzero=3 + 2 * (index + 1))
        assert priority_targets == ["y__amount_hemicarbonate"]
        assert Path(target_region_table[0]).exists()
        return out_dir


def test_read_target_metric_resolves_short_names(tmp_path):
    _seed_global_artifacts(tmp_path / "gdb", r2=0.1, nonzero=3)
    metric = read_target_metric(tmp_path / "gdb", "hemicarbonate")
    assert metric["target_column"] == "y__amount_hemicarbonate"
    assert metric["r2"] == pytest.approx(0.1)
    assert metric["nonzero_count"] == 3


def test_campaign_tracks_trajectory_and_improvement(tmp_path, monkeypatch):
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    fake = FakeCycle(db, acquires=[3, 3], r2_steps=[0.25, 0.45])
    monkeypatch.setattr(cc, "run_global_chemistry_acquisition_cycle", fake)

    report = run_coverage_campaign(
        target="hemicarbonate", db=db, out=tmp_path / "campaign", cycles=2, candidates_per_cycle=3
    )
    assert report["cycles_run"] == 2
    assert report["stop_reason"] == "cycle_limit_reached"
    assert report["candidates_used"] == 6
    assert report["baseline"]["r2"] == pytest.approx(0.10)
    assert report["final"]["r2"] == pytest.approx(0.45)
    assert report["improvement"]["delta_r2"] == pytest.approx(0.35)
    assert report["trajectory"][0]["delta_r2"] == pytest.approx(0.15)
    assert report["trajectory"][1]["delta_r2"] == pytest.approx(0.20)
    assert Path(report["campaign_report"]).exists()
    md = (tmp_path / "campaign" / "campaign_report.md").read_text(encoding="utf-8")
    assert "y__amount_hemicarbonate" in md


def test_campaign_stops_on_r2_goal_and_budget(tmp_path, monkeypatch):
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    fake = FakeCycle(db, acquires=[3, 3, 3], r2_steps=[0.55, 0.6, 0.7])
    monkeypatch.setattr(cc, "run_global_chemistry_acquisition_cycle", fake)
    report = run_coverage_campaign(
        target="hemicarbonate", db=db, out=tmp_path / "c1", cycles=3,
        candidates_per_cycle=3, stop_r2=0.5,
    )
    assert report["stop_reason"] == "r2_goal_reached"
    assert report["cycles_run"] == 1

    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    fake2 = FakeCycle(db, acquires=[4, 4], r2_steps=[0.2, 0.3])
    monkeypatch.setattr(cc, "run_global_chemistry_acquisition_cycle", fake2)
    report2 = run_coverage_campaign(
        target="hemicarbonate", db=db, out=tmp_path / "c2", cycles=5,
        candidates_per_cycle=4, max_total_candidates=4,
    )
    assert report2["stop_reason"] == "candidate_budget_exhausted"
    assert report2["candidates_used"] == 4


def test_campaign_region_generate_source_feeds_generated_recipes(tmp_path, monkeypatch):
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    captured = {}

    class CapturingCycle(FakeCycle):
        def __call__(self, **kwargs):
            captured["recipes_csv"] = kwargs.get("recipes_csv")
            return super().__call__(**kwargs)

    fake = CapturingCycle(db, acquires=[2], r2_steps=[0.2])
    monkeypatch.setattr(cc, "run_global_chemistry_acquisition_cycle", fake)
    report = run_coverage_campaign(
        target="hemicarbonate", db=db, out=tmp_path / "campaign",
        cycles=1, candidates_per_cycle=2,
        candidate_source="region_generate", generate_n=8, generate_seed=5,
    )
    assert report["cycles_run"] == 1
    recipes = Path(captured["recipes_csv"])
    assert recipes.name == "region_generated_recipes.csv"
    frame = pd.read_csv(recipes)
    assert len(frame) == 8
    assert set(frame["material_system"]) == {"OPC_slag"}


def test_campaign_rejects_unknown_candidate_source(tmp_path):
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    with pytest.raises(ValueError, match="candidate_source"):
        run_coverage_campaign(
            target="hemicarbonate", db=db, out=tmp_path / "c",
            candidate_source="nope",
        )


def test_campaign_stops_on_empty_acquisition(tmp_path, monkeypatch):
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    fake = FakeCycle(db, acquires=[0], r2_steps=[0.10])
    monkeypatch.setattr(cc, "run_global_chemistry_acquisition_cycle", fake)
    report = run_coverage_campaign(
        target="hemicarbonate", db=db, out=tmp_path / "c", cycles=3, candidates_per_cycle=3
    )
    assert report["stop_reason"] == "acquisition_empty"
    assert report["cycles_run"] == 1


def _candidate_row(recipe_id: str, opc: float, slag: float, age: float) -> dict:
    row = {
        "meta__recipe_id": recipe_id,
        "meta__chem_hash": f"hash_{recipe_id}",
        "meta__template_name": "campaign_test",
        "meta__material_system": "OPC_slag",
        "meta__target_profile": "campaign",
        "x__OPC": opc,
        "x__slag": slag,
        "x__fly_ash": 0.0,
        "x__metakaolin": 0.0,
        "x__silica_fume": 0.0,
        "x__limestone": 0.0,
        "x__gypsum": 0.0,
        "x__w_b": 0.4,
        "x__water_g": 40.0,
        "x__age_days": age,
        "x__temperature_celsius": 20.0,
        "x__xgems_water_g": 40.0,
    }
    for oxide in ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O", "CO2", "H2O"]:
        row[f"x__chem_oxide_equiv_mol_{oxide}"] = 0.0
    row["x__chem_oxide_equiv_mol_CaO"] = opc / 100.0
    row["x__chem_oxide_equiv_mol_SiO2"] = slag / 100.0
    return row


def test_campaign_runs_real_mock_cycle_end_to_end(tmp_path):
    """Integration smoke: real acquisition cycle machinery, mock xGEMS batch."""
    db = tmp_path / "gdb"
    _seed_global_artifacts(db, r2=0.10, nonzero=3)
    candidate_table = tmp_path / "pool.csv"
    pd.DataFrame(
        [
            _candidate_row("camp_1", 68.0, 32.0, 28.0),
            _candidate_row("camp_2", 52.0, 48.0, 28.0),
        ]
    ).to_csv(candidate_table, index=False)

    report = run_coverage_campaign(
        target="hemicarbonate",
        db=db,
        out=tmp_path / "campaign",
        cycles=1,
        candidates_per_cycle=2,
        candidate_table=candidate_table,
        use_mock=True,
        refresh=False,
        train_surrogate=False,
    )
    assert report["cycles_run"] == 1
    assert report["candidates_used"] >= 1
    entry = report["trajectory"][0]
    assert entry["stage_statuses"]["acquire_candidates"] == "complete"
    assert entry["stage_statuses"]["run_batch_cached"] == "complete"
    from inverse_gems.database import InverseGemsDatabase

    assert InverseGemsDatabase(db).get_recipe_run("camp_1") is not None
