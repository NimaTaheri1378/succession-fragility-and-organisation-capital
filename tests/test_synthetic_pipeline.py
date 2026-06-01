from __future__ import annotations

from succession_fragility.pipeline.synthetic import run_synthetic


def test_synthetic_pipeline_outputs_manifest_and_figures(tmp_path) -> None:
    manifest = run_synthetic(tmp_path)
    assert manifest["n_firm_month_rows"] > 0
    assert manifest["fama_macbeth_months"] > 0
    assert all(item["nonblank"] for item in manifest["visual_qa"])
    assert (tmp_path / "manifests" / "synthetic_run.json").exists()
    assert (tmp_path / "figures" / "synthetic_ocf_long_short.png").exists()
