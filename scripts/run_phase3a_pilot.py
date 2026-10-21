"""Run the locked one-day Phase 3A pooled/Reptile TEM-1 pilot."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esm2_mutation_effects.adaptation import MaskedMarginalResidualModel
from esm2_mutation_effects.data import (
    default_assay_path,
    default_reference_path,
    load_assay_metadata,
    project_root,
)
from esm2_mutation_effects.diagnostics import position_ranking_metrics
from esm2_mutation_effects.esm_utils import REPRESENTATION_LAYER, choose_device, load_model
from esm2_mutation_effects.lora import inject_lora
from esm2_mutation_effects.phase3_pilot import META_TRAIN_TASKS
from esm2_mutation_effects.phase3_protocol import (
    canonical_sha256,
    manifest_sha256,
    validate_strict_target_manifest,
)
from esm2_mutation_effects.phase3_training import (
    adapt_fixed_steps,
    prepare_pilot_data,
    train_pooled,
    train_reptile,
)
from esm2_mutation_effects.training import _predict_rows, seed_everything

EPISODE_COUNTS = {task.dms_id: 38 if index < 2 else 37 for index, task in enumerate(META_TRAIN_TASKS)}
TARGET_STEPS = 50

def args_() -> argparse.Namespace:
    root = project_root(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto"); p.add_argument("--microbatch-size", type=int, default=16)
    p.add_argument("--raw-dir", type=Path, default=root / "data" / "phase3" / "raw")
    p.add_argument("--zero-shot-dir", type=Path, default=root / "results" / "phase3a_pilot" / "zero_shot")
    p.add_argument("--manifest", type=Path, default=root / "results" / "phase3a_pilot" / "target_manifest.csv")
    p.add_argument("--output-dir", type=Path, default=root / "results" / "phase3a_pilot" / "run")
    return p.parse_args()

def model(device: torch.device) -> MaskedMarginalResidualModel:
    backbone, _ = load_model(device); inject_lora(backbone, rank=16, alpha=16, dropout=0.05)
    return MaskedMarginalResidualModel(backbone, representation_layer=REPRESENTATION_LAYER).to(device)

def trainable_state(value: MaskedMarginalResidualModel) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in value.named_parameters() if p.requires_grad}

def predict(value, data, rows, device):
    value.eval()
    with torch.no_grad(): return _predict_rows(value, data, np.asarray(rows), device).cpu().numpy()

def evaluate(name, data, rows, predictions):
    frame = data.frame.iloc[rows][["mutant", "DMS_score", "position"]].copy(); frame["predicted_score"] = predictions
    metrics = position_ranking_metrics(frame.DMS_score.to_numpy(float), frame.predicted_score.to_numpy(float), frame.position.to_numpy())
    return frame, {"method": name, **metrics}

def main() -> None:
    a = args_(); root = project_root(); device = choose_device(a.device)
    if not 1 <= a.microbatch_size <= 16: raise ValueError("microbatch size must be 1..16")
    manifest = pd.read_csv(a.manifest); validate_strict_target_manifest(manifest, label_budget=401)
    metadata = pd.read_csv(default_reference_path()).set_index("DMS_id", verify_integrity=True)
    missing = [x for x in ["BLAT_ECOLX_Stiffler_2015", *EPISODE_COUNTS] if not (a.zero_shot_dir / f"{x}.csv").exists()]
    if missing: raise FileNotFoundError(f"Run score_phase3a_zero_shot.py first; missing {missing}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = a.output_dir / "protocol_lock.json"
    if lock_path.exists(): raise FileExistsError(f"Pilot output already locked: {lock_path}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    lock = {"phase": "phase3a_one_day_pilot", "git_commit": commit, "manifest_sha256": manifest_sha256(manifest), "target_label_budget": 401, "target_steps": TARGET_STEPS, "reptile_episodes": 150, "reptile_inner_steps": 2, "pooled_updates": 300, "test_opened": False}
    lock["protocol_sha256"] = canonical_sha256(lock); lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    # Test scores are first read only after the lock exists.
    _, alphabet = load_model(device)
    target = prepare_pilot_data(pd.read_csv(default_assay_path()), pd.read_csv(a.zero_shot_dir / "BLAT_ECOLX_Stiffler_2015.csv"), str(load_assay_metadata()["target_seq"]), alphabet)
    tasks = {task.dms_id: prepare_pilot_data(pd.read_csv(a.raw_dir / f"{task.dms_id}.csv"), pd.read_csv(a.zero_shot_dir / f"{task.dms_id}.csv"), str(metadata.loc[task.dms_id, "target_seq"]), alphabet) for task in META_TRAIN_TASKS}
    visible = np.flatnonzero(manifest.set_index("mutant").loc[target.frame.mutant, "target_label_visible"].to_numpy())
    test_rows = np.flatnonzero(manifest.set_index("mutant").loc[target.frame.mutant, "subset"].to_numpy() == "test")
    seed_everything(1); records=[]; all_predictions=[]
    zero_frame, zero_metrics = evaluate("zero_shot", target, test_rows, target.fixed_zero_shot[test_rows].numpy()); records.append(zero_metrics); all_predictions.append(zero_frame.assign(method="zero_shot"))
    for name in ("standard", "pooled", "reptile"):
        seed_everything(1); current = model(device)
        if name == "pooled": history = train_pooled(current, tasks, {key: count * 2 for key, count in EPISODE_COUNTS.items()}, device, 1)
        elif name == "reptile": history = train_reptile(current, tasks, EPISODE_COUNTS, device, 1)
        else: history = []
        losses = adapt_fixed_steps(current, target, visible, device, steps=TARGET_STEPS, seed=1)
        frame, metrics = evaluate(name, target, test_rows, predict(current, target, test_rows, device)); records.append(metrics); all_predictions.append(frame.assign(method=name))
        torch.save({"trainable_state": trainable_state(current), "pretraining_history": history, "adaptation_losses": losses}, a.output_dir / f"{name}.pt")
    pd.concat(all_predictions).to_csv(a.output_dir / "test_predictions.csv", index=False)
    summary = {"protocol_lock": str(lock_path.resolve()), "device": str(device), "test_evaluated": True, "results": records}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == "__main__": main()
