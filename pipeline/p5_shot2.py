"""Shot #2 -- two architecture/training changes against the same 1,000-row manifest.

Question, one factor per arm, nothing else moved:

    aux_head   L_simple + 0.1 * CE(Linear(GAP(bottleneck)), label) on t/T <= 0.25
               hypothesis: a class boundary on the deepest feature lifts accuracy toward 0.90
    edm_heun   EDM noise levels + preconditioning + lambda(sigma) weight + Heun-25
               hypothesis: variance-normalised noise levels lift FID toward 80
    aux_stem   diagnostic only (name it with --arms): the same CE on the unconditioned input stem
               control for "does the head read the label the trunk was already given?"

Both arms start from the code in `pipeline/arms/<arm>/` -- NOT from `submission/`.  The graded
package stays frozen while an experiment runs: `submission/*.py` is unchanged, its manifest is
unchanged, and `results/legacy_1000row/eval_final.json` still records the 1,000-row tree that was
graded then (the shipped numbers are in `results/ab/eval_mixed60k_n1000.json`).

Every arm is trained with the shot-#1 flags (`FLAGS` below, identical to `p4_shot1.FLAGS`) and
20,000 steps, which is 20,000 x 64 / 1,000 = 1,280 passes over the data -- the same epoch budget as
the shipped `admitted_e20k` checkpoint, so a difference cannot be an epoch artifact.

Baseline, not retrained: `results/ab/eval_admitted_e20k_n1000.json` (the checkpoint currently
sitting in `submission/weights.pt`, md5 32a70a7e): acc 0.8400, FID 92.5499, KID 0.03310,
precision 0.4580, recall 0.4646, 11/15.

Stages
------
    --stage smoke    300 steps per arm, then determinism / shape / range / own-manifest checks
    --stage train    20,000 steps into results/ab/<arm>/
    --stage eval     build results/ab/<arm>_sub/ from the arm's own code dir, official evaluator
    --stage summary  six metrics + marks + epochs for the three arms
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "results" / "ab"

ARMS = {
    "aux_head": {
        "code": "pipeline/arms/aux_head",
        "manifest": "submission/manifest.csv",
        "steps": 20000,
        "extra": ["--aux-weight", "0.1", "--aux-tmax", "0.25"],
        "what": "bottleneck class head (514 params), CE on t/T <= 0.25, weight 0.1",
    },
    "edm_heun": {
        "code": "pipeline/arms/edm",
        "manifest": "submission/manifest.csv",
        "steps": 20000,
        "extra": ["--ddim-steps", "25"],
        "what": "EDM preconditioning + lambda(sigma) weight + Heun-25 (49 NFE, same as DDIM-50)",
    },
    # Diagnostic, not a third proposal. The `mid` head saturates at CE ~0 within 100 steps because
    # the bottleneck is class-conditioned: a linear map can read the label the mid-block was given.
    # The `stem` head reads `in_conv(x)` instead, which never sees the label -- so its CE cannot
    # collapse by reading the conditioning, only by reading the pixels. If it also collapses, the
    # finding is about the objective, not about where the head was attached.
    "aux_stem": {
        "code": "pipeline/arms/aux_head",
        "manifest": "submission/manifest.csv",
        "steps": 20000,
        "extra": ["--aux-weight", "0.1", "--aux-tmax", "0.25", "--aux-src", "stem"],
        "what": "diagnostic -- same CE on the unconditioned input stem (130 params)",
        "diagnostic": True,
    },
}

DEFAULT_ARMS = ("aux_head", "edm_heun")

BASELINE_JSON = AB / "eval_admitted_e20k_n1000.json"

FLAGS = ["--steps", "20000", "--batch", "64", "--lr", "2e-4", "--lr-schedule", "constant",
         "--warmup", "0", "--ema", "0.999", "--cond", "add", "--ckpt-every", "10000", "--seed", "0"]

CODE_FILES = ("generate.py", "unet.py", "diffusion.py")
SMOKE_STEPS = 300


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def manifest_rows(p: Path) -> int:
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def flags_for(arm: str, stage: str) -> list[str]:
    f = list(FLAGS) + list(ARMS[arm]["extra"])
    steps = SMOKE_STEPS if stage == "smoke" else ARMS[arm]["steps"]
    f[f.index("--steps") + 1] = str(steps)
    if stage == "smoke":
        f[f.index("--ckpt-every") + 1] = "150"
        f += ["--sample-every", "150"]
    return f


def train(arm: str, stage: str, data_root: str, force: bool = False) -> dict:
    out_dir = AB / (arm if stage == "train" else f"{arm}_smoke")
    weights = out_dir / "weights.pt"
    if weights.exists() and not force:
        print(f"[{arm}/{stage}] {weights.parent.name}/weights.pt exists -- not retraining")
        return {"arm": arm, "stage": stage, "trained": False, "weights": str(weights),
                "md5": md5(weights)}
    cmd = [sys.executable, "-u", str(ROOT / "train_diffusion.py"),
           "--manifest", ARMS[arm]["manifest"], "--data-root", data_root,
           "--code-dir", ARMS[arm]["code"],
           "--out", str(weights), "--log-dir", str(out_dir), *flags_for(arm, stage)]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.log"
    print(f"[{arm}/{stage}] {' '.join(cmd[1:])}", flush=True)
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    wall = round(time.perf_counter() - t0, 1)
    if code != 0 or not weights.exists():
        raise SystemExit(f"[{arm}/{stage}] training failed (exit {code}); see {log}")
    print(f"[{arm}/{stage}] {wall / 60:.1f} min -> {weights.name} "
          f"({weights.stat().st_size / 1e6:.1f} MB)", flush=True)
    return {"arm": arm, "stage": stage, "trained": True, "weights": str(weights),
            "md5": md5(weights), "wall_s": wall, "log": str(log)}


def build_sub(arm: str, weights: Path) -> Path:
    """`results/ab/<arm>_sub`: this arm's code, this arm's manifest, this arm's weights."""
    d = AB / f"{arm}_sub"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in CODE_FILES:
        shutil.copy2(ROOT / ARMS[arm]["code"] / f, d / f)
    shutil.copy2(ROOT / ARMS[arm]["manifest"], d / "manifest.csv")
    shutil.copy2(weights, d / "weights.pt")
    return d


def smoke_report(arm: str) -> dict:
    import torch

    d = AB / f"{arm}_smoke_sub"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in CODE_FILES:
        shutil.copy2(ROOT / ARMS[arm]["code"] / f, d / f)
    shutil.copy2(ROOT / ARMS[arm]["manifest"], d / "manifest.csv")
    shutil.copy2(AB / f"{arm}_smoke" / "weights.pt", d / "weights.pt")

    sys.path.insert(0, str(d))
    for m in ("generate", "diffusion", "unet"):
        sys.modules.pop(m, None)
    gen = __import__("generate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gen.load_model(device)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    a = gen.generate(model, labels, seed=0)
    b = gen.generate(model, labels, seed=0)
    c = gen.generate(model, labels, seed=1)
    same, diff = float((a - b).abs().max()), float((a - c).abs().max())

    run = AB / f"{arm}_smoke"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    losses = [float(x.split(",")[1]) for x in
              (run / "losses.csv").read_text(encoding="utf-8").strip().splitlines()[1:]]
    snaps = sorted(p.name for p in run.glob("snap_step*.pt"))
    rep = {
        "arm": arm, "code": ARMS[arm]["code"], "what": ARMS[arm]["what"],
        "manifest": ARMS[arm]["manifest"], "dataset_rows": cfg["rows"],
        "dataset_from_own_manifest": cfg["rows"] == manifest_rows(ROOT / ARMS[arm]["manifest"]),
        "unet_params": cfg["unet_params"], "scheme": cfg.get("scheme"),
        "steps": cfg["steps"], "seconds_per_step": cfg["seconds_per_step"],
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_last20_mean": round(sum(losses[-20:]) / len(losses[-20:]), 5),
        "snapshots": snaps, "samples_png": (run / "samples_final.png").exists(),
        "determinism_maxdiff_same_seed": same, "maxdiff_different_seed": diff,
        "output_shape": list(a.shape), "output_range": [float(a.min()), float(a.max())],
        "nan": bool(torch.isnan(a).any()),
    }
    if (run / "auxloss.csv").exists():
        rows = (run / "auxloss.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
        last = rows[-1].split(",")
        rep["aux_last"] = {"step": int(last[0]), "mse": float(last[1]), "ce": float(last[2]),
                           "aux_acc": float(last[3]), "n_low_noise": int(last[4])}
        first = rows[0].split(",")
        rep["aux_first"] = {"step": int(first[0]), "ce": float(first[2]),
                            "aux_acc": float(first[3])}
    ok = (rep["dataset_from_own_manifest"] and same <= 1e-4 and diff > 1e-3
          and len(snaps) == (SMOKE_STEPS - 1) // 150 and rep["samples_png"] and not rep["nan"])
    rep["smoke_ok"] = bool(ok)
    print(f"\n[{arm}] rows {rep['dataset_rows']} own_manifest={rep['dataset_from_own_manifest']} "
          f"params {rep['unet_params']:,} scheme={rep['scheme']} "
          f"{rep['seconds_per_step'] * 1000:.0f} ms/step")
    print(f"[{arm}] loss {rep['loss_first']:.4f} -> {rep['loss_last']:.4f} "
          f"(last20 {rep['loss_last20_mean']:.4f})")
    if "aux_last" in rep:
        print(f"[{arm}] aux ce {rep['aux_first']['ce']:.4f} -> {rep['aux_last']['ce']:.4f}  "
              f"aux_acc {rep['aux_first']['aux_acc']:.3f} -> {rep['aux_last']['aux_acc']:.3f}  "
              f"n_low_noise {rep['aux_last']['n_low_noise']}")
    print(f"[{arm}] same-seed maxdiff {same:.2e} (<=1e-4), other seed {diff:.3f}, "
          f"shape {tuple(a.shape)}, range [{rep['output_range'][0]:.3f}, "
          f"{rep['output_range'][1]:.3f}], NaN {rep['nan']}")
    print(f"[{arm}] SMOKE {'PASS' if ok else 'FAIL'}")
    return rep


def evaluator_script() -> Path:
    graded = Path("D:/CISC3024/PROJECT/starter_kit/evaluate.py")
    local = ROOT / "starter_kit" / "evaluate.py"
    script = graded if (graded.exists()
                        and (graded.parent / "catdog_eval" / "classifier.pt").exists()) else local
    if not (script.parent / "catdog_eval" / "classifier.pt").exists():
        raise SystemExit(f"no classifier.pt next to {script}")
    return script


def evaluate(arm: str, n_per_class: int, data_root: str, force: bool = False,
             guidance: float | None = None) -> dict:
    """Official evaluation. `guidance` patches the constant in a throw-away copy of generate.py.

    Patching `GUIDANCE` is the only thing that changes, and it is a *sampler* knob: no retraining,
    no new checkpoint, one factor per point, seed 0, 1000/class -- the project's standing rules for
    a post-training scan (LIMITS.md §11). The arm's canonical `_sub` and eval JSON are untouched.
    """
    weights = AB / arm / "weights.pt"
    if not weights.exists():
        raise SystemExit(f"[{arm}] no weights yet; run --stage train first")
    d = build_sub(arm, weights)
    tag = ""
    if guidance is not None:
        tag = f"_w{guidance:g}"
        d = d.rename(AB / f"{arm}{tag}_sub")
        src = (d / "generate.py").read_text(encoding="utf-8")
        lines = [f"GUIDANCE = {guidance:g}" if ln.startswith("GUIDANCE =") else ln
                 for ln in src.splitlines()]
        (d / "generate.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = AB / f"eval_{arm}{tag}_n{n_per_class}.json"
    if out.exists() and not force:
        print(f"[{arm}{tag}] {out.name} exists -- not re-evaluating (use --force)")
        return json.loads(out.read_text(encoding="utf-8")) | {"arm": f"{arm}{tag}"}
    log = out.with_suffix(".log")
    script = evaluator_script()
    cmd = [sys.executable, "-u", str(script),
           "--submission", str(d), "--n-per-class", str(n_per_class), "--seed", "0",
           "--data-root", str(Path(data_root).resolve()), "--out", str(out)]
    print(f"[{arm}{tag}] {' '.join(cmd[1:])}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(script.parent.parent))
    if code != 0 or not out.exists():
        raise SystemExit(f"[{arm}{tag}] evaluation failed (exit {code}); see {log}")
    r = json.loads(out.read_text(encoding="utf-8"))
    m = r["metrics"]
    print(f"[{arm}{tag}] marks {r['marks']['total']}/15  acc {m['accuracy']:.4f}  "
          f"FID {m['fid']:.2f}  KID {m['kid']:.4f}  prec {m['precision']:.4f}  "
          f"rec {m['recall']:.4f}  mem {m['memorization_rate']}", flush=True)
    return r | {"arm": f"{arm}{tag}"}


def summarize(n_per_class: int, out_path: Path | None = None) -> None:
    rows = []
    if BASELINE_JSON.exists():
        r = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        rows.append({"arm": "baseline_e20k", "rows": r["training_data"]["stats"]["rows"],
                     "epochs": 20000 * 64 // r["training_data"]["stats"]["rows"],
                     "metrics": r["metrics"], "marks": r["marks"]["total"],
                     "violations": r["violations"], "what": "shipped checkpoint (DDPM-50, w=2.5)"})
    for arm in ARMS:
        p = AB / f"eval_{arm}_n{n_per_class}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"arm": arm, "rows": r["training_data"]["stats"]["rows"],
                     "epochs": ARMS[arm]["steps"] * 64 // r["training_data"]["stats"]["rows"],
                     "metrics": r["metrics"], "marks": r["marks"]["total"],
                     "violations": r["violations"], "what": ARMS[arm]["what"]})
    if not rows:
        print("no eval JSONs yet")
        return
    print(f"\n{'arm':16s} {'rows':>5s} {'epochs':>7s} {'acc':>7s} {'fid':>7s} {'kid':>8s} "
          f"{'prec':>7s} {'recall':>7s} {'mem':>6s} {'marks':>6s}")
    for r in rows:
        m = r["metrics"]
        print(f"{r['arm']:16s} {r['rows']:5d} {r['epochs']:7d} {m['accuracy']:7.4f} "
              f"{m['fid']:7.2f} {m['kid']:8.4f} {m['precision']:7.4f} {m['recall']:7.4f} "
              f"{m['memorization_rate']:6.3f} {r['marks']:6d}")
    out = out_path or (AB / f"summary_shot2_n{n_per_class}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"flags": FLAGS, "arms": ARMS, "results": rows}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "train", "eval", "summary", "all"),
                    default="smoke")
    ap.add_argument("--arms", nargs="*", default=None, choices=sorted(ARMS))
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--data-root", default="D:/CISC3024/PROJECT/data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sweep", nargs="*", type=float, default=None,
                    help="guidance weights to evaluate for --arms (sampler-only, no retraining); "
                         "writes eval_<arm>_w<g>_n<N>.json so the canonical record is untouched")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    arms = args.arms or list(DEFAULT_ARMS)
    AB.mkdir(parents=True, exist_ok=True)
    if args.stage in ("smoke", "all"):
        reps = []
        for a in arms:
            train(a, "smoke", args.data_root, force=args.force)
            reps.append(smoke_report(a))
        (AB / "smoke_shot2.json").write_text(json.dumps(reps, indent=2) + "\n", encoding="utf-8")
        if not all(r["smoke_ok"] for r in reps):
            raise SystemExit("SMOKE FAILED -- do not start the 20k runs")
    if args.stage in ("train", "all"):
        for a in arms:
            train(a, "train", args.data_root, force=args.force)
    if args.stage in ("eval", "all"):
        for a in arms:
            evaluate(a, args.n_per_class, args.data_root, force=args.force)
    if args.sweep:
        for a in arms:
            for g in args.sweep:
                evaluate(a, args.n_per_class, args.data_root, force=args.force, guidance=g)
    if args.stage in ("summary", "all"):
        summarize(args.n_per_class, Path(args.summary_out) if args.summary_out else None)


if __name__ == "__main__":
    main()
