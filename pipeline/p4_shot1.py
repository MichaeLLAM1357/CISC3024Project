"""Phase 3/4 shot #1 -- two arms, one variable: `--manifest`.

The question is whether the Phase 1.5 human admission step buys anything, so the two training runs
differ in exactly one argument. Everything else is a literal constant in `FLAGS` below:

    --steps 40000 --batch 64 --lr 2e-4 --lr-schedule constant --warmup 0 --ema 0.999
    --cond add --ckpt-every 10000 --seed 0

No FiLM, no null token, no Min-SNR, no capacity change, no 60k. The 11/15 checkpoint in
`submission/weights.pt` is **not** used as an arm and is not overwritten: it was trained in a
different code tree on a different manifest, so it cannot be one side of this comparison.

Arms, unequal on purpose
------------------------
    draft           results/draft_pre_review.csv   2,000 rows (1,000/class), no human verdict.
                    Pure scientific control. Never enters `submission/`, never enters the final zip.
    admitted        submission/manifest.csv        1,000 rows (500/class), every row human_ok=Y.
                    The only deliverable candidate.
    admitted_e20k   submission/manifest.csv, 20,000 steps (diagnostic, must be named with --arms).
                    The two arms above hold steps fixed, which holds samples fixed but *not* epochs:
                    40,000 x 64 is 1,280 epochs on 2,000 rows and 2,560 on 1,000. This arm runs the
                    same 1,280 epochs as the control, so "fewer, better rows" can be separated from
                    "fewer epochs per row seen twice as often".

Evaluation directories (built by `--stage eval`, one manifest per directory)
---------------------------------------------------------------------------
    results/ab/draft_sub/      generate.py unet.py diffusion.py manifest.csv(=draft)  weights.pt
    results/ab/admitted_sub/   generate.py unet.py diffusion.py manifest.csv(=admitted) weights.pt

`evaluate.py` rebuilds the memorization reference from the manifest that sits next to the weights,
so the two arms must not share one manifest. Passing only `--weights` would score both models
against the same crops, which is exactly the mistake this layout prevents.

Stages
------
    --stage smoke  300 steps per arm (same program, same flags; only --steps and the two periodic
                   knobs move, so that a snapshot and a sample grid land inside 300 steps). Prints
                   dataset rows, parameters, s/step, loss trend, snapshots, determinism/maxdiff.
    --stage train  the real runs into results/ab/<arm>/ (40,000 steps, or the arm's own budget)
    --stage eval   build the *_sub directories and run the official evaluator at 1000/class
    --stage summary  table of the six metrics + marks (and epochs) for the requested arms

Idempotency: an existing weights.pt is never retrained and an existing eval JSON is never
re-evaluated without --force, so re-running a stage cannot silently replace a number.
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
SUB = ROOT / "submission"
AB = ROOT / "results" / "ab"

# The one and only difference between the two shot-#1 arms. `steps` is per arm so that the
# diagnostic can hold *epochs* fixed instead of steps; it defaults to 40,000 (the frozen budget).
ARMS = {
    "draft": {
        "manifest": "results/draft_pre_review.csv",
        "role": "control -- 2,000 rows (1,000/class), no human verdict; never deliverable",
        "steps": 40000,
    },
    "admitted": {
        "manifest": "submission/manifest.csv",
        "role": "treatment -- 1,000 rows (500/class), every row human_ok=Y; deliverable candidate",
        "steps": 40000,
    },
    # Diagnostic, not a third shot. The shot-#1 arms differ in row count *and* epochs at a fixed
    # step budget: 40,000 x 64 samples is 1,280 epochs on 2,000 rows but 2,560 epochs on 1,000.
    # 20,000 steps on the admitted manifest is 1,280 epochs -- the same number of passes over the
    # data as the control. If this arm recovers the recall, the earlier gap was epoch budget; if
    # it does not, the difference is in the rows themselves.
    "admitted_e20k": {
        "manifest": "submission/manifest.csv",
        "role": "diagnostic -- same 1,280 epochs as draft; separates row count from row quality",
        "steps": 20000,
        "diagnostic": True,
    },
}

# Shot #1 is these two and only these two; the diagnostic must be asked for by name.
DEFAULT_ARMS = ("draft", "admitted")

# Frozen by the human for this experiment. Nothing here is chosen by the agent.
FLAGS = ["--steps", "40000", "--batch", "64", "--lr", "2e-4", "--lr-schedule", "constant",
         "--warmup", "0", "--ema", "0.999", "--cond", "add", "--ckpt-every", "10000", "--seed", "0"]

# Files copied into each evaluation directory. `generate.py` is copied byte for byte, so the
# sampler cannot differ between arms.
CODE_FILES = ("generate.py", "unet.py", "diffusion.py")

SMOKE_STEPS = 300


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def arm_run(arm: str) -> Path:
    return AB / arm


def arm_sub(arm: str) -> Path:
    return AB / f"{arm}_sub"


def manifest_rows(p: Path) -> int:
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def flags_for(stage: str, arm: str) -> list[str]:
    """The frozen flags, with only the step budget varying per arm / stage."""
    f = list(FLAGS)
    steps = SMOKE_STEPS if stage == "smoke" else ARMS[arm]["steps"]
    f[f.index("--steps") + 1] = str(steps)
    if stage == "smoke":
        f[f.index("--ckpt-every") + 1] = "150"      # 10000 would never fire inside 300 steps
        f += ["--sample-every", "150"]              # likewise for the sample grid
    return f


def train(arm: str, stage: str, data_root: str, force: bool = False) -> dict:
    spec = ARMS[arm]
    out_dir = arm_run(arm) if stage == "train" else AB / f"{arm}_smoke"
    weights = out_dir / "weights.pt"
    steps = SMOKE_STEPS if stage == "smoke" else ARMS[arm]["steps"]
    if weights.exists() and not force:
        print(f"[{arm}/{stage}] {weights.parent.name}/weights.pt exists -- not retraining")
        return {"arm": arm, "stage": stage, "trained": False, "weights": str(weights),
                "md5": md5(weights)}
    cmd = [sys.executable, "-u", str(ROOT / "train_diffusion.py"),
           "--manifest", spec["manifest"], "--data-root", data_root,
           "--out", str(weights), "--log-dir", str(out_dir), *flags_for(stage, arm)]
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
          f"({weights.stat().st_size / 1e6:.1f} MB), {steps} steps", flush=True)
    return {"arm": arm, "stage": stage, "trained": True, "weights": str(weights),
            "md5": md5(weights), "wall_s": wall, "steps": steps, "log": str(log)}


def build_sub(arm: str, weights: Path) -> Path:
    """`results/ab/<arm>_sub`: the three code files, this arm's manifest, this arm's weights."""
    d = arm_sub(arm)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in CODE_FILES:
        shutil.copy2(SUB / f, d / f)
    shutil.copy2(ROOT / ARMS[arm]["manifest"], d / "manifest.csv")   # renamed on purpose
    shutil.copy2(weights, d / "weights.pt")
    return d


def smoke_report(arm: str, data_root: str) -> dict:
    """Determinism / maxdiff on the smoke checkpoint, plus the numbers the gate asks to see."""
    import torch

    d = arm_sub(arm)
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
    same = float((a - b).abs().max())
    diff = float((a - c).abs().max())

    run = AB / f"{arm}_smoke"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    losses = [float(x.split(",")[1]) for x in
              (run / "losses.csv").read_text(encoding="utf-8").strip().splitlines()[1:]]
    snaps = sorted(p.name for p in run.glob("snap_step*.pt"))
    rep = {
        "arm": arm, "manifest": ARMS[arm]["manifest"], "dataset_rows": cfg["rows"],
        "dataset_from_own_manifest": cfg["rows"] == manifest_rows(ROOT / ARMS[arm]["manifest"]),
        "unet_params": cfg["unet_params"], "steps": cfg["steps"],
        "seconds_per_step": cfg["seconds_per_step"],
        "loss_first": losses[0], "loss_mid": losses[len(losses) // 2], "loss_last": losses[-1],
        "loss_last20_mean": round(sum(losses[-20:]) / len(losses[-20:]), 5),
        "snapshots": snaps, "samples_png": (run / "samples_final.png").exists(),
        "determinism_maxdiff_same_seed": same, "maxdiff_different_seed": diff,
        "output_shape": list(a.shape), "output_range": [float(a.min()), float(a.max())],
        "nan": bool(torch.isnan(a).any()),
    }
    ok = (rep["dataset_from_own_manifest"] and same <= 1e-4 and diff > 1e-3
          # `step % ckpt_every == 0 and step < steps`: a snapshot lands at every multiple of 150
          # strictly below 300, i.e. exactly one (step 150). Step 300 is the final weights.
          and len(snaps) == (SMOKE_STEPS - 1) // 150 and rep["samples_png"] and not rep["nan"])
    rep["smoke_ok"] = bool(ok)
    print(f"\n[{arm}] smoke: rows {rep['dataset_rows']} (from own manifest: "
          f"{rep['dataset_from_own_manifest']}), params {rep['unet_params'] / 1e6:.2f}M, "
          f"{rep['seconds_per_step'] * 1000:.0f} ms/step")
    print(f"[{arm}] loss {rep['loss_first']:.4f} -> {rep['loss_mid']:.4f} -> {rep['loss_last']:.4f}"
          f" (last20 mean {rep['loss_last20_mean']:.4f})")
    print(f"[{arm}] snapshots {snaps}, samples_final.png {rep['samples_png']}")
    print(f"[{arm}] determinism same-seed maxdiff {same:.2e} (<=1e-4), different seed {diff:.3f}; "
          f"shape {tuple(a.shape)} range [{rep['output_range'][0]:.3f}, "
          f"{rep['output_range'][1]:.3f}] NaN {rep['nan']}")
    print(f"[{arm}] SMOKE {'PASS' if ok else 'FAIL'}")
    return rep


def evaluator_script() -> Path:
    """The official evaluator, and the tree it must run from.

    `evaluate.py` is byte-identical in both trees (sha256 `21f881b1…`), but it loads
    `catdog_eval/classifier.pt` from *its own* package directory. This redo workspace deliberately
    has no `classifier.pt`: copying it in is forbidden (LIMITS.md §1.4), and adding anything to
    `starter_kit/` is forbidden too (§1.9). So the evaluation runs the graded tree's copy of the
    same file -- unmodified, hash-checked at the start of the run -- with `--submission` pointing
    at the arm directories here. Nothing is copied and nothing under either `starter_kit/` changes.
    """
    graded = Path("D:/CISC3024/PROJECT/starter_kit/evaluate.py")
    local = ROOT / "starter_kit" / "evaluate.py"
    script = graded if (graded.exists()
                        and (graded.parent / "catdog_eval" / "classifier.pt").exists()) else local
    if not (script.parent / "catdog_eval" / "classifier.pt").exists():
        raise SystemExit(f"no classifier.pt next to {script}; the evaluator cannot check the "
                         f"interface. Point --evaluate-script at the graded tree.")
    return script


def evaluate(arm: str, n_per_class: int, data_root: str, force: bool = False,
             script: Path | None = None) -> dict:
    weights = arm_run(arm) / "weights.pt"
    if not weights.exists():
        raise SystemExit(f"[{arm}] no 40k weights yet; run --stage train first")
    d = build_sub(arm, weights)
    out = AB / f"eval_{arm}_n{n_per_class}.json"
    if out.exists() and not force:
        print(f"[{arm}] {out.name} exists -- not re-evaluating (use --force)")
        return json.loads(out.read_text(encoding="utf-8")) | {"arm": arm}
    log = out.with_suffix(".log")
    script = script or evaluator_script()
    cmd = [sys.executable, "-u", str(script),
           "--submission", str(d), "--n-per-class", str(n_per_class), "--seed", "0",
           "--data-root", str(Path(data_root).resolve()), "--out", str(out)]
    print(f"[{arm}] {' '.join(cmd[1:])}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        # cwd = the tree that owns the evaluator, so `./data` and the package both resolve there.
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(script.parent.parent))
    if code != 0 or not out.exists():
        raise SystemExit(f"[{arm}] evaluation failed (exit {code}); see {log}")
    r = json.loads(out.read_text(encoding="utf-8"))
    m = r["metrics"]
    print(f"[{arm}] marks {r['marks']['total']}/15  acc {m['accuracy']:.4f}  FID {m['fid']:.2f}  "
          f"KID {m['kid']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}  "
          f"mem {m['memorization_rate']}", flush=True)
    return r | {"arm": arm}


def summarize(n_per_class: int, arms: list[str], out_path: Path | None = None) -> None:
    rows = []
    for arm in arms:
        p = AB / f"eval_{arm}_n{n_per_class}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")) | {"arm": arm})
    if not rows:
        print("no eval JSONs yet")
        return
    print(f"\n{'arm':14s} {'rows':>5s} {'epochs':>7s} {'acc':>7s} {'fid':>7s} {'kid':>8s} "
          f"{'prec':>7s} {'recall':>7s} {'mem':>6s} {'marks':>6s}")
    for r in rows:
        m, t = r["metrics"], r["training_data"]["stats"]
        ep = ARMS[r["arm"]]["steps"] * 64 / t["rows"]
        print(f"{r['arm']:14s} {t['rows']:5d} {ep:7.0f} {m['accuracy']:7.4f} {m['fid']:7.2f} "
              f"{m['kid']:8.4f} {m['precision']:7.4f} {m['recall']:7.4f} "
              f"{m['memorization_rate']:6.3f} {r['marks']['total']:6d}")
    out = out_path or (AB / f"summary_n{n_per_class}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": {k: ARMS[k] for k in arms}, "flags": FLAGS,
                               "results": [{"arm": r["arm"],
                                            "rows": r["training_data"]["stats"]["rows"],
                                            "epochs": round(ARMS[r["arm"]]["steps"] * 64
                                                            / r["training_data"]["stats"]["rows"]),
                                            **{k: r["metrics"][k] for k in
                                               ("accuracy", "fid", "kid", "precision", "recall",
                                                "memorization_rate")},
                                            "marks": r["marks"]["total"],
                                            "violations": r["violations"]} for r in rows]},
                              indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "train", "eval", "summary", "all"),
                    default="smoke")
    ap.add_argument("--arms", nargs="*", default=None, choices=sorted(ARMS))
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--evaluate-script", default=None,
                    help="the official evaluate.py to run (default: the graded tree's copy, which "
                         "is the only one with catdog_eval/classifier.pt next to it)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-out", default=None,
                    help="where --stage summary writes (default results/ab/summary_n<N>.json; give "
                         "a different path when summarising a diagnostic so the shot-#1 record "
                         "is not overwritten)")
    args = ap.parse_args()

    arms = args.arms or list(DEFAULT_ARMS)
    AB.mkdir(parents=True, exist_ok=True)
    reps = []
    if args.stage in ("smoke", "all"):
        for a in arms:
            train(a, "smoke", args.data_root, force=args.force)
            build_sub(a, AB / f"{a}_smoke" / "weights.pt")
            reps.append(smoke_report(a, args.data_root))
        (AB / "smoke.json").write_text(json.dumps(reps, indent=2) + "\n", encoding="utf-8")
        if not all(r["smoke_ok"] for r in reps):
            raise SystemExit("SMOKE FAILED -- do not start the 40k runs")
    if args.stage in ("train", "all"):
        for a in arms:
            train(a, "train", args.data_root, force=args.force)
    if args.stage in ("eval", "all"):
        script = Path(args.evaluate_script) if args.evaluate_script else None
        for a in arms:
            evaluate(a, args.n_per_class, args.data_root, force=args.force, script=script)
    if args.stage in ("summary", "all"):
        summarize(args.n_per_class, arms,
                  Path(args.summary_out) if args.summary_out else None)


if __name__ == "__main__":
    main()
