"""Seven cross-checks that must be green before the 40k run (Phase 3 gate).

The four official pytest tests (`pytest -q tests`) cover the course requirements. These seven are
the couplings *between* components that those tests do not see: the places where two files have to
agree, and where a silent disagreement still produces a valid-looking number. Each check is a
statement about a pair, and every one of them has a failure mode that raises no exception:

  1. noise-dist   the sampler's initial noise and the training epsilon come from the same
                  distribution (the 0.289-vs-0.997 `torch.rand`/`randn` bug: precision looked
                  fine at 0.72 while recall collapsed to 0.008).
  2. device       `generate` takes the device from the model, not from `labels`.
  3. interface    `load_model` -> `generate`: same seed reproduces, different seed differs, and a
                  single-class batch equals the same rows inside a mixed batch.
  4. ckpt-every   the flag declared in the parser actually writes snapshots (it was dead once).
  5. scan-tag     per-variant results carry the checkpoint label and are never overwritten.
  6. artefacts    config.json, losses.csv and samples_final.png describe the same run.
  7. gamma-zero   `snr_gamma = 0` is bit-identical to `L_simple`, and nothing defaults it to > 0.

Usage
-----
    python pipeline/cross_assert.py                 # run all seven, exit 1 if any fails
    python pipeline/cross_assert.py --only 1 4      # run a subset
    python pipeline/cross_assert.py --json results/ab/cross_assert.json

The script never writes inside `submission/`, never touches `data/dev/`, and never loads
`classifier.pt`. Checks 4-6 train a 6-step toy model (24 images) into a temporary directory, so the
whole gate costs well under a minute on CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "starter_kit", ROOT / "submission"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS: list[dict] = []


def record(cid: int, name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})
    print(f"[{cid}] {name:12s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- 1. noise distribution
def check_noise_dist(tmp: Path) -> None:
    from diffusion import DiffusionModel

    torch.manual_seed(0)
    model = DiffusionModel(base=16, emb=32, n_steps=100).eval()
    n = 4096
    sampler_noise = model._initial_noise(n, seed=7, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(1234)
    train_eps = torch.randn((n, 3, 32, 32), generator=g)

    s_std, s_mean = float(sampler_noise.std()), float(sampler_noise.abs().mean())
    t_std, t_mean = float(train_eps.std()), float(train_eps.abs().mean())
    # Both must be N(0,1): mean |x| of N(0,1) is 0.798, of U(-1,1) is 0.5 -- the historical bug
    # was exactly this substitution (std 0.289 vs 0.997).
    ok = (abs(s_std - t_std) < 0.01 and abs(t_std - 1.0) < 0.01 and abs(t_mean - 0.798) < 0.01
          and abs(s_mean - 0.798) < 0.01)
    src = inspect.getsource(model._initial_noise)
    uses_randn = "torch.randn(" in src
    uses_rand = "torch.rand(" in src
    ok = ok and uses_randn and not uses_rand
    record(1, "noise-dist", ok,
           f"sampler std {s_std:.4f} |x| {s_mean:.4f} vs eps std {t_std:.4f} |x| {t_mean:.4f}; "
           f"_initial_noise uses randn={uses_randn} rand={uses_rand}")


# ------------------------------------------------------------------------------------- 2. device
def check_device(tmp: Path) -> None:
    from generate import generate, load_model

    # The graded checkpoint, because that is the object the evaluator puts on a device. It also
    # avoids a trap found while writing this check: the UNet carries `nn.Dropout(0.1)` and is
    # stochastic in train mode, so an untrained model compared twice differs by 1e-2 for reasons
    # that have nothing to do with devices. `load_model` returns `.eval()`, which is what the
    # contract requires.
    model = load_model(torch.device("cpu"))
    labels_cpu = torch.tensor([0, 1], dtype=torch.long)

    out_cpu = generate(model, labels_cpu, seed=3)
    ok_cpu = (out_cpu.shape == (2, 3, 32, 32) and out_cpu.device.type == "cpu"
              and float(out_cpu.min()) >= 0.0 and float(out_cpu.max()) <= 1.0
              and not torch.isnan(out_cpu).any())

    detail = f"cpu->{tuple(out_cpu.shape)} on {out_cpu.device.type}, range ok={ok_cpu}"
    ok = ok_cpu
    if torch.cuda.is_available():
        # The bug this catches: noise built on `labels.device` while the weights are on the GPU.
        # A hand-written check passes CPU labels with a CUDA model, so that is the tested case --
        # with the bug it raises a device-mismatch error instead of returning a tensor.
        # CPU-vs-GPU equality is deliberately NOT asserted: different kernels are allowed to
        # differ, and the evaluator only ever runs one device at a time. What matters is that the
        # output lives on the model's device and that the same seed reproduces there.
        model_gpu = load_model(torch.device("cuda"))
        g1 = generate(model_gpu, labels_cpu, seed=3)
        g2 = generate(model_gpu, labels_cpu, seed=3)
        drift = float((g1 - g2).abs().max())
        on_gpu = g1.device.type == "cuda" and g1.shape == (2, 3, 32, 32)
        ok = ok and on_gpu and drift <= 1e-4
        detail += (f"; cuda weights + cpu labels -> {g1.device.type}, same-seed drift {drift:.2e} "
                   f"(<=1e-4)")
    else:
        detail += "; cuda n/a on this machine"
    record(2, "device", ok, detail)


# ---------------------------------------------------------------------------------- 3. interface
def check_interface(tmp: Path) -> None:
    from generate import generate, load_model

    weights = ROOT / "submission" / "weights.pt"
    model = load_model(torch.device("cpu"))
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    a = generate(model, labels, seed=11)
    b = generate(model, labels, seed=11)
    c = generate(model, labels, seed=12)
    same = float((a - b).abs().max())
    diff = float((a - c).abs().max())

    single = generate(model, torch.zeros(6, dtype=torch.long), seed=5)
    mixed = generate(model, torch.tensor([0, 1, 0, 1, 1, 1]), seed=5)
    d0 = float((single[0] - mixed[0]).abs().max())
    d2 = float((single[2] - mixed[2]).abs().max())

    ok = same <= 1e-4 and diff > 1e-3 and max(d0, d2) <= 1e-4
    record(3, "interface", ok,
           f"load_model({weights.name}): same-seed maxdiff {same:.2e} (<=1e-4), "
           f"seed-change {diff:.3f} (>1e-3), single-vs-mixed {d0:.2e}/{d2:.2e} (<=1e-4)")


# -------------------------------------------------------------------------------- 4. ckpt-every
def toy_run(tmp: Path, steps: int = 6, limit: int = 24, ckpt_every: int = 3) -> Path:
    """A 6-step, 24-image training run; the cheapest thing that exercises the real entry point."""
    d = tmp / "toy"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(ROOT / "train_diffusion.py"),
        "--manifest", "submission/manifest.csv",
        "--data-root", "data",
        "--out", str(d / "weights.pt"),
        "--log-dir", str(d),
        "--steps", str(steps),
        "--batch", "8",
        "--limit", str(limit),
        "--lr", "2e-4",
        "--lr-schedule", "constant",
        "--warmup", "0",
        "--ema", "0.999",
        "--base", "16",
        "--emb", "32",
        "--timesteps", "100",
        "--ddim-steps", "5",
        "--seed", "0",
        "--sample-every", str(steps),
        "--ckpt-every", str(ckpt_every),
        "--device", "cpu",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"toy training failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return d


def check_ckpt_every(tmp: Path, toy: Path) -> None:
    steps, every = 6, 3
    snaps = sorted(p.name for p in toy.glob("snap_step*.pt"))
    # `if step % ckpt_every == 0 and step < steps`: only step 3 qualifies in a 6-step run.
    expected = [f"snap_step{s:06d}.pt" for s in range(every, steps, every)]
    ok = snaps == expected
    detail = f"snapshots written {snaps} (expected {expected})"
    if snaps:
        payload = torch.load(toy / snaps[0], map_location="cpu")
        ok = ok and payload.get("kind") == "ddpm" and payload.get("train_steps") == every
        detail += f"; first is kind={payload.get('kind')} train_steps={payload.get('train_steps')}"
    record(4, "ckpt-every", ok, detail)


# ------------------------------------------------------------------------------------ 5. scan tag
def check_scan_tag(tmp: Path, toy: Path) -> None:
    """`run()` must label its output with the checkpoint and refuse to overwrite it.

    The evaluator subprocess is replaced by a stub that writes a valid-shaped JSON, so this checks
    the naming and the refusal -- the two properties that silently failed before -- without
    spending a real 1000/class evaluation.
    """
    scan = load_module(ROOT / "scan_posttrain.py", "scan_posttrain_probe")

    fake = {
        "metrics": {"accuracy": 0.5, "fid": 100.0, "kid": 0.05, "precision": 0.4,
                    "recall": 0.4, "memorization_rate": 0.0},
        "marks": {"total": 7, "per_metric": {k: {"tier": "Pass", "marks": 2}
                                             for k in ("accuracy", "fid", "recall")}},
        "resources": {"seconds_per_1000": 1.0},
        "checks": {"deterministic": True, "single_class_batch_accuracy": 0.5,
                   "mixed_batch_accuracy": 0.5},
        "violations": [],
    }
    out_dir = tmp / "scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    @contextmanager
    def patched():
        real = scan.subprocess.call

        def stub(cmd, **kw):
            calls.append(list(cmd))
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps(fake), encoding="utf-8")
            return 0

        scan.subprocess.call = stub
        try:
            yield
        finally:
            scan.subprocess.call = real

    with patched():
        first = scan.run("contrast250", n_per_class=1000, seed=0, weights=toy / "weights.pt",
                         out_dir=out_dir, ckpt="ARMTAG")
        p = Path(first["json"])
        md5_before = p.read_bytes()
        second = scan.run("contrast250", n_per_class=1000, seed=0, weights=toy / "weights.pt",
                          out_dir=out_dir, ckpt="ARMTAG")

    named = p.name == "contrast250__ARMTAG_n1000.json"
    refused = second.get("exit") == "refused-to-overwrite"
    untouched = p.read_bytes() == md5_before
    only_one = len(calls) == 1
    ok = named and refused and untouched and only_one and not first.get("failed")
    record(5, "scan-tag", ok,
           f"file {p.name} (tag in name={named}), second run {second.get('exit')}, "
           f"bytes unchanged={untouched}, evaluator calls={len(calls)}")


# ----------------------------------------------------------------------------------- 6. artefacts
def check_artefacts(tmp: Path, toy: Path) -> None:
    cfg = json.loads((toy / "config.json").read_text(encoding="utf-8"))
    losses = (toy / "losses.csv").read_text(encoding="utf-8").strip().splitlines()
    n_loss = len(losses) - 1
    first_s = losses[1].split(",")[1]
    last_s = losses[-1].split(",")[1]
    png = toy / "samples_final.png"
    # losses.csv is written as "%.5f", so the comparison is done at that precision: the json must
    # carry the same numbers (full float) that the csv rounded, not a number from another run.
    same_first = f"{cfg['loss_first']:.5f}" == first_s
    same_last = f"{cfg['loss_last']:.5f}" == last_s
    ok = (cfg["steps"] == n_loss == 6 and cfg["rows"] == 24 and same_first and same_last
          and png.exists() and png.stat().st_size > 0 and Path(cfg["weights"]).exists())
    record(6, "artefacts", ok,
           f"config steps={cfg['steps']} rows={cfg['rows']}; loss_first {cfg['loss_first']:.5f} "
           f"== csv {first_s} ({same_first}), loss_last {cfg['loss_last']:.5f} == csv {last_s} "
           f"({same_last}); {n_loss} loss rows; samples_final.png "
           f"{png.stat().st_size if png.exists() else 0} B; weights.pt present="
           f"{Path(cfg['weights']).exists()}")


# ---------------------------------------------------------------------------------- 7. gamma zero
def check_gamma_zero(tmp: Path) -> None:
    from diffusion import DiffusionModel, load_payload

    torch.manual_seed(0)
    model = DiffusionModel(base=16, emb=32, n_steps=100)
    x0 = torch.rand(8, 3, 32, 32) * 2 - 1
    y = torch.tensor([0, 1] * 4, dtype=torch.long)
    # eval() is required, not cosmetic: the UNet contains `nn.Dropout(0.1)`, so in train mode two
    # calls with the same t and eps still differ by ~6e-3 (measured). The assertion is about the
    # loss arithmetic, so the stochastic layer has to be out of the picture. Training itself runs
    # in train mode, where the dropout is intended.
    model.eval()

    # Same generator seed => the same t and the same eps => the two losses must be bit-identical.
    a = model.loss(x0, y, generator=torch.Generator().manual_seed(99))
    b = model.loss(x0, y, generator=torch.Generator().manual_seed(99), snr_gamma=0.0)
    c = model.loss(x0, y, generator=torch.Generator().manual_seed(99), snr_gamma=0.0, p_uncond=0.0)
    bitwise = torch.equal(a, b) and torch.equal(a, c)

    default = inspect.signature(DiffusionModel.loss).parameters["snr_gamma"].default
    # The payload has to carry the tiny architecture explicitly; `load_payload` fills in the
    # 64/128/1000 defaults for anything missing, which is what a real checkpoint always has.
    payload = {"kind": "ddpm", "state_dict": model.state_dict(), "base": 16, "emb": 32,
               "n_steps": 100, "ddim_steps": 5, "cond": "add", "null_token": False}
    loaded_default = inspect.signature(load_payload(payload).loss).parameters["snr_gamma"].default
    src = (ROOT / "train_diffusion.py").read_text(encoding="utf-8")
    trainer_zero = 'gamma = args.snr_gamma if args.loss_weight == "minsnr" else 0.0' in src

    ok = bitwise and default == 0.0 and loaded_default == 0.0 and trainer_zero
    record(7, "gamma-zero", ok,
           f"L(gamma=0) bitwise equal to L_simple: {bitwise} (|diff| {float((a - b).abs()):.1e}); "
           f"loss() default {default}, loaded default {loaded_default}, "
           f"trainer passes 0 when --loss-weight none: {trainer_zero}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--json", default="results/ab/cross_assert.json")
    ap.add_argument("--keep", action="store_true", help="keep the temporary toy run")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(False)   # cuDNN drift is measured, not forbidden
    wanted = set(args.only or range(1, 8))
    tmp = Path(tempfile.mkdtemp(prefix="cross_assert_"))
    print(f"tmp: {tmp}\n")
    try:
        toy = None
        if wanted & {4, 5, 6}:
            toy = toy_run(tmp)
        if 1 in wanted:
            check_noise_dist(tmp)
        if 2 in wanted:
            check_device(tmp)
        if 3 in wanted:
            check_interface(tmp)
        if 4 in wanted:
            check_ckpt_every(tmp, toy)
        if 5 in wanted:
            check_scan_tag(tmp, toy)
        if 6 in wanted:
            check_artefacts(tmp, toy)
        if 7 in wanted:
            check_gamma_zero(tmp)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if not r["ok"]]
    out = Path(args.json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": RESULTS, "passed": len(RESULTS) - len(failed),
                               "failed": len(failed)}, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} green -> {out.relative_to(ROOT)}")
    if failed:
        print("GATE CLOSED: " + ", ".join(f"[{r['id']}] {r['name']}" for r in failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
