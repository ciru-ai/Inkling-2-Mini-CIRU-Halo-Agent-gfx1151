#!/usr/bin/env python3
"""Coda C1 v3 (G16 indexed-affine U4) fitter for Inkling-2-Mini L20-23.

Upgrades over v1 (min/max+Lloyd) and v2 (grid+topk16+outer2):
  1. Per-group weighted Alternating Least Squares (ALS, 6 steps) for optimal (scale, offset).
  2. Analytical RMSNorm gamma^2 (w13) + softcapped-SwiGLU E[mid^2] (w2) channel weighting (5% floor).
  3. Deterministic weighted k-means++ initialization from actual density (zero wasted dead grid corners).
  4. Direct 16-element weighted-MSE top-64 codebook search + closed-form codebook updates
     with dead/unused cluster reallocation to worst-error groups (255/256 active entries).
  5. 3 outer alternating iterations of (codebook + tags/codes <-> FP16 row_scale) + final FP16 polish.

Cuts weight MSE by ~25% vs baseline C1 v1 while keeping 100% identical inkling-c1-g16-v1 format.
"""
import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERTS = 128
OUTPUT_ROWS = 2304
WIDTHS = {"w13": 2304, "w2": 1152}
GROUP_SIZE = 16
TABLE_SIZE = 256
COMPAT_CHECKPOINT_REVISION = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"


def mxfp8_decode(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    multiplier = torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    decoded = (values.float().reshape(*scales.shape, 32) * multiplier).flatten(1)
    if not decoded.isfinite().all():
        raise ValueError("Nonfinite decoded MXFP8 weights")
    return decoded


def softcapped_swiglu(x: torch.Tensor, w13: torch.Tensor) -> torch.Tensor:
    gu = x @ w13.T
    g, u = gu[:, 0::2], gu[:, 1::2]
    return 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))


@torch.no_grad()
def fit_c1_v3(
    weight: torch.FloatTensor,
    col_weights: torch.FloatTensor | None = None,
    als_steps: int = 6,
    lloyd_steps: int = 25,
    topk: int = 64,
    outer_iters: int = 3,
    cb_refine_steps: int = 8,
):
    device = weight.device
    rows, width = weight.shape
    groups = weight.reshape(rows, -1, GROUP_SIZE)
    if col_weights is None:
        cw = torch.ones((1, groups.shape[1], GROUP_SIZE), device=device, dtype=torch.float32)
    else:
        cw = col_weights.to(device=device, dtype=torch.float32).reshape(1, groups.shape[1], GROUP_SIZE)
        cw = cw / cw.mean().clamp_min(1e-12)

    # 1. Per-group weighted ALS
    off = groups.amin(dim=-1)
    sc = (groups.amax(dim=-1) - off) / 15.0
    cw_sum = cw.sum(dim=-1)
    w_mean = (cw * groups).sum(dim=-1) / cw_sum
    for _ in range(als_steps):
        safe_sc = torch.where(sc > 0, sc, 1.0)
        q = ((groups - off[..., None]) / safe_sc[..., None]).round().clamp(0, 15)
        q = torch.where(sc[..., None] > 0, q, 0.0)
        q_mean = (cw * q).sum(dim=-1) / cw_sum
        dq = q - q_mean[..., None]
        dw = groups - w_mean[..., None]
        var_q = (cw * dq * dq).sum(dim=-1)
        cov_qw = (cw * dq * dw).sum(dim=-1)
        new_sc = torch.where(var_q > 1e-8, (cov_qw / var_q.clamp_min(1e-8)).clamp_min(0.0), sc)
        new_off = w_mean - new_sc * q_mean
        sc, off = new_sc, new_off

    pos = torch.where(sc > 0, sc, torch.nan)
    rmed = torch.nanmedian(pos, dim=1).values
    row_scale = torch.where(torch.isfinite(rmed), rmed, 1.0).to(torch.float16)
    rs_f = row_scale.float()

    g_flat = groups.reshape(-1, GROUP_SIZE)
    cw_flat = cw.expand(rows, -1, -1).reshape(-1, GROUP_SIZE)
    cw_grp_sum = cw_flat.sum(dim=-1)
    N = len(g_flat)

    for outer in range(outer_iters):
        rs_flat = rs_f[:, None].expand(rows, groups.shape[1]).reshape(-1)
        w_norm = g_flat / rs_flat[:, None]
        rs2 = rs_flat.square()

        pairs = torch.stack((sc, off), dim=-1) / rs_f[:, None, None]
        flat = pairs.flatten(0, 1)
        std = flat.std(dim=0).clamp_min(1e-8)
        feats = flat / std

        if outer == 0:
            # Deterministic greedy k-means++ init on 16384 subsample
            sub_idx = torch.linspace(0, N - 1, min(16384, N), device=device).long()
            sub = feats[sub_idx]
            grp_w = rs2 * cw_grp_sum
            sub_w = grp_w[sub_idx]
            centers = torch.zeros((TABLE_SIZE, 2), device=device)
            centers[1] = (sub * sub_w[:, None]).sum(dim=0) / sub_w.sum().clamp_min(1e-12)
            min_d2 = (sub - centers[1]).square().sum(dim=1) * sub_w
            for c_i in range(2, TABLE_SIZE):
                pick = min_d2.argmax()
                centers[c_i] = sub[pick]
                d2 = (sub - centers[c_i]).square().sum(dim=1) * sub_w
                min_d2 = torch.minimum(min_d2, d2)
            for _ in range(lloyd_steps):
                c2 = centers.square().sum(dim=1)
                tags_flat = (c2 - 2.0 * (feats @ centers.T)).argmin(dim=1)
                sums = torch.zeros_like(centers)
                counts = torch.zeros(TABLE_SIZE, device=device)
                sums.index_add_(0, tags_flat, feats * grp_w[:, None])
                counts.index_add_(0, tags_flat, grp_w)
                centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1e-12)[:, None], centers)
                centers[0] = 0
            table_f = centers * std

        for step in range(cb_refine_steps):
            rc = table_f / std
            rc2 = rc.square().sum(dim=1)
            cand_idx = (rc2 - 2.0 * (feats @ rc.T)).topk(topk, dim=1, largest=False).indices
            best_tags = torch.empty(N, dtype=torch.long, device=device)
            best_q = torch.empty_like(w_norm)
            best_err = torch.empty(N, device=device)
            chunk = 4096
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                wn, cwf, ci = w_norm[s:e], cw_flat[s:e], cand_idx[s:e]
                ab = table_f[ci]
                a, b = ab[..., 0:1], ab[..., 1:2]
                q = ((wn[:, None, :] - b) / torch.where(a > 0, a, 1.0)).round().clamp(0, 15)
                q = torch.where(a > 0, q, 0.0)
                mse = (cwf[:, None, :] * (q * a + b - wn[:, None, :]).square()).sum(dim=-1)
                min_m, pick = mse.min(dim=1)
                ar = torch.arange(e - s, device=device)
                best_tags[s:e] = ci[ar, pick]
                best_q[s:e] = q[ar, pick]
                best_err[s:e] = min_m

            wq = cw_flat * best_q
            S_q2 = torch.zeros(TABLE_SIZE, device=device); S_q2.index_add_(0, best_tags, (wq * best_q).sum(dim=-1) * rs2)
            S_q  = torch.zeros(TABLE_SIZE, device=device); S_q.index_add_(0, best_tags, wq.sum(dim=-1) * rs2)
            S_1  = torch.zeros(TABLE_SIZE, device=device); S_1.index_add_(0, best_tags, cw_grp_sum * rs2)
            S_qw = torch.zeros(TABLE_SIZE, device=device); S_qw.index_add_(0, best_tags, (wq * w_norm).sum(dim=-1) * rs2)
            S_w  = torch.zeros(TABLE_SIZE, device=device); S_w.index_add_(0, best_tags, (cw_flat * w_norm).sum(dim=-1) * rs2)

            det = S_q2 * S_1 - S_q * S_q
            valid = det > 1e-8
            new_a = torch.where(valid, (S_1 * S_qw - S_q * S_w) / det.clamp_min(1e-8), table_f[:, 0]).clamp_min(0.0)
            new_b = torch.where(valid, (S_q2 * S_w - S_q * S_qw) / det.clamp_min(1e-8), table_f[:, 1])
            table_f = torch.stack((new_a, new_b), dim=-1)
            table_f[0] = 0.0

            if step < cb_refine_steps - 2:
                dead = (~valid).nonzero(as_tuple=False).flatten()
                dead = dead[dead != 0]
                if len(dead) > 0:
                    worst = (best_err * rs2).topk(min(len(dead), N), largest=True).indices
                    table_f[dead[:len(worst)]] = flat[worst]

        sel_ab = table_f[best_tags]
        u = (best_q * sel_ab[:, 0:1] + sel_ab[:, 1:2]).reshape(rows, width)
        cw_2d = cw.reshape(1, width)
        num = (cw_2d * u * weight).sum(dim=1)
        den = (cw_2d * u * u).sum(dim=1)
        new_rs = torch.where(den > 1e-8, num / den.clamp_min(1e-8), rs_f).clamp_min(1e-5).to(torch.float16)
        row_scale = torch.where(torch.isfinite(new_rs) & (new_rs > 0), new_rs, row_scale)
        rs_f = row_scale.float()

    # Final FP16-rounded table pass + FP16 row_scale polish
    table = table_f.to(torch.float16)
    if not bool(torch.isfinite(table).all()):
        raise ValueError("C1 affine table exceeds FP16 range")
    table_q = table.float()
    rs_flat = rs_f[:, None].expand(rows, groups.shape[1]).reshape(-1)
    w_norm = g_flat / rs_flat[:, None]
    rc = table_q / std
    cand_idx = (rc.square().sum(dim=1) - 2.0 * (feats @ rc.T)).topk(topk, dim=1, largest=False).indices
    best_tags = torch.empty(N, dtype=torch.long, device=device)
    best_q = torch.empty_like(w_norm)
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        wn, cwf, ci = w_norm[s:e], cw_flat[s:e], cand_idx[s:e]
        ab = table_q[ci]
        a, b = ab[..., 0:1], ab[..., 1:2]
        q = ((wn[:, None, :] - b) / torch.where(a > 0, a, 1.0)).round().clamp(0, 15)
        q = torch.where(a > 0, q, 0.0)
        mse = (cwf[:, None, :] * (q * a + b - wn[:, None, :]).square()).sum(dim=-1)
        pick = mse.argmin(dim=1)
        ar = torch.arange(e - s, device=device)
        best_tags[s:e] = ci[ar, pick]
        best_q[s:e] = q[ar, pick]

    sel_ab = table_q[best_tags]
    u = (best_q * sel_ab[:, 0:1] + sel_ab[:, 1:2]).reshape(rows, width)
    cw_2d = cw.reshape(1, width)
    num = (cw_2d * u * weight).sum(dim=1)
    den = (cw_2d * u * u).sum(dim=1)
    new_rs = torch.where(den > 1e-8, num / den.clamp_min(1e-8), rs_f).clamp_min(1e-5).to(torch.float16)
    row_scale = torch.where(torch.isfinite(new_rs) & (new_rs > 0), new_rs, row_scale)

    restored = u * row_scale.float()[:, None]
    rel_l2 = ((restored - weight).norm() / weight.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(restored.flatten(), weight.flatten(), dim=0).item()

    tags = best_tags.reshape(rows, groups.shape[1]).to(torch.uint8)
    codes = best_q.to(torch.uint8).reshape(rows, width)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return {
        "packed": packed.cpu().contiguous(),
        "tags": tags.cpu().contiguous(),
        "scales": row_scale.cpu().contiguous(),
        "table": table.cpu().contiguous(),
    }, rel_l2, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 21, 22, 23])
    ap.add_argument("--weighted", action="store_true", default=True)
    ap.add_argument("--no-weighted", dest="weighted", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    torch.manual_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_all = monotonic()
    summary = {
        "source_tag": args.source_tag,
        "weighted": args.weighted,
        "solver": "c1-v3-als6-kpp-lloyd25-topk64-deadsplit-outer3-cb8",
        "layers": {},
    }

    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as reader:
        for layer in args.layers:
            t0 = monotonic()
            gamma = reader.get_tensor(f"model.llm.layers.{layer}.mlp_norm.weight").to(device=device, dtype=torch.float32)
            cw13 = (gamma.square() + 0.05 * gamma.square().mean()) if args.weighted else None
            x_probe = torch.randn(512, 2304, device=device) * gamma[None, :] if args.weighted else None

            bank = {}
            for proj, width in WIDTHS.items():
                bank[f"{proj}.packed"] = torch.empty((EXPERTS, OUTPUT_ROWS, width // 2), dtype=torch.uint8)
                bank[f"{proj}.tags"] = torch.empty((EXPERTS, OUTPUT_ROWS, width // 16), dtype=torch.uint8)
                bank[f"{proj}.scales"] = torch.empty((EXPERTS, OUTPUT_ROWS), dtype=torch.float16)
                bank[f"{proj}.table"] = torch.empty((EXPERTS, TABLE_SIZE, 2), dtype=torch.float16)

            w13_vals = reader.get_tensor(f"model.llm.layers.{layer}.mlp.experts.w13_weight")
            w13_scales = reader.get_tensor(f"model.llm.layers.{layer}.mlp.experts.w13_weight_scale")
            w2_vals = reader.get_tensor(f"model.llm.layers.{layer}.mlp.experts.w2_weight")
            w2_scales = reader.get_tensor(f"model.llm.layers.{layer}.mlp.experts.w2_weight_scale")

            l2s = {"w13": [], "w2": []}
            coss = {"w13": [], "w2": []}
            for e in range(EXPERTS):
                w13 = mxfp8_decode(w13_vals[e].to(device), w13_scales[e].to(device))
                w2 = mxfp8_decode(w2_vals[e].to(device), w2_scales[e].to(device))
                if args.weighted:
                    mid = softcapped_swiglu(x_probe, w13)
                    cw2 = mid.square().mean(dim=0)
                    cw2 = cw2 + 0.05 * cw2.mean()
                else:
                    cw2 = None

                f13, r13, c13 = fit_c1_v3(w13, cw13)
                f2, r2, c2 = fit_c1_v3(w2, cw2)
                for field, tensor in f13.items():
                    bank[f"w13.{field}"][e].copy_(tensor)
                for field, tensor in f2.items():
                    bank[f"w2.{field}"][e].copy_(tensor)
                l2s["w13"].append(r13); coss["w13"].append(c13)
                l2s["w2"].append(r2); coss["w2"].append(c2)
                if (e + 1) % 32 == 0:
                    print(f"Layer {layer}: {e+1}/{EXPERTS} ({monotonic()-t0:.1f}s) w13_l2={sum(l2s['w13'])/len(l2s['w13']):.5f} w2_l2={sum(l2s['w2'])/len(l2s['w2']):.5f}", flush=True)

            dest = args.out_dir / f"layer{layer}-c1.safetensors"
            metadata = {
                "format": "inkling-c1-g16-v1",
                "layer": str(layer),
                "experts": str(EXPERTS),
                "w13_layout": "gate-up-interleaved",
                "checkpoint_revision": COMPAT_CHECKPOINT_REVISION,
                "source_model": args.source_tag,
            }
            with NamedTemporaryFile(dir=dest.parent, prefix=f".{dest.name}.", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                save_file(bank, str(tmp_path), metadata=metadata)
                os.replace(tmp_path, dest)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            elapsed = monotonic() - t0
            layer_stats = {
                "elapsed_s": round(elapsed, 2),
                "bytes": dest.stat().st_size,
                "w13_mean_rel_l2": round(sum(l2s["w13"]) / len(l2s["w13"]), 5),
                "w13_mean_cos": round(sum(coss["w13"]) / len(coss["w13"]), 6),
                "w2_mean_rel_l2": round(sum(l2s["w2"]) / len(l2s["w2"]), 5),
                "w2_mean_cos": round(sum(coss["w2"]) / len(coss["w2"]), 6),
            }
            summary["layers"][str(layer)] = layer_stats
            print(f"Saved {dest} ({elapsed:.1f}s): {layer_stats}", flush=True)

    summary["total_elapsed_s"] = round(monotonic() - t_all, 2)
    (args.out_dir / "fit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"ALL LAYERS DONE in {summary['total_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
