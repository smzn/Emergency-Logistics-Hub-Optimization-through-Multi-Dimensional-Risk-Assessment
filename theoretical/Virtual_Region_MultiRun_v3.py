# -*- coding: utf-8 -*-
"""
Virtual_Region_MultiRun_v3.py
---------------------------------
Multi-run driver for TSHCO using Virtual_Region_v3 (category/threshold both supported).

- v3のポイント
  * HR集合Hの抽出: 分位カテゴリ(category)方式がデフォルト
    - E, V を QE, QV 分位に区分し、E_levels × V_levels で高リスク集合を定義
    - --E_levels/--V_levels は 1-based（例: "3" / "2,3"）
  * 閾値(threshold)方式も互換で利用可 (--hr_mode threshold)

- 本ドライバの役割
  * 1回ずつ VirtualRegion.run_once(...) を呼び出し、各ランの出力は
    out_base/<tag>/run_<NNN>/... の下に作成
  * ランごとの要約を集約CSVに追記し、併せて統計CSVも生成

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations
import argparse
import os
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from Virtual_Region_v3 import VirtualRegion, RunParams


# ----------------------------
# helpers
# ----------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def rng_from_seed(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = np.random.SeedSequence().entropy
    return np.random.default_rng(int(seed) % (2**32 - 1))

def parse_levels_1based(s: str | None, Q: int) -> List[int] | None:
    """
    "3" / "2,3" / "(2,3)" → 0-based list（内部表現）に変換。
    None または空文字は None を返す（＝既定: 最上位のみを v3 側で採用）。
    """
    if not s:
        return None
    t = s.strip().replace("(", "").replace(")", "")
    if not t:
        return None
    nums = [int(x) for x in t.split(",")]
    for v in nums:
        if v < 1 or v > Q:
            raise argparse.ArgumentTypeError(f"Class out of range: {v} (must be 1..{Q})")
    return [v - 1 for v in nums]  # 0-based

def to_levels_str_0based(lvls: List[int] | None) -> str:
    if not lvls:
        return "[]"
    return "[" + ",".join(str(x) for x in lvls) + "]"


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="TSHCO multi-run driver (v3, category-friendly)")

    # --- core knobs ---
    ap.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    ap.add_argument("--k", type=int, default=5, help="Existing depots (MCLP)")
    ap.add_argument("--radius", type=float, default=12.0, help="Service radius [km]")
    ap.add_argument("--scenario", type=str, default="random", choices=["even_k", "random"])

    # --- HR extraction mode ---
    ap.add_argument("--hr_mode", type=str, default="category", choices=["category", "threshold"])
    ap.add_argument("--QE", type=int, default=3)
    ap.add_argument("--QV", type=int, default=3)
    ap.add_argument("--E_levels", type=str, default=None,
                    help="Target E classes (1-based; e.g., '3' or '2,3'). Omit for top-only.")
    ap.add_argument("--V_levels", type=str, default=None,
                    help="Target V classes (1-based; e.g., '3' or '2,3'). Omit for top-only.")
    ap.add_argument("--thr", type=float, default=0.95,
                    help="Used only when --hr_mode threshold (kept for compatibility).")

    # --- SCLP / greedy options ---
    ap.add_argument("--target_wcov", type=float, default=1.0)
    ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="Candidate stride (>=1) for SCLP greedy")

    # --- composite S weights ---
    ap.add_argument("--w1", type=float, default=0.6)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--w3", type=float, default=0.15)
    ap.add_argument("--kcap", type=int, default=10)

    # --- seeds & randomness ---
    ap.add_argument("--master_seed", type=int, default=None,
                    help="Master seed for deriving per-run seeds if set.")
    ap.add_argument("--randomize_depots", action="store_true",
                    help="Randomize ONLY existing depot seed each run.")

    # --- output ---
    ap.add_argument("--out_base", type=str, default="outputs_multirun_v3",
                    help="Base folder to store all runs.")
    ap.add_argument("--aggregate_out", type=str, default=None,
                    help="If set, write/update an aggregate CSV at this path. "
                         "Default: <out_base>/<tag>/aggregate_summary.csv")

    args = ap.parse_args()

    # --- sanity ---
    if args.hr_mode == "category":
        if args.QE < 1 or args.QV < 1:
            raise ValueError("QE and QV must be >= 1 for category mode.")

    # --- prepare out base and tag ---
    tag = f"multirun_{now_tag()}"
    root = os.path.join(args.out_base, tag)
    os.makedirs(root, exist_ok=True)

    # aggregate file path
    agg_path = args.aggregate_out if args.aggregate_out else os.path.join(root, "aggregate_summary.csv")

    # seed generator
    mrng = rng_from_seed(args.master_seed)

    # parse E/V levels (0-based) if provided
    E_levels0 = parse_levels_1based(args.E_levels, args.QE) if args.hr_mode == "category" and args.E_levels else None
    V_levels0 = parse_levels_1based(args.V_levels, args.QV) if args.hr_mode == "category" and args.V_levels else None

    # record meta
    meta = {
        "tag": tag,
        "n_runs": args.n_runs,
        "args": vars(args),
        "version": "Virtual_Region_MultiRun_v3",
    }
    with open(os.path.join(root, "multirun_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # aggregate rows
    rows = []

    for r in range(1, args.n_runs + 1):
        run_dir = os.path.join(root, f"run_{r:03d}")
        os.makedirs(run_dir, exist_ok=True)

        # per-run master seed（未指定なら乱数から）
        per_seed = int(mrng.integers(1, 2**31 - 1))

        # build RunParams (v3)
        params = RunParams(
            k=args.k,
            radius_km=args.radius,
            thr=args.thr,
            scenario=args.scenario,
            target_wcov=args.target_wcov,
            kmax=args.kmax,
            stride_candidates=args.stride,
            master_seed=per_seed,
            out_base=run_dir,          # 各ランの下に v3 がさらに run フォルダを作成
            w1=args.w1, w2=args.w2, w3=args.w3, kcap=args.kcap,
            randomize_depots=args.randomize_depots,

            # v3 HR settings
            hr_mode=args.hr_mode,
            QE=args.QE, QV=args.QV,
            E_levels=E_levels0,
            V_levels=V_levels0,
        )

        # run
        region = VirtualRegion()
        res = region.run_once(params)

        # one-line summary row
        row = {
            "run_dir": res.outdir,
            "run_index": r,
            "k": args.k,
            "R": args.radius,
            "thr": (args.thr if args.hr_mode == "threshold" else -1.0),
            "scenario": args.scenario,
            "target_wcov": args.target_wcov,
            "C_max": res.C_max,
            "wCov_max": res.wCov_max,
            "C_final": res.C_final,
            "wCov_final": res.wCov_final,
            "Coverage_final": res.Coverage_final,
            "k_added": res.k_added,
            "k_at95": res.k_at95,
            "AUC": res.AUC,
            "Gap": res.Gap,
            "S": res.S,
            "uncovered_count": res.uncovered_count,
            "master_seed": per_seed,
            "randomize_depots": bool(args.randomize_depots),
            "w1": args.w1, "w2": args.w2, "w3": args.w3, "kcap": args.kcap,
            # v3: HR info
            "HR_mode": args.hr_mode,
            "QE": args.QE,
            "QV": args.QV,
            "E_levels": to_levels_str_0based(E_levels0) if E_levels0 is not None else "[]",
            "V_levels": to_levels_str_0based(V_levels0) if V_levels0 is not None else "[]",
        }
        rows.append(row)

        # write per-run quick CSV
        pd.DataFrame([row]).to_csv(os.path.join(run_dir, "run_quick_summary.csv"), index=False)

    # aggregate CSV
    df = pd.DataFrame(rows)
    df.to_csv(agg_path, index=False)

    # simple stats
    stats = df.describe(include="number").T
    stats.to_csv(os.path.join(root, "aggregate_stats.csv"))

    print(f"[MULTIRUN v3] out_base={args.out_base} tag={tag} n_runs={args.n_runs}")
    print(f"[OUT] aggregate: {agg_path}")
    print(f"[OUT] stats: {os.path.join(root, 'aggregate_stats.csv')}")


if __name__ == "__main__":
    main()
