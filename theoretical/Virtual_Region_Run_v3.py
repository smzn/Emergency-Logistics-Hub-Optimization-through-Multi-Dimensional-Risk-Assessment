# -*- coding: utf-8 -*-
# Virtual_Region_Run_v3.py
# 1回実行ランナー：図・CSV一式を出力し、必要ならサマリーCSVに追記します。
# v3: 閾値(threshold)方式 と カテゴリ(category)方式 の両対応

from __future__ import annotations
import argparse
import os
import pandas as pd

from Virtual_Region_v3 import VirtualRegion, RunParams  # ← v3 を呼ぶ

# ---- helpers ----
def parse_levels_1based(s: str, Q: int) -> list[int]:
    """
    "3" / "2,3" / "(2,3)" など 1-based 入力を 0-based に変換して返す。
    Q（分位数）チェックあり。
    """
    t = s.strip().replace("(", "").replace(")", "")
    if not t:
        return []
    nums = [int(x) for x in t.split(",")]
    for v in nums:
        if v < 1 or v > Q:
            raise argparse.ArgumentTypeError(f"Class out of range: {v} (must be 1..{Q})")
    # 0-based に
    return [v - 1 for v in nums]

def to_levels_str(lvls: list[int] | None) -> str:
    if not lvls:
        return "[]"
    return "[" + ",".join(str(x) for x in lvls) + "]"

def cat_label(QE: int, QV: int, E_levels0: list[int], V_levels0: list[int]) -> str:
    # 例: E3of3_V3of3 / E(2,3)of4_V(4)of4  ※ここは 1-based 表示
    def part(lbl, lvls, Q):
        lvls1 = [x + 1 for x in sorted(set(lvls))]
        if len(lvls1) == 1:
            return f"{lbl}{lvls1[0]}of{Q}"
        return f"{lbl}({','.join(str(x) for x in lvls1)})of{Q}"
    return part("E", E_levels0, QE) + "_" + part("V", V_levels0, QV)


def main():
    ap = argparse.ArgumentParser(description="Run one simulation (v3) and export figures/CSVs.")

    # 共通（従来と同じ）
    ap.add_argument("--k", type=int, default=5, help="Number of existing depots (MCLP stage).")
    ap.add_argument("--radius", type=float, default=12.0, help="Service radius [km].")
    ap.add_argument("--thr", type=float, default=0.95, help="Risk threshold tau for HR set (threshold mode only).")
    ap.add_argument("--scenario", type=str, default="random", choices=["even_k", "random"],
                    help="Existing depots placement scenario.")
    ap.add_argument("--target_wcov", type=float, default=1.0,
                    help="Target weighted coverage for SCLP greedy (<=1.0).")
    ap.add_argument("--kmax", type=int, default=None,
                    help="Max additional depots for SCLP greedy (None for no cap).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Candidate stride (>=1) for SCLP greedy.")
    ap.add_argument("--master_seed", type=int, default=2492289588,
                    help="Master seed (risk/layer/depot seeds are derived from this).")
    ap.add_argument("--randomize_depots", action="store_true",
                    help="Randomize ONLY existing depot seed each run (for non-repro fixed depots).")
    ap.add_argument("--out_base", type=str, default="outputs", help="Base output directory.")
    # Composite indicator weights
    ap.add_argument("--w1", type=float, default=0.6)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--w3", type=float, default=0.15)
    ap.add_argument("--kcap", type=int, default=10, help="Normalization cap for S indicator.")
    # 集計CSV（追記）
    ap.add_argument("--aggregate_out", type=str, default=None,
                    help="If set, append a one-line summary to this CSV.")

    # v3: HR 抽出方式
    ap.add_argument("--hr_mode", type=str, default="category", choices=["threshold", "category"],
                    help="How to define high-risk set H.")
    ap.add_argument("--QE", type=int, default=3, help="Number of quantile bins for E (>=1).")
    ap.add_argument("--QV", type=int, default=3, help="Number of quantile bins for V (>=1).")
    ap.add_argument("--E_levels", type=str, default=None,
                    help="Target E classes (1-based, comma separated). Example: '3' or '2,3'. If omitted, use top class only.")
    ap.add_argument("--V_levels", type=str, default=None,
                    help="Target V classes (1-based, comma separated). Example: '3' or '2,3'. If omitted, use top class only.")

    args = ap.parse_args()

    # 1-based → 0-based 変換（指定がない場合は「最上位のみ」）
    if args.hr_mode == "category":
        if args.QE < 1 or args.QV < 1:
            raise ValueError("QE and QV must be >= 1.")
        E_levels0 = parse_levels_1based(args.E_levels, args.QE) if args.E_levels else [args.QE - 1]
        V_levels0 = parse_levels_1based(args.V_levels, args.QV) if args.V_levels else [args.QV - 1]
    else:
        E_levels0, V_levels0 = None, None  # 閾値方式では未使用

    region = VirtualRegion()
    params = RunParams(
        k=args.k,
        radius_km=args.radius,
        thr=args.thr,
        scenario=args.scenario,
        target_wcov=args.target_wcov,
        kmax=args.kmax,
        stride_candidates=args.stride,
        master_seed=args.master_seed,
        out_base=args.out_base,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        kcap=args.kcap,
        randomize_depots=args.randomize_depots,

        # v3 追加
        hr_mode=args.hr_mode,
        QE=args.QE,
        QV=args.QV,
        E_levels=E_levels0,
        V_levels=V_levels0,
    )

    res = region.run_once(params)

    # ---- コンソール出力 ----
    print("[DONE] outdir:", res.outdir)
    print(f"  C_max={res.C_max:.3f}, wCov_max={res.wCov_max:.3f}")
    print(f"  C_final={res.C_final:.3f}, wCov_final={res.wCov_final:.3f}, Coverage_final={res.Coverage_final:.3f}")
    print(f"  k_added={res.k_added}, k@0.95={res.k_at95}, AUC={res.AUC:.3f}, Gap={res.Gap:.3f}, S={res.S:.3f}")
    print(f"  uncovered_count={res.uncovered_count}")

    # ---- 集計CSV（追記）----
    if args.aggregate_out:
        # HR ラベル（表示用）
        if args.hr_mode == "category":
            hr_label = cat_label(args.QE, args.QV, E_levels0, V_levels0)
            thr_out = -1.0  # category の場合は -1 にしておく（互換のため）
        else:
            hr_label = f">= {args.thr:.2f}"
            thr_out = args.thr

        row = {
            "outdir": res.outdir,
            "k": args.k,
            "R": args.radius,
            "thr": thr_out,
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
            "master_seed": args.master_seed,
            "randomize_depots": bool(args.randomize_depots),
            "w1": args.w1,
            "w2": args.w2,
            "w3": args.w3,
            "kcap": args.kcap,
            # v3 fields
            "HR_mode": args.hr_mode,
            "QE": args.QE,
            "QV": args.QV,
            "E_levels": to_levels_str(E_levels0) if E_levels0 is not None else "",
            "V_levels": to_levels_str(V_levels0) if V_levels0 is not None else "",
            "HR_label": hr_label,
        }
        os.makedirs(os.path.dirname(args.aggregate_out), exist_ok=True)
        if os.path.exists(args.aggregate_out):
            df = pd.read_csv(args.aggregate_out)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(args.aggregate_out, index=False)
        print("[AGG] wrote:", args.aggregate_out)


if __name__ == "__main__":
    main()

#python3 Virtual_Region_Run_v3.py \
#  --k 5 --radius 12 --scenario random \
#  --hr_mode category --QE 3 --QV 3 \
#  --master_seed 2492289588 --randomize_depots \
#  --out_base outputs_v3 \
#  --aggregate_out outputs_v3/summary_cat_E3of3_V3of3.csv
