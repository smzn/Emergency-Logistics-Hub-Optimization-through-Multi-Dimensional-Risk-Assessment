#!/usr/bin/env bash
# run_once_v3.sh
# ================================
# Virtual_Region_Run_v3.py を 1回だけ実行するスクリプト
# - 既定: カテゴリ方式（E/Vとも最上位クラス）
# - 環境変数で簡単に条件を切替可能
# ================================
set -euo pipefail

# ------- 基本パラメータ（必要に応じて変更）-------
K="${K:-5}"                    # 既存デポ数
RADIUS="${RADIUS:-12}"         # サービス半径 [km]
SCENARIO="${SCENARIO:-random}" # even_k | random
MASTER_SEED="${MASTER_SEED:-123456}"
RANDOMIZE_DEPOTS="${RANDOMIZE_DEPOTS:-1}"  # 1なら --randomize_depots を付与
OUT_BASE_ROOT="${OUT_BASE_ROOT:-outputs_v3_single}"

# ------- HR 抽出方式 -------
HR_MODE="${HR_MODE:-category}" # category | threshold

# --- カテゴリ方式用（1-based 指定。例: "3" や "2,3"）---
QE="${QE:-3}"
QV="${QV:-3}"
E_LEVELS="${E_LEVELS:-}"   # 例: "3" または "2,3"（未指定なら最上位のみ）
V_LEVELS="${V_LEVELS:-}"   # 例: "3" または "2,3"（未指定なら最上位のみ）

# --- 閾値方式用 ---
THR="${THR:-0.95}"

# ------- 付録パラメータ（必要なら調整）-------
TARGET_WCOV="${TARGET_WCOV:-1.0}"
KMAX="${KMAX:-}"                     # 空なら未指定
STRIDE="${STRIDE:-1}"
W1="${W1:-0.6}"
W2="${W2:-0.25}"
W3="${W3:-0.15}"
KCAP="${KCAP:-10}"

# ------- 出力先（タイムスタンプ付でユニーク化）-------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE_ROOT}/${TS}"
AGG_OUT="${OUT_BASE_ROOT}/summary_${HR_MODE}.csv"

# ------- 引数の組立て -------
ARGS=( --k "${K}"
       --radius "${RADIUS}"
       --scenario "${SCENARIO}"
       --master_seed "${MASTER_SEED}"
       --out_base "${OUT_BASE}"
       --target_wcov "${TARGET_WCOV}"
       --stride "${STRIDE}"
       --w1 "${W1}" --w2 "${W2}" --w3 "${W3}"
       --kcap "${KCAP}"
       --aggregate_out "${AGG_OUT}" )

# 追加デポ上限
if [[ -n "${KMAX}" ]]; then
  ARGS+=( --kmax "${KMAX}" )
fi

# 既存デポのみ毎回ランダム
if [[ "${RANDOMIZE_DEPOTS}" == "1" ]]; then
  ARGS+=( --randomize_depots )
fi

# HR 抽出方式
if [[ "${HR_MODE}" == "category" ]]; then
  ARGS+=( --hr_mode category --QE "${QE}" --QV "${QV}" )
  # E/V レベル（未指定なら最上位のみ→ランナー側既定を利用するので渡さなくてOK）
  if [[ -n "${E_LEVELS}" ]]; then
    ARGS+=( --E_levels "${E_LEVELS}" )
  fi
  if [[ -n "${V_LEVELS}" ]]; then
    ARGS+=( --V_levels "${V_LEVELS}" )
  fi
elif [[ "${HR_MODE}" == "threshold" ]]; then
  ARGS+=( --hr_mode threshold --thr "${THR}" )
else
  echo "[ERROR] HR_MODE must be 'category' or 'threshold' (got: ${HR_MODE})" >&2
  exit 1
fi

# ------- 実行 -------
echo "[INFO] Running Virtual_Region_Run_v3.py ..."
echo "python3 Virtual_Region_Run_v3.py ${ARGS[*]}"
python3 Virtual_Region_Run_v3.py "${ARGS[@]}"

echo "[DONE] Output base: ${OUT_BASE}"
echo "[DONE] Aggregated summary: ${AGG_OUT}"
