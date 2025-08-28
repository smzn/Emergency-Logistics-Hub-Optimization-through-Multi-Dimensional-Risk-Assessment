#!/usr/bin/env bash
set -euo pipefail

# ======================================
# マルチ実行スクリプト（v3 / カテゴリ方式対応）
# - Virtual_Region_MultiRun_v3.py を呼び出す
# ======================================

# ========= ユーザ設定（環境変数で上書き可）=========
N_RUNS="${N_RUNS:-30}"          # 実行回数
K="${K:-5}"                      # 既存拠点数（MCLP）
RADIUS="${RADIUS:-12}"           # サービス半径 [km]
SCENARIO="${SCENARIO:-random}"   # even_k | random
MASTER_SEED="${MASTER_SEED:-0}"  # 0なら内部で自動

# HR 抽出方式
HR_MODE="${HR_MODE:-category}"   # category | threshold

# --- カテゴリ方式 ---
QE="${QE:-3}"
QV="${QV:-3}"
E_LEVELS="${E_LEVELS:-}"         # 例: "3" or "2,3"
V_LEVELS="${V_LEVELS:-}"         # 例: "3" or "2,3"

# --- 閾値方式 ---
THR="${THR:-0.95}"

# ランダム化設定
RANDOMIZE="${RANDOMIZE:-1}"      # 1なら --randomize_depots を付与

# 出力ルート
OUTROOT="${OUTROOT:-outputs_v3_multi}"
# ======================================

STAMP="$(date +%Y%m%d-%H%M%S)"

# ラベル
if [[ "${HR_MODE}" == "category" ]]; then
  ELBL="${E_LEVELS:-${QE}}"
  VLBL="${V_LEVELS:-${QV}}"
  TAG="cat_E${ELBL}of${QE}_V${VLBL}of${QV}"
else
  TAG="thr-${THR}"
fi

OUTDIR="${OUTROOT}/${STAMP}_${TAG}_R-${RADIUS}_k-${K}_${SCENARIO}"
mkdir -p "${OUTDIR}"

echo "===> Outdir: ${OUTDIR}"
echo "===> Runs  : ${N_RUNS}"
echo "===> Mode  : ${HR_MODE}"

# ===== 実行 =====
ARGS=( --n_runs "${N_RUNS}"
       --k "${K}"
       --radius "${RADIUS}"
       --scenario "${SCENARIO}"
       --out_base "${OUTDIR}" )

if [[ "${MASTER_SEED}" -gt 0 ]]; then
  ARGS+=( --master_seed "${MASTER_SEED}" )
fi

if [[ "${RANDOMIZE}" == "1" ]]; then
  ARGS+=( --randomize_depots )
fi

if [[ "${HR_MODE}" == "category" ]]; then
  ARGS+=( --hr_mode category --QE "${QE}" --QV "${QV}" )
  if [[ -n "${E_LEVELS}" ]]; then ARGS+=( --E_levels "${E_LEVELS}" ); fi
  if [[ -n "${V_LEVELS}" ]]; then ARGS+=( --V_levels "${V_LEVELS}" ); fi
else
  ARGS+=( --hr_mode threshold --thr "${THR}" )
fi

# ---- 呼び出し ----
python3 Virtual_Region_MultiRun_v3.py "${ARGS[@]}"

echo "===> Finished. See ${OUTDIR}"
