# -*- coding: utf-8 -*-
"""
Virtual_Region_v3.py
---------------------------------
TSHCO（Two-Stage Hazard Coverage Optimization）用の仮想領域シミュレータ

v3 主な変更点（v2 → v3）:
- 高リスク集合 H の抽出を「閾値(threshold)方式」に加え「カテゴリ(category)方式」に対応
  * E, V を QE, QV 分位でクラス化（0..QE-1 / 0..QV-1）
  * 既定: 最上位×最上位（例: QE=QV=3 なら E=2 かつ V=2）
  * 任意に E_levels, V_levels（複数クラス）を指定可能
- 図/CSV/メタデータに HR_mode, 分位情報, カテゴリラベルを出力
- ファイル名・図タイトルをカテゴリ表記に対応（例: cat-E3of3_V3of3）

互換性:
- v2 の API/戻り値/ファイル構成は維持しつつ列を追加（coverage_summary.csv）
- hr_mode="threshold" を選べば従来どおりの挙動
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable, Literal, Union
import os
import numpy as np
import matplotlib.pyplot as plt

from Utils_v1_1 import (
    COLORS, slugify_float, derive_subseeds, build_outdir,
    save_run_metadata, save_json, save_csv,
    common_imshow, add_colorbar_outside, draw_circles, draw_points
)

# ================== Params & Results ==================
@dataclass
class RunParams:
    k: int
    radius_km: float
    thr: float
    scenario: str = "random"           # "even_k" | "random"
    target_wcov: float = 1.0
    kmax: Optional[int] = None
    stride_candidates: int = 1
    master_seed: Optional[int] = None
    out_base: str = "outputs"
    # for composite indicator / score
    w1: float = 0.6
    w2: float = 0.25
    w3: float = 0.15
    kcap: int = 10
    randomize_depots: bool = False   # True: 既存デポ位置だけ都度ランダム

    # === v3: カテゴリ方式のための追加パラメータ ===
    hr_mode: Literal["threshold", "category"] = "category"
    QE: int = 3               # E の分位数（既定: 3分位）
    QV: int = 3               # V の分位数（既定: 3分位）
    E_levels: Optional[List[int]] = None  # 許容 E クラス（0..QE-1）。未指定は [QE-1]
    V_levels: Optional[List[int]] = None  # 許容 V クラス（0..QV-1）。未指定は [QV-1]


@dataclass
class RunResult:
    C_max: float
    wCov_max: float
    C_final: float
    wCov_final: float
    Coverage_final: float
    k_added: int
    k_at95: Optional[int]
    AUC: float
    Gap: float
    S: float
    uncovered_count: int
    outdir: str


# ================== Core Class ==================
class VirtualRegion:
    """
    v3: 多次元リスク（E:災害リスク, V:社会的脆弱性, R=βE*E+βV*V） + 分位カテゴリ方式
        - E/V を QE/QV 分位に区分し、E_levels × V_levels で高リスク集合 H を定義
        - 既定: 最上位×最上位（E_levels=[QE-1], V_levels=[QV-1]）
    互換性:
        - v2 の入出力は維持。coverage_summary.csv に列を追加。
        - hr_mode="threshold" で従来の thr による H 抽出も選択可。
    """
    def __init__(self, width_km=70, height_km=50, cell_size_m=500):
        self.width_km = width_km
        self.height_km = height_km
        self.cell_km = cell_size_m / 1000.0
        self.n_cols = int(round(width_km / self.cell_km))
        self.n_rows = int(round(height_km / self.cell_km))
        self.extent = (0, width_km, 0, height_km)

        # 互換: v1_1 の名残（未使用でも保持）
        self.layer_probs: Dict[str, np.ndarray] = {}
        # v3 では self.risk = 統合リスク R
        self.risk: Optional[np.ndarray] = None

        # 多次元リスク保持
        self.E_map: Optional[np.ndarray] = None
        self.V_map: Optional[np.ndarray] = None
        self.R_map: Optional[np.ndarray] = None
        self.E_cls: Optional[np.ndarray] = None  # 0..QE-1
        self.V_cls: Optional[np.ndarray] = None
        self.E_q: Optional[Dict[str, Union[List[float], float]]] = None  # {"edges": [...]}
        self.V_q: Optional[Dict[str, Union[List[float], float]]] = None

    # ---------- 基本ユーティリティ ----------
    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        if mx <= mn:
            return np.zeros_like(x, dtype=float)
        return (x - mn) / (mx - mn)

    @staticmethod
    def _to_quantile_classes(values: np.ndarray, Q: int) -> Tuple[np.ndarray, List[float]]:
        """
        values を Q 分位で 0..Q-1 に割当て。返り値: (classes, edges)
        edges は 1/Q, 2/Q, ..., (Q-1)/Q 分位の値（境界値）リスト。
        """
        if Q <= 1:
            return np.zeros_like(values, dtype=np.int16), []
        flat = values.ravel()
        qs = [i / Q for i in range(1, Q)]
        edges = list(np.quantile(flat, qs))
        cls = np.searchsorted(edges, flat, side="right").astype(np.int16)  # 0..Q-1
        return cls.reshape(values.shape), edges

    @staticmethod
    def _make_category_mask(E_cls: np.ndarray, V_cls: np.ndarray,
                            E_levels: Iterable[int], V_levels: Iterable[int]) -> np.ndarray:
        Eok = np.isin(E_cls, np.array(list(E_levels), dtype=int))
        Vok = np.isin(V_cls, np.array(list(V_levels), dtype=int))
        return Eok & Vok

    @staticmethod
    def _slugify_str(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    @staticmethod
    def _cat_label(QE: int, QV: int, E_levels: Iterable[int], V_levels: Iterable[int]) -> str:
        # 例: E3of3_V3of3 / E(2,3)of3_V(3)of3
        e_sorted = sorted(set(E_levels))
        v_sorted = sorted(set(V_levels))
        def part(lbl, lvls, Q):
            if len(lvls) == 1:
                return f"{lbl}{lvls[0]+1}of{Q}"
            return f"{lbl}({','.join(str(x+1) for x in lvls)})of{Q}"
        return part("E", e_sorted, QE) + "_" + part("V", v_sorted, QV)

    def _layer_field(self, rng: np.random.Generator, *, num_spots=5, sigma_km=5.0, base=0.05) -> np.ndarray:
        """多点ガウス核の線形和（領域座標: km）"""
        yy, xx = np.mgrid[0:self.n_rows, 0:self.n_cols]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        field = np.full_like(xs, base, dtype=float)
        s2 = sigma_km ** 2
        for _ in range(num_spots):
            cx = rng.uniform(2.0, self.width_km - 2.0)
            cy = rng.uniform(2.0, self.height_km - 2.0)
            amp = float(rng.lognormal(mean=-0.1, sigma=0.5))
            field += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s2))
        return field

    @staticmethod
    def _to_prob_with_quantiles(F: np.ndarray) -> np.ndarray:
        """10%/90%分位に基づく傾き補正付きロジスティック写像（v1_1 の層生成に準拠）"""
        q10, q90 = np.quantile(F, [0.10, 0.90])
        k = 4.0 / max(q90 - q10, 1e-6)
        m = 0.5 * (q90 + q10)
        prob = 1.0 / (1.0 + np.exp(-k * (F - m)))
        return np.clip(prob, 0.0, 1.0)

    # ---------- 多次元リスク生成 ----------
    def generate_multidim_risk(
        self,
        layer_seed: int,
        alpha: Tuple[float, float, float] = (1/3, 1/3, 1/3),  # 災害層の重み
        w:     Tuple[float, float, float] = (1/3, 1/3, 1/3),  # 脆弱性要素の重み
        beta_E: float = 1.0,
        beta_V: float = 1.0,
        QE: int = 3,
        QV: int = 3,
    ) -> Dict[str, Union[np.ndarray, dict, list, int]]:
        """
        3ハザード層 + 3脆弱性要素 → E, V, R と 分位カテゴリ、既定の HighE∩HighV を返す
        """
        rng = np.random.default_rng(layer_seed)

        # --- 災害リスク層（例: 津波/洪水, 土砂, 地震） ---
        F1 = self._layer_field(rng, num_spots=6, sigma_km=5.0, base=0.05)
        F2 = self._layer_field(rng, num_spots=5, sigma_km=6.0, base=0.05)
        F3 = self._layer_field(rng, num_spots=4, sigma_km=7.0, base=0.05)
        h1 = self._to_prob_with_quantiles(F1)
        h2 = self._to_prob_with_quantiles(F2)
        h3 = self._to_prob_with_quantiles(F3)
        E = alpha[0] * h1 + alpha[1] * h2 + alpha[2] * h3
        E = np.clip(E, 0.0, 1.0)

        # --- 社会的脆弱性要素（例: 人口/高齢化, アクセス性, 地理的孤立） ---
        G1 = self._layer_field(rng, num_spots=7, sigma_km=6.0, base=0.10)  # 人口密度・高齢化
        G2 = self._layer_field(rng, num_spots=6, sigma_km=7.0, base=0.10)  # アクセス性
        G3 = self._layer_field(rng, num_spots=5, sigma_km=8.0, base=0.10)  # 地理的孤立
        z1 = self._minmax01(G1)
        z2 = self._minmax01(G2)
        z3 = self._minmax01(G3)
        V = w[0] * z1 + w[1] * z2 + w[2] * z3
        V = np.clip(V, 0.0, 1.0)

        # --- 統合リスク（[0,1] に正規化） ---
        R = beta_E * E + beta_V * V
        R = self._minmax01(R)

        # --- 分位クラス化 ---
        E_cls, E_edges = self._to_quantile_classes(E, QE)
        V_cls, V_edges = self._to_quantile_classes(V, QV)
        H_cat_mask = (E_cls == (QE - 1)) & (V_cls == (QV - 1))  # 既定: 最上位×最上位

        # 保持（描画/出力で再利用）
        self.E_map, self.V_map, self.R_map = E, V, R
        self.E_cls, self.V_cls = E_cls, V_cls
        self.E_q = {"edges": E_edges}
        self.V_q = {"edges": V_edges}

        return {
            "E": E, "V": V, "R": R,
            "E_cls": E_cls, "V_cls": V_cls,
            "E_edges": E_edges, "V_edges": V_edges,
            "H_cat_mask": H_cat_mask,
            "QE": QE, "QV": QV,
        }

    # ---------- v1_1 名残の互換（未使用） ----------
    def build_random_layers(self, layer_seed: int, L: int = 3) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(layer_seed)
        layers = {}
        yy, xx = np.mgrid[0:self.n_rows, 0:self.n_cols]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        for li in range(L):
            base = np.zeros_like(xs, dtype=float)
            M = rng.integers(2, 5)
            sigma = rng.uniform(3.0, 6.0)  # km
            s2 = sigma**2
            for _ in range(M):
                cx = rng.uniform(8, self.width_km-8)
                cy = rng.uniform(5, self.height_km-5)
                amp = float(rng.lognormal(mean=-0.1, sigma=0.5))
                base += amp * np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*s2))
            q10, q90 = np.quantile(base, [0.10, 0.90])
            k = 4.0 / max(q90 - q10, 1e-6)
            m = 0.5 * (q90 + q10)
            prob = 1.0 / (1.0 + np.exp(-k * (base - m)))
            layers[f"layer{li+1}"] = np.clip(prob, 0.0, 1.0)
        self.layer_probs = layers
        return layers

    def compose_risk(self, combine: str = "noisy_or") -> np.ndarray:
        assert self.layer_probs, "call build_random_layers first"
        vs = list(self.layer_probs.values())
        if combine == "noisy_or":
            r = 1.0 - np.prod([1.0 - v for v in vs], axis=0)
        else:
            r = np.maximum.reduce(vs)
        self.risk = np.clip(r, 0.0, 1.0)
        return self.risk

    # ---------- HR & coverage ----------
    def threshold_mask(self, thr: float) -> np.ndarray:
        assert self.risk is not None
        return (self.risk >= thr)

    def generate_depots_even(self, k: int) -> np.ndarray:
        # 完全整列
        xs = np.linspace(12, self.width_km-12, k)
        ys = np.linspace(7, 15, k)
        return np.stack([xs, ys], axis=1)

    def generate_depots_random(self, k: int, depot_seed: int) -> np.ndarray:
        rng = np.random.default_rng(depot_seed)
        xs = rng.uniform(2, self.width_km-2, size=k)
        ys = rng.uniform(2, self.height_km-2, size=k)
        return np.stack([xs, ys], axis=1)

    def compute_coverage_masks(self, centers_xy: np.ndarray, radius_km: float) -> np.ndarray:
        yy, xx = np.mgrid[0:self.n_rows, 0:self.n_cols]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        r2 = radius_km**2
        covered = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        for (x, y) in centers_xy:
            covered |= ((xs - x)**2 + (ys - y)**2) <= r2
        return covered

    def coverage_stats(self, hr_mask: np.ndarray, covered: np.ndarray) -> Tuple[float, float, int]:
        assert self.risk is not None
        H = hr_mask
        if H.sum() == 0:
            return 1.0, 1.0, 0
        cov = (covered & H)
        C = cov.sum() / H.sum()
        wCov = float(self.risk[cov].sum() / self.risk[H].sum())
        return C, wCov, int(H.sum() - cov.sum())

    def candidate_coords_from_non_highrisk(self, hr_mask: np.ndarray, stride: int) -> np.ndarray:
        idx = (~hr_mask)[::stride, ::stride]
        yy, xx = np.mgrid[0:self.n_rows:stride, 0:self.n_cols:stride]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        cand = np.stack([xs.ravel(), ys.ravel()], axis=1)
        return cand[idx.ravel()]

    def greedy_mincover_progress(
        self, hr_mask: np.ndarray, covered_init: np.ndarray,
        radius_km: float, stride_candidates: int,
        target_wcov: float, kmax: Optional[int],
    ) -> Tuple[List[Tuple[int, float]], np.ndarray]:
        assert self.risk is not None
        progress: List[Tuple[int, float]] = []
        cur = covered_init.copy()
        added: List[Tuple[float, float]] = []

        cands = self.candidate_coords_from_non_highrisk(hr_mask, stride_candidates)
        k = 0
        _, w0, _ = self.coverage_stats(hr_mask, cur)
        progress.append((0, w0))

        # 逐次的に最も wCov 改善が大きい候補を追加
        yy, xx = np.mgrid[0:self.n_rows, 0:self.n_cols]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        r2 = radius_km**2

        while True:
            C, wCov, _ = self.coverage_stats(hr_mask, cur)
            if wCov >= target_wcov:
                break
            if kmax is not None and k >= kmax:
                break

            best_idx = -1
            best_gain = -1.0
            best_cover = None

            for idx, (x, y) in enumerate(cands):
                test_cov = cur.copy()
                test_cov |= ((xs - x)**2 + (ys - y)**2) <= r2
                _, wtmp, _ = self.coverage_stats(hr_mask, test_cov)
                gain = wtmp - wCov
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_cover = test_cov

            if best_idx < 0:
                break
            cur = best_cover
            added.append(tuple(cands[best_idx]))
            cands = np.delete(cands, best_idx, axis=0)
            k += 1
            _, wnow, _ = self.coverage_stats(hr_mask, cur)
            progress.append((k, wnow))

        return progress, np.array(added) if len(added) > 0 else np.zeros((0, 2))

    # ---------- 追加: 多次元生成物の出力（閾値適用前） ----------
    def _export_multidim_products(self, outdir: str, multi: Dict[str, np.ndarray]) -> None:
        os.makedirs(outdir, exist_ok=True)

        # 行列としての E/V/R マップ
        np.savetxt(os.path.join(outdir, "E_map.csv"), multi["E"], delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(outdir, "V_map.csv"), multi["V"], delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(outdir, "R_map.csv"), multi["R"], delimiter=",", fmt="%.6f")

        # EV クラス（row, col, E_class, V_class）
        rows = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                rows.append((i, j, int(multi["E_cls"][i, j]), int(multi["V_cls"][i, j])))
        save_csv(os.path.join(outdir, "EV_classes.csv"), "row,col,E_class,V_class", rows)

        # ラベル版（Low/Mid/High 相当の簡易表記: Q=3想定、Q≠3でも text 付け替え可能）
        labels = np.array(["Low", "Mid", "High"])
        rows_lab = []
        QE = int(multi.get("QE", 3)); QV = int(multi.get("QV", 3))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                ecls = int(multi["E_cls"][i, j])
                vcls = int(multi["V_cls"][i, j])
                elab = labels[ecls] if QE == 3 and 0 <= ecls < 3 else str(ecls)
                vlab = labels[vcls] if QV == 3 and 0 <= vcls < 3 else str(vcls)
                rows_lab.append((i, j, elab, vlab))
        save_csv(os.path.join(outdir, "EV_classes_labeled.csv"), "row,col,E_label,V_label", rows_lab)

        # HighE∩HighV （既定）セル index（行優先のフラットID）
        H_idx = np.flatnonzero(multi["H_cat_mask"].ravel())
        save_csv(os.path.join(outdir, "H_cat_idx.csv"), "idx", [(int(x),) for x in H_idx])

        # PNG 可視化
        try:
            for key in ["E", "V", "R"]:
                fig, ax = plt.subplots(figsize=(9.5, 6))
                im = common_imshow(ax, multi[key], self.extent)
                ax.set_title(key)
                ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
                add_colorbar_outside(ax, im, f"{key} (0–1)")
                fig.subplots_adjust(right=0.80)
                fig.savefig(os.path.join(outdir, f"{key}.png"), dpi=180)
                plt.close(fig)

            # EV クラスまとめ（QE×QV）
            cls_id = (multi["E_cls"] * QV + multi["V_cls"]).astype(int)
            fig, ax = plt.subplots(figsize=(9.5, 6))
            im = ax.imshow(cls_id, extent=self.extent, origin="lower", interpolation="nearest")
            ax.set_title(f"EV classes ({QE}x{QV})")
            ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
            fig.subplots_adjust(right=0.80)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("class id")
            fig.savefig(os.path.join(outdir, f"EV_classes_{QE}x{QV}.png"), dpi=180)
            plt.close(fig)
        except Exception as e:
            print("[WARN] Failed to export E/V/R PNGs:", e)

    # ---------- 可視化（カテゴリ対応） ----------
    def plot_hr_mask(self, out_png: str, hr_label: str, hr_mask: np.ndarray):
        assert self.risk is not None
        fig, ax = plt.subplots(figsize=(9.5, 6))
        im = common_imshow(ax, self.risk, self.extent)
        mask_rgba = np.zeros((*hr_mask.shape, 4))
        mask_rgba[hr_mask] = COLORS["hr_mask"]
        ax.imshow(mask_rgba, extent=self.extent, origin="lower", interpolation="nearest")
        ax.set_title(f"High-Risk Mask ({hr_label})")
        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        add_colorbar_outside(ax, im, "Risk (0–1)")
        fig.subplots_adjust(right=0.80)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

    def plot_mclp(self, out_png_risk: str, out_png_hr: str, hr_label: str,
                  depots: np.ndarray, radius_km: float, hr_mask: np.ndarray):
        from Utils_v1_1 import header_legend
        n_ex = 0 if depots is None else len(depots)

        # ---- risk 背景 ----
        fig, ax = plt.subplots(figsize=(9.8, 6))
        im = common_imshow(ax, self.risk, self.extent)
        mask_rgba = np.zeros((*hr_mask.shape, 4))
        mask_rgba[hr_mask] = COLORS["hr_mask"]
        ax.imshow(mask_rgba, extent=self.extent, origin="lower", interpolation="nearest")
        draw_circles(ax, depots, radius_km, COLORS["existing_edge"], ls="-", lw=2.0)
        draw_points(ax, depots, COLORS["existing_face"], marker="o", s=28)
        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        add_colorbar_outside(ax, im, "Risk (0–1)")
        fig.subplots_adjust(top=0.92, right=0.88)
        header_legend(fig, n_exist=n_ex, n_added=None, y=0.965)
        fig.savefig(out_png_risk, dpi=180); plt.close(fig)

        # ---- HR-only ----
        fig2, ax2 = plt.subplots(figsize=(9.8, 6))
        mask_rgba = np.zeros((*hr_mask.shape, 4))
        mask_rgba[hr_mask] = COLORS["hr_mask"]
        ax2.imshow(mask_rgba, extent=self.extent, origin="lower", interpolation="nearest")
        draw_circles(ax2, depots, radius_km, COLORS["existing_edge"], ls="-", lw=2.0)
        draw_points(ax2, depots, COLORS["existing_face"], marker="o", s=28)
        ax2.set_xlabel("X (km)"); ax2.set_ylabel("Y (km)")
        fig2.subplots_adjust(top=0.92, right=0.88)
        header_legend(fig2, n_exist=n_ex, n_added=None, y=0.965)
        fig2.savefig(out_png_hr, dpi=180); plt.close(fig2)

    def plot_sclp_overlay(
        self, out_png_risk: str, out_png_hr: str, hr_label: str,
        depots_exist: np.ndarray, depots_add: np.ndarray,
        radius_km: float, hr_mask: np.ndarray, newly_cov_mask: np.ndarray
    ):
        from Utils_v1_1 import header_legend
        n_ex = 0 if depots_exist is None else len(depots_exist)
        n_ad = 0 if depots_add is None else len(depots_add)

        # ---------- risk 背景 ----------
        fig, ax = plt.subplots(figsize=(9.8, 6))
        im = common_imshow(ax, self.risk, self.extent)

        mask_rgba = np.zeros((*hr_mask.shape, 4))
        mask_rgba[hr_mask] = COLORS["hr_mask"]
        ax.imshow(mask_rgba, extent=self.extent, origin="lower", interpolation="nearest")
        new_rgba = np.zeros((*newly_cov_mask.shape, 4))
        new_rgba[newly_cov_mask] = COLORS["newly_cov"]
        ax.imshow(new_rgba, extent=self.extent, origin="lower", interpolation="nearest")

        draw_circles(ax, depots_exist, radius_km, COLORS["existing_edge"], ls="-", lw=2.0)
        draw_points(ax, depots_exist, COLORS["existing_face"], marker="o", s=28)
        if n_ad > 0:
            draw_circles(ax, depots_add, radius_km, COLORS["added_edge"], ls="--", lw=2.0)
            draw_points(ax, depots_add, COLORS["added_face"], marker="^", s=36)

        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        add_colorbar_outside(ax, im, "Risk (0–1)")
        fig.subplots_adjust(top=0.92, right=0.88)
        header_legend(fig, n_exist=n_ex, n_added=n_ad, y=0.965)
        fig.savefig(out_png_risk, dpi=180); plt.close(fig)

        # ---------- HR-only ----------
        fig2, ax2 = plt.subplots(figsize=(9.8, 6))
        mask_rgba = np.zeros((*hr_mask.shape, 4))
        mask_rgba[hr_mask] = COLORS["hr_mask"]
        ax2.imshow(mask_rgba, extent=self.extent, origin="lower", interpolation="nearest")
        new_rgba = np.zeros((*newly_cov_mask.shape, 4))
        new_rgba[newly_cov_mask] = COLORS["newly_cov"]
        ax2.imshow(new_rgba, extent=self.extent, origin="lower", interpolation="nearest")

        draw_circles(ax2, depots_exist, radius_km, COLORS["existing_edge"], ls="-", lw=2.0)
        draw_points(ax2, depots_exist, COLORS["existing_face"], marker="o", s=28)
        if n_ad > 0:
            draw_circles(ax2, depots_add, radius_km, COLORS["added_edge"], ls="--", lw=2.0)
            draw_points(ax2, depots_add, COLORS["added_face"], marker="^", s=36)

        ax2.set_xlabel("X (km)"); ax2.set_ylabel("Y (km)")
        fig2.subplots_adjust(top=0.92, right=0.88)
        header_legend(fig2, n_exist=n_ex, n_added=n_ad, y=0.965)
        fig2.savefig(out_png_hr, dpi=180); plt.close(fig2)

    # ---------- オーケストレーター ----------
    def run_once(self, p: RunParams) -> RunResult:
        """
        1 ケース実行して図/CSV/メタデータを出力し、要約値を返す。
        - 既存デポ: p.scenario に応じて整列 or ランダム
        - MCLP: 既存デポでカバー率算出
        - SCLP: 未達分を貪欲法で追加し、到達進捗と最終カバー率算出
        - v3: H 抽出は hr_mode により threshold か category を選択
        """
        import secrets
        base_seed = p.master_seed if p.master_seed is not None else 123456
        seeds = derive_subseeds(base_seed, 3)
        if getattr(p, "randomize_depots", False):
            seeds["depot"] = secrets.randbits(31)

        # ---- 多次元リスク生成（QE/QV 対応）----
        multi = self.generate_multidim_risk(
            seeds["layer"],
            QE=p.QE, QV=p.QV
        )
        self.risk = multi["R"]  # 以降は self.risk を参照

        # ---- HR マスク（カテゴリ or 閾値）----
        if p.hr_mode == "category":
            E_levels = p.E_levels if p.E_levels is not None else [p.QE - 1]
            V_levels = p.V_levels if p.V_levels is not None else [p.QV - 1]
            hr_mask = self._make_category_mask(multi["E_cls"], multi["V_cls"], E_levels, V_levels)
            hr_label = self._cat_label(p.QE, p.QV, E_levels, V_levels)   # タイトル用
            hr_suffix = "cat-" + self._slugify_str(hr_label)             # ファイル名用
        else:
            hr_mask = self.threshold_mask(p.thr)
            hr_label = f"≥ {p.thr:.2f}"
            hr_suffix = f"thr-{slugify_float(p.thr)}"

        # ---- 既存デポ（MCLP）----
        if p.scenario == "even_k":
            depots_exist = self.generate_depots_even(p.k)
        else:
            depots_exist = self.generate_depots_random(p.k, seeds["depot"])

        covered_mclp = self.compute_coverage_masks(depots_exist, p.radius_km)
        C_max, wCov_max, _ = self.coverage_stats(hr_mask, covered_mclp)

        # ---- 追加デポ（SCLP; 貪欲）----
        progress, depots_add = self.greedy_mincover_progress(
            hr_mask, covered_mclp, p.radius_km, p.stride_candidates, p.target_wcov, p.kmax
        )
        covered_final = covered_mclp.copy()
        if len(depots_add) > 0:
            covered_final |= self.compute_coverage_masks(depots_add, p.radius_km)

        C_final, wCov_final, uncovered = self.coverage_stats(hr_mask, covered_final)
        cov_final_simple, _, _ = self.coverage_stats(hr_mask, covered_final)  # 単純被覆率
        newly = (covered_final & hr_mask) & (~covered_mclp)                   # 新規被覆

        # ---- 出力ディレクトリ作成 ----
        outdir = build_outdir(
            p.out_base,
            {
                "thr": (p.thr if p.hr_mode == "threshold" else -1),  # 互換のため残置
                "radius_km": p.radius_km, "k": p.k, "scenario": p.scenario,
                "hrmode": p.hr_mode, "QE": p.QE, "QV": p.QV,
                "Elevels": (p.E_levels if p.E_levels is not None else [p.QE-1]),
                "Vlevels": (p.V_levels if p.V_levels is not None else [p.QV-1]),
            },
            seeds
        )

        # ---- 生成物（E/V/R, EVクラス, PNG）を保存 ----
        self._export_multidim_products(outdir, multi)

        # ---- HRセル一覧（カテゴリ/閾値 でファイル名切替）----
        yy, xx = np.mgrid[0:self.n_rows, 0:self.n_cols]
        xs = (xx + 0.5) * self.cell_km
        ys = (yy + 0.5) * self.cell_km
        hr_rows = [
            (int(i), int(j), float(xs[i, j]), float(ys[i, j]), float(self.risk[i, j]))
            for i, j in zip(*np.where(hr_mask))
        ]
        save_csv(
            f"{outdir}/hr_cells_{hr_suffix}.csv",
            "row,col,x_km,y_km,risk", hr_rows
        )

        # ---- 既存/追加デポ, 進捗 ----
        save_csv(
            f"{outdir}/depots_existing.csv",
            "x_km,y_km",
            [(float(x), float(y)) for x, y in depots_exist]
        )
        if len(depots_add) > 0:
            save_csv(
                f"{outdir}/depots_added.csv",
                "x_km,y_km",
                [(float(x), float(y)) for x, y in depots_add]
            )

        save_csv(f"{outdir}/progress_wcov_vs_k.csv", "k,wCov", progress)

        # ---- 指標計算（coverage_summary.csv: 互換列 + 新列）----
        RM = float(self.risk.sum())
        HRA = float(hr_mask.sum() * (self.cell_km ** 2))
        HRI = float(self.risk[hr_mask].mean() if hr_mask.any() else 0.0)
        k_added = int(progress[-1][0])
        k_at95 = int(next((kk for kk, ww in progress if ww >= 0.95), k_added))
        AUC = float(sum(w for _, w in progress))
        S = p.w1 * C_final - p.w2 * (k_added / max(p.kcap, 1)) - p.w3 * (1.0 - C_max)

        # カテゴリ情報の文字列化（CSV 書式のため）
        E_levels_out = (p.E_levels if p.E_levels is not None else [p.QE - 1])
        V_levels_out = (p.V_levels if p.V_levels is not None else [p.QV - 1])
        E_levels_str = "[" + ",".join(str(x) for x in E_levels_out) + "]"
        V_levels_str = "[" + ",".join(str(x) for x in V_levels_out) + "]"
        hr_label = (self._cat_label(p.QE, p.QV, E_levels_out, V_levels_out)
                    if p.hr_mode == "category" else f">= {p.thr:.2f}")

        save_csv(
            f"{outdir}/coverage_summary.csv",
            "RM,HRA,HRI,C_max,OneMinusCmax,C_final,Coverage_final,k_star,k_at95,AUC,S,thr,R,k,scenario,run_folder,HR_mode,QE,QV,E_levels,V_levels,HR_label",
            [(
                RM, HRA, HRI,
                float(C_max), float(1.0 - C_max),
                float(C_final), float(cov_final_simple),
                k_added, k_at95, AUC, S,
                (p.thr if p.hr_mode == "threshold" else -1.0),
                p.radius_km, p.k, p.scenario, outdir,
                p.hr_mode, p.QE, p.QV, E_levels_str, V_levels_str, hr_label
            )]
        )

        # ---- 図 ----
        self.plot_hr_mask(
            f"{outdir}/01_hr_mask_{hr_suffix}.png",
            hr_label, hr_mask
        )
        self.plot_mclp(
            f"{outdir}/02_mclp_risk_{hr_suffix}_R-{slugify_float(p.radius_km)}_k-{p.k}_sce-{p.scenario}.png",
            f"{outdir}/02b_mclp_hronly_{hr_suffix}_R-{slugify_float(p.radius_km)}_k-{p.k}_sce-{p.scenario}.png",
            hr_label, depots_exist, p.radius_km, hr_mask
        )
        self.plot_sclp_overlay(
            f"{outdir}/03_sclp_overlay_risk_{hr_suffix}_R-{slugify_float(p.radius_km)}_k-{p.k}_sce-{p.scenario}_twcov-{slugify_float(p.target_wcov)}.png",
            f"{outdir}/03b_sclp_overlay_hronly_{hr_suffix}_R-{slugify_float(p.radius_km)}_k-{p.k}_sce-{p.scenario}_twcov-{slugify_float(p.target_wcov)}.png",
            hr_label, depots_exist, depots_add, p.radius_km, hr_mask, newly
        )

        # ---- メタデータ保存（分位境界・HR情報） ----
        extra_meta = {
            "grid": {"width_km": self.width_km, "height_km": self.height_km, "cell_km": self.cell_km},
            "multi_quantiles": {"E_edges": multi.get("E_edges", []), "V_edges": multi.get("V_edges", [])},
            "H_cat_size": int(np.count_nonzero(hr_mask)),
            "H_thr_size": int(np.count_nonzero(self.threshold_mask(p.thr))) if p.hr_mode == "threshold" else None,
            "HR_label": hr_label,
        }
        save_run_metadata(
            outdir,
            {
                "thr": p.thr, "radius_km": p.radius_km, "k": p.k, "scenario": p.scenario,
                "target_wcov": p.target_wcov, "kmax": p.kmax, "stride_candidates": p.stride_candidates,
                "w1": p.w1, "w2": p.w2, "w3": p.w3, "kcap": p.kcap,
                "randomize_depots": getattr(p, "randomize_depots", False),
                "hr_mode": p.hr_mode, "QE": p.QE, "QV": p.QV,
                "E_levels": E_levels_out, "V_levels": V_levels_out,
            },
            seeds,
            extra_meta
        )

        # ---- 結果返却 ----
        return RunResult(
            C_max=float(C_max), wCov_max=float(wCov_max),
            C_final=float(C_final), wCov_final=float(wCov_final),
            Coverage_final=float(cov_final_simple),
            k_added=k_added, k_at95=k_at95, AUC=AUC, Gap=float(1.0 - C_max),
            S=float(S),
            uncovered_count=int(np.count_nonzero(hr_mask & (~covered_final))),
            outdir=outdir
        )
