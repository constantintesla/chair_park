#!/usr/bin/env python3
# movement_analysis.py
"""
Универсальный анализ циклов «сидя-стоя-сидя» по YOLO-pose CSV.
Полностью соответствует протоколу 6.5.1.7 «Вставание с кресла (0-4)».
"""

from __future__ import annotations
import argparse
import json
import math
import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------ конфиг 6.5.1.7 ------------------------
MIN_HIP_RISE_PX          = 15          # минимальный подъём таза (px)
MIN_CYCLE_DURATION_S     = 1.0         # ≥ 1 с
PEAK_DISTANCE_S          = 0.4
PROMINENCE               = 2
WRIST_HEIGHT_RATIO       = 0.35
TRUNK_ANGLE_THRESHOLD    = 50
# новые пороги по протоколу
MAX_NORMAL_UP_S     = 2.0   # норма ≤ 2 с
MAX_ACCEPT_UP_S     = 5.0   # лёгкое нарушение ≤ 5 с
# ---------------------------------------------------------------


def load_keypoints(csv_path: str) -> pd.DataFrame:
    """Загружает CSV и добавляет производные столбцы."""
    df = pd.read_csv(csv_path)
    df["hip_y"]   = (df.kp_11_y + df.kp_12_y) / 2
    df["hip_x"]   = (df.kp_11_x + df.kp_12_x) / 2
    df["sh_y"]    = (df.kp_5_y  + df.kp_6_y)  / 2
    df["sh_x"]    = (df.kp_5_x  + df.kp_6_x)  / 2
    df["wrist_y"] = (df.kp_9_y  + df.kp_10_y) / 2
    return df


def _find_peaks_valleys(y: np.ndarray, fps: float):
    distance = int(fps * PEAK_DISTANCE_S)
    min_ids, _ = find_peaks(-y, distance=distance, prominence=PROMINENCE)
    max_ids, _ = find_peaks(y,  distance=distance, prominence=PROMINENCE)
    return min_ids, max_ids


def _hip_knee_ankle_angle(df: pd.DataFrame, idx: int, side: str = "left") -> float:
    h, k, a = (11, 13, 15) if side == "left" else (12, 14, 16)
    v1 = np.array([df[f"kp_{h}_x"].iloc[idx] - df[f"kp_{k}_x"].iloc[idx],
                   df[f"kp_{h}_y"].iloc[idx] - df[f"kp_{k}_y"].iloc[idx]])
    v2 = np.array([df[f"kp_{a}_x"].iloc[idx] - df[f"kp_{k}_x"].iloc[idx],
                   df[f"kp_{a}_y"].iloc[idx] - df[f"kp_{k}_y"].iloc[idx]])
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cosang, -1, 1)))


def _trunk_angle(df: pd.DataFrame, idx: int) -> float:
    dx = df.sh_x.iloc[idx] - df.hip_x.iloc[idx]
    dy = df.hip_y.iloc[idx] - df.sh_y.iloc[idx]
    return math.degrees(math.atan2(dx, dy))


# ---------- универсальный поиск циклов ----------
def _find_cycles(df: pd.DataFrame, fps: float):
    window = max(5, int(fps * 0.20))
    y = df.hip_y.rolling(window, min_periods=1, center=False).mean().values
    mins, maxs = _find_peaks_valleys(y, fps)

    cycles: List[tuple[int, int, int]] = []

    # если видео начинается «сидя» – разрешаем старт с кадра 0
    starts_sitting = bool(len(mins)) and (mins[0] < 10)  # первый мин в первых 0.4 с

    for stand in maxs:
        right_mins = mins[mins > stand]
        if len(right_mins) == 0:
            continue
        left_mins = mins[mins < stand]
        # берём ближайший мин СЛЕВА; если его нет и мы «сидим» – стартуем с 0
        if len(left_mins):
            sit1 = left_mins[-1]
        elif starts_sitting:
            sit1 = 0
        else:
            continue  # нет полного цикла

        sit2 = right_mins[0]
        if (df.time.iloc[sit2] - df.time.iloc[sit1]) < MIN_CYCLE_DURATION_S:
            continue
        if y[stand] - y[sit1] > MIN_HIP_RISE_PX:
            cycles.append((sit1, stand, sit2))

    return cycles


# ---------- метрики по 6.5.1.7 ----------
def _cycle_metrics(df: pd.DataFrame, sit1: int, stand: int, sit2: int, fps: float) -> Dict:
    t1, t2, t3 = df.time.iloc[[sit1, stand, sit2]]
    t_up   = t2 - t1
    t_down = t3 - t2

    # 1. неудачные попытки
    fails = 0
    if t_up > MAX_ACCEPT_UP_S:
        fails = 1
    if t_up > 7.0 or (df.time.iloc[sit2] - t1) > 10.0:
        fails = 2

    # 2. использование рук
    wrist_low = (df.wrist_y.iloc[sit1:stand] <
                 df.hip_y.iloc[sit1:stand] * WRIST_HEIGHT_RATIO).mean()
    uses_hands = wrist_low > 0.25

    # 3. скрещивание рук на груди
    l_elb_in = (df.kp_9_x.iloc[sit1:stand].between(
                df.kp_5_x.iloc[sit1:stand], df.kp_6_x.iloc[sit1:stand])).mean()
    r_elb_in = (df.kp_10_x.iloc[sit1:stand].between(
                df.kp_5_x.iloc[sit1:stand], df.kp_6_x.iloc[sit1:stand])).mean()
    arms_crossed = (l_elb_in > 0.7) and (r_elb_in > 0.7)

    angle = _trunk_angle(df, stand)

    # 4. категория 0-4 по 6.5.1.7
    if t_up <= MAX_NORMAL_UP_S and fails == 0 and not uses_hands and arms_crossed:
        cat = 0
    elif fails <= 1 and t_up <= MAX_ACCEPT_UP_S and not uses_hands:
        cat = 1
    elif fails <= 2 and t_up <= 7.0:
        cat = 2
    elif uses_hands or fails > 2:
        cat = 3
    else:
        cat = 4

    return {
        "sit_start_sec":   float(t1),
        "stand_sec":       float(t2),
        "sit_end_sec":     float(t3),
        "t_up_sec":        float(t_up),
        "t_down_sec":      float(t_down),
        "failed_attempts": int(fails),
        "uses_hands":      bool(uses_hands),
        "arms_crossed":    bool(arms_crossed),
        "trunk_angle_deg": float(angle),
        "hip_delta_px":    float(df.hip_y.iloc[stand] - df.hip_y.iloc[sit1]),
        "category":        cat,
        "category_desc":   ["норма", "лёгкое", "умеренное", "выраженное", "максимальная декомпенсация"][cat]
    }


def _summary(cycles: List[Dict]) -> Dict:
    if not cycles:
        return {"cycles_found": 0}
    cats = [c["category"] for c in cycles]
    return {
        "cycles_found":          len(cycles),
        "mean_t_up_sec":         float(np.mean([c["t_up_sec"] for c in cycles])),
        "median_category":       int(np.median(cats)),
        "worst_category":        int(np.max(cats)),
        "uses_hands_any":        any(c["uses_hands"] for c in cycles),
        "total_failed_attempts": int(sum(c["failed_attempts"] for c in cycles))
    }



def analyse(csv_path: str,
            min_cycle_duration: float = MIN_CYCLE_DURATION_S,
            plot: bool = False) -> Dict:
    df = load_keypoints(csv_path)
    fps = 1 / np.diff(df.time).mean()
    raw = _find_cycles(df, fps)
    cycles = [_cycle_metrics(df, s1, st, s2, fps) for s1, st, s2 in raw
              if (df.time.iloc[s2] - df.time.iloc[s1]) >= min_cycle_duration]
    summary = _summary(cycles)
    report = {"summary": summary, "cycles": cycles}

    if plot:
        plt.figure(figsize=(12, 4))
        plt.title("Высота центра таза (циклы)")
        plt.plot(df.time, df.hip_y, color="silver", label="hip Y")
        for c in cycles:
            color = ["green", "yellow", "orange", "red", "darkred"][c["category"]]
            plt.axvspan(c["sit_start_sec"], c["sit_end_sec"], alpha=0.25, color=color)
        plt.xlabel("время, с")
        plt.ylabel("Y, px")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return report



def main():
    parser = argparse.ArgumentParser(description="Анализ циклов «вставание из кресла» по 6.5.1.7")
    parser.add_argument("--input", required=True, help="CSV-файл с ключевыми точками")
    parser.add_argument("--output", help="JSON-отчёт (по умолчанию: <input>_movement.json)")
    parser.add_argument("--min-cycle-duration", type=float, default=MIN_CYCLE_DURATION_S)
    parser.add_argument("--plot", action="store_true", help="Показать график")
    args = parser.parse_args()

    out_path = args.output or (pathlib.Path(args.input).stem + "_movement.json")
    report = analyse(args.input, args.min_cycle_duration, args.plot)
    pathlib.Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Отчёт сохранён → {out_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()