#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ функциональной пробы «Вставание с кресла» (5 подъёмов)
на основе CSV-файла с трекингом ключевых точек тела.

Три состояния рук:
  - pushed_off_chair  – опирался кистью на сидушку ≥ 0.3 с
  - hands_on_chest    – держал руки на груди всё время
  - no_arm_contact    – ни опоры, ни «на груди»
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------  Утилиты  ----------
def normalize_keypoint_name(c: str) -> str:
    return c.replace("kp_", "")


def hip_height(keypoints: pd.DataFrame, t: int) -> float:
    return (keypoints.loc[t, "11_y"] + keypoints.loc[t, "12_y"]) / 2.0


def shoulder_speed(keypoints: pd.DataFrame, t: int, fps: float) -> float:
    if t == 0:
        return 0.0
    dt = 1.0 / fps
    dx = keypoints.loc[t, "5_x"] - keypoints.loc[t - 1, "5_x"]
    dy = keypoints.loc[t, "5_y"] - keypoints.loc[t - 1, "5_y"]
    return np.hypot(dx, dy) / dt


# ----------  Детектор циклов  ----------
def detect_cycles(keypoints: pd.DataFrame, fps: float, min_duration: float):
    hip = np.array([hip_height(keypoints, t) for t in range(len(keypoints))])
    kernel = np.ones(5) / 5
    hip_smooth = np.convolve(hip, kernel, mode="same")
    grad = np.gradient(hip_smooth)
    sit_threshold = np.percentile(hip_smooth, 10)

    cycles, n = [], len(grad)
    i = 0
    while i < n - 1:
        if grad[i] < 0.5:
            i += 1
            continue
        start = i
        while i < n - 1 and grad[i] > -0.5:
            i += 1
        peak = i
        while i < n - 1 and hip_smooth[i] > sit_threshold:
            i += 1
        end = i
        duration = (end - start) / fps
        if duration >= min_duration:
            cycles.append((start, end, duration))
        i += 1
    return cycles


# ----------  Попытки (первые 2 с)  ----------
def detect_attempts(df: pd.DataFrame, fps: float) -> int:
    speed = [shoulder_speed(df, t, fps) for t in range(len(df))]
    time = df["time"].values
    mask = time <= 2.0
    speed = np.array(speed) * mask
    thd, attempts, last = 50.0, 0, -np.inf
    for t, v in enumerate(speed):
        if v > thd and (time[t] - last) > 0.3:
            attempts += 1
            last = time[t]
    return max(attempts, 1)


# ----------  Три состояния рук  ----------
def detect_arm_status(df: pd.DataFrame) -> dict:
    # 1. Опора на сидушку (кисть 9 внутри прямоугольника ≥ 0.3 с)
    seat_top = np.percentile(df["11_y"], 5)
    seat_bot = np.percentile(df["11_y"], 25)
    seat_l   = np.percentile(df["11_x"], 10)
    seat_r   = np.percentile(df["11_x"], 90)

    inside = (seat_l < df["9_x"]) & (df["9_x"] < seat_r) & \
             (seat_top < df["9_y"]) & (df["9_y"] < seat_bot)
    fps = len(df) / df["time"].iloc[-1]
    min_frames = int(0.3 * fps)
    pushed_off = (inside.rolling(window=min_frames, center=False).min() == 1).any()

    # 2. Руки на груди (допускаем 5 % «выбросов»)
    shoulder_y = (df["5_y"] + df["6_y"]) / 2
    hip_y      = (df["11_y"] + df["12_y"]) / 2
    body_size  = np.abs(shoulder_y - hip_y).mean()
    chest_band = 0.15 * body_size
    on_chest = (np.abs(df["9_y"]  - shoulder_y) < chest_band) & \
               (np.abs(df["10_y"] - shoulder_y) < chest_band)
    # разрешаем 5 % кадров вне зоны
    hands_on_chest = on_chest.mean() >= 0.95

    # 3. Итог
    if pushed_off:
        return {"pushed_off_chair": True, "hands_on_chest": False, "no_arm_contact": False}
    if hands_on_chest:
        return {"pushed_off_chair": False, "hands_on_chest": True, "no_arm_contact": False}
    return {"pushed_off_chair": False, "hands_on_chest": False, "no_arm_contact": True}


# ----------  Аномалии  ----------
def detect_needed_assistance(keypoints: pd.DataFrame) -> bool:
    hip = np.array([hip_height(keypoints, t) for t in range(len(keypoints))])
    grad = np.gradient(hip)
    z = np.abs((grad - np.mean(grad)) / (np.std(grad) + 1e-8))
    return np.any(z > 3.0)


def calculate_score(completion_time: float, attempts: int, arm_status: dict, needed_assistance: bool) -> int:
    """0 = отлично, 4 = максимальная декомпенсация"""
    if needed_assistance:
        return 4
    if arm_status["pushed_off_chair"]:
        return 3
    if completion_time > 15.0:
        return 2
    if completion_time < 10.0 and arm_status["hands_on_chest"]:
        return 0
    return 1


# ----------  Главная функция анализа  ----------
def analyse(input_csv: str, min_cycle_duration: float, plot: bool):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV: {e}")

    required = ["frame", "time"] + [f"kp_{i}_{ax}" for i in range(17) for ax in ("x", "y")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.rename(columns={c: normalize_keypoint_name(c) for c in df.columns if c.startswith("kp_")})
    fps = len(df) / df["time"].iloc[-1]
    if fps < 30:
        print("Warning: sampling rate < 30 Hz, results may be inaccurate")

    cycles = detect_cycles(df, fps, min_cycle_duration)
    total_cycles = len(cycles)
    if total_cycles == 0:
        return {
            "summary": {
                "score": 0,
                "description": "Циклы не обнаружены",
                "completion_time": None,
                "cycles_completed": 0,
                "attempts": 1,
                "arm_status": {"pushed_off_chair": False, "hands_on_chest": False, "no_arm_contact": True},
                "needed_assistance": True,
            },
            "details": {"cycle_durations": [], "total_cycles_detected": 0},
        }

    relevant = cycles[:5]
    durations = [d for _, _, d in relevant]
    completion_time = sum(durations)

    attempts = detect_attempts(df, fps)
    arm_status = detect_arm_status(df)
    needed_assistance = detect_needed_assistance(df)

    score = calculate_score(completion_time, attempts, arm_status, needed_assistance)
    descr_map = {
        4: "Требовалась помощь",
        3: "Опирался руками на сидушку",
        2: "Медленное выполнение (>15 с)",
        1: "Удовлетворительно (10–15 с или руки не на груди)",
        0: "Отлично (<10 с, руки на груди)",
    }
    if plot:
        plt.figure(figsize=(10, 4))
        hip = [hip_height(df, t) for t in range(len(df))]
        plt.plot(df["time"], hip, label="Hip height (smooth)")
        for start, end, _ in relevant:
            plt.axvspan(df["time"].iloc[start], df["time"].iloc[end], alpha=0.2, color="green")
        plt.title("Detected cycles (hip height)")
        plt.xlabel("Time, s")
        plt.ylabel("Y, px")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "summary": {
            "score": score,
            "description": descr_map[score],
            "completion_time": round(completion_time, 2),
            "cycles_completed": len(relevant),
            "attempts": attempts,
            "arm_status": arm_status,
            "needed_assistance": bool(needed_assistance),
        },
        "details": {
            "cycle_durations": [round(d, 2) for d in durations],
            "total_cycles_detected": total_cycles,
        },
    }


# ----------  CLI  ----------
def main():
    parser = argparse.ArgumentParser(description="Анализ функциональной пробы «Вставание с кресла» (5 подъёмов)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV-файл с трекингом ключевых точек")
    parser.add_argument("--output", help="JSON-отчёт (по умолчанию: <input>_movement.json)")
    parser.add_argument("--min-cycle-duration", type=float, default=1.0, help="Минимальная длительность цикла, с")
    parser.add_argument("--plot", action="store_true", help="Показать графики")
    args = parser.parse_args()

    if not Path(args.input).is_file():
        print(f"Error: file not found {args.input}")
        sys.exit(1)

    out_path = args.output or str(Path(args.input).with_suffix("")) + "_movement.json"
    report = analyse(args.input, args.min_cycle_duration, args.plot)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {out_path}")
    print("Summary:")
    for k, v in report["summary"].items():
        if k == "arm_status":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()