# movement_analysis.py
"""
Универсальный анализ циклов «сидя-стоя-сидя» по YOLO-pose CSV.
соответствует протоколу 6.5.1.7 «Вставание с кресла (0-4)».
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
MIN_HIP_RISE_PX          = 10           # минимальный подъём таза (px) ← СНИЖЕНО
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

    starts_sitting = bool(len(mins)) and (mins[0] < 10)

    for stand in maxs:
        right_mins = mins[mins > stand]
        if len(right_mins) == 0:
            continue
        left_mins = mins[mins < stand]
        if len(left_mins):
            sit1 = left_mins[-1]
        elif starts_sitting:
            sit1 = 0
        else:
            continue

        sit2 = right_mins[0]
        if (df.time.iloc[sit2] - df.time.iloc[sit1]) < MIN_CYCLE_DURATION_S:
            continue
        if y[stand] - y[sit1] > MIN_HIP_RISE_PX:
            cycles.append((sit1, stand, sit2))
        # else: пропущен – можно вывести отладку при необходимости

    return cycles


# ---------- метрики по 6.5.1.7 ----------
def _cycle_metrics(df: pd.DataFrame, sit1: int, stand: int, sit2: int, fps: float) -> Dict:
    """
    Вычисление метрик для одного цикла вставания-садания.
    
    Args:
        df: DataFrame с ключевыми точками
        sit1: индекс начала вставания (сидя)
        stand: индекс полного вставания 
        sit2: индекс окончания садания (сидя)
        fps: кадров в секунду
    
    Returns:
        Словарь с метриками цикла
    """
    
    # Валидация индексов
    if not (0 <= sit1 < stand < sit2 < len(df)):
        raise ValueError("Некорректные индексы фаз цикла")
    
    t1, t2, t3 = df.time.iloc[[sit1, stand, sit2]]
    t_up = t2 - t1
    t_down = t3 - t2

    # 1. Константы для классификации
    MAX_NORMAL_UP_S = 3.0    # Макс время для нормальной категории
    MAX_ACCEPT_UP_S = 5.0    # Макс время для приемлемой категории  
    MAX_CYCLE_S = 10.0       # Макс время всего цикла
    WRIST_HEIGHT_RATIO = 0.9 # Порог для определения использования рук
    ELBOW_CROSS_THRESH = 0.7 # Порог для скрещивания рук

    # 1. Анализ неудачных попыток (улучшенная логика)
    fails = 0
    cycle_duration = t3 - t1
    
    # Неудачная попытка 1: слишком медленное вставание
    if t_up > MAX_ACCEPT_UP_S:
        fails = 1
    
    # Неудачная попытка 2: критически медленное вставание или слишком долгий цикл
    if t_up > 7.0 or cycle_duration > MAX_CYCLE_S:
        fails = 2
    
    # Дополнительная проверка: наличие промежуточных остановок
    if fails > 0:
        # Анализ скорости вставания (производная по высоте бедра)
        hip_heights = df.hip_y.iloc[sit1:stand].values
        if len(hip_heights) > 10:
            velocities = np.diff(hip_heights) * fps
            # Если есть значительные падения скорости (остановки)
            slow_periods = np.sum(velocities < np.max(velocities) * 0.3)
            if slow_periods > len(velocities) * 0.4:  # >40% времени медленно
                fails = max(fails, 1)

    # 2. Использование рук (улучшенный детектор)
    wrist_low_ratio = (df.wrist_y.iloc[sit1:stand] < 
                      df.hip_y.iloc[sit1:stand] * WRIST_HEIGHT_RATIO).mean()
    uses_hands = wrist_low_ratio > 0.25

    # Дополнительная проверка: движение рук вверх
    if not uses_hands:
        # Если запястья поднимаются вместе с телом - это может быть компенсаторное движение
        wrist_start = df.wrist_y.iloc[sit1]
        wrist_peak = df.wrist_y.iloc[sit1:stand].min()
        if wrist_peak < wrist_start * 0.95:  # Запястья поднялись на 5% относительно старта
            uses_hands = True

    # 3. Скрещивание рук на груди (более надежная проверка)
    l_elb_in_ratio = (df.kp_9_x.iloc[sit1:stand].between(
        df.kp_5_x.iloc[sit1:stand].min(),  # Левый плечо (min/max для учета движения)
        df.kp_6_x.iloc[sit1:stand].max()
    )).mean()
    
    r_elb_in_ratio = (df.kp_10_x.iloc[sit1:stand].between(
        df.kp_5_x.iloc[sit1:stand].min(),  # Правый плечо
        df.kp_6_x.iloc[sit1:stand].max()
    )).mean()
    
    arms_crossed = (l_elb_in_ratio > ELBOW_CROSS_THRESH) and (r_elb_in_ratio > ELBOW_CROSS_THRESH)

    # 4. Угол наклона туловища (улучшенный расчет)
    angle = _trunk_angle(df, stand)
    
    # 5. Дополнительные метрики
    hip_delta = df.hip_y.iloc[stand] - df.hip_y.iloc[sit1]
    
    # Стабильность в вертикальном положении
    stand_stability = 0.0
    if stand + int(fps) < len(df):  # Анализ 1 секунды после вставания
        stand_heights = df.hip_y.iloc[stand:stand + int(fps)]
        stand_stability = np.std(stand_heights) / hip_delta if hip_delta > 0 else 0

    # 6. Категория по 6.5.1.7 (уточненная логика)
    if (t_up <= MAX_NORMAL_UP_S and fails == 0 and 
        not uses_hands and arms_crossed and stand_stability < 0.1):
        cat = 0  # норма
    elif (fails <= 1 and t_up <= MAX_ACCEPT_UP_S and 
          not uses_hands and stand_stability < 0.15):
        cat = 1  # легкое
    elif (fails <= 2 and t_up <= 7.0 and 
          stand_stability < 0.2):
        cat = 2  # умеренное
    elif uses_hands or fails > 2 or stand_stability >= 0.2:
        cat = 3  # выраженное
    else:
        cat = 4  # максимальная декомпенсация

    return {
        "sit_start_sec": float(t1),
        "stand_sec": float(t2),
        "sit_end_sec": float(t3),
        "t_up_sec": float(t_up),
        "t_down_sec": float(t_down),
        "cycle_duration_sec": float(cycle_duration),
        "failed_attempts": int(fails),
        "uses_hands": bool(uses_hands),
        "arms_crossed": bool(arms_crossed),
        "trunk_angle_deg": float(angle),
        "hip_delta_px": float(hip_delta),
        "stand_stability": float(stand_stability),  # новая метрика
        "wrist_assist_ratio": float(wrist_low_ratio),  # для отладки
        "category": cat,
        "category_desc": ["норма", "лёгкое нарушение", "умеренное нарушение", 
                         "выраженное нарушение", "максимальная декомпенсация"][cat]
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

# Экспортируем функции для использования в других модулях
def get_cycle_metrics(df: pd.DataFrame, sit1: int, stand: int, sit2: int, fps: float) -> Dict:
    """Публичная версия функции _cycle_metrics"""
    return _cycle_metrics(df, sit1, stand, sit2, fps)

def get_summary(cycles: List[Dict]) -> Dict:
    """Публичная версия функции _summary"""
    return _summary(cycles)

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