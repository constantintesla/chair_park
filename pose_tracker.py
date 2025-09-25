import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from filterpy.kalman import KalmanFilter


class PoseSmoother:
    def __init__(self, num_keypoints=17, dt=1.0, process_noise=1e-5, measurement_noise=1e-4):
        self.num_keypoints = num_keypoints
        self.kalman_filters = []
        
        # Создаем отдельный фильтр Калмана для каждой ключевой точки
        for _ in range(num_keypoints):
            kf = KalmanFilter(dim_x=4, dim_z=2)  # x, y, vx, vy
            kf.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])  # Матрица состояния
            
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])  # Матрица измерения
            
            kf.R = np.eye(2) * measurement_noise  # Шум измерения
            kf.Q = np.eye(4) * process_noise  # Шум процесса
            
            kf.P = np.eye(4) * 10  # Ковариационная матрица
            
            self.kalman_filters.append(kf)
    
    def update(self, keypoints):
        smoothed_keypoints = np.zeros_like(keypoints)
        
        for i, (x, y) in enumerate(keypoints):
            if i >= self.num_keypoints:
                break
                
            kf = self.kalman_filters[i]
            
            # Предсказание
            kf.predict()
            
            # Обновление на основе измерения
            if x > 0 and y > 0:  # Если точка обнаружена
                measurement = np.array([x, y])
                kf.update(measurement)
                # Исправление: преобразуем (2,1) в (2,)
                smoothed_keypoints[i] = kf.x[:2].flatten()
            else:
                # Если точка не обнаружена, используем предсказанное значение
                smoothed_keypoints[i] = kf.x[:2].flatten()
        
        return smoothed_keypoints

class SimpleSmoother:
    """Простой сглаживатель на основе скользящего среднего"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
    
    def update(self, keypoints):
        self.history.append(keypoints.copy())
        
        # Сохраняем только последние window_size кадров
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Если недостаточно данных, возвращаем текущее значение
        if len(self.history) < 2:
            return keypoints
        
        # Вычисляем среднее по истории
        smoothed = np.mean(self.history, axis=0)
        return smoothed

def process_video(input_path, output_path, show=False, smoothing=True):
    # Загрузка модели YOLOv8 Pose
    model = YOLO('yolov8s-pose.pt')
    
    # Инициализация сглаживателя
    if smoothing:
        smoother = PoseSmoother()
        print("Using Kalman filter for smoothing") 
    else:
        smoother = None
        print("Smoothing disabled")
    
    # Открытие видеофайла
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Получение информации о видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    
    # Подготовка для сохранения результатов
    all_keypoints = []
    frame_numbers = []
    
    # Обработка видео
    for frame_num in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Выполнение предсказания позы
        results = model(frame, verbose=False)
        
        # Получение ключевых точек для первого обнаруженного человека
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Сглаживание ключевых точек
            if smoother is not None:
                keypoints = smoother.update(keypoints)
            
            # Сохранение результатов
            frame_keypoints = {
                'frame': frame_num,
                'time': frame_num / fps
            }
            
            # Добавление координат каждой ключевой точки
            for i, (x, y) in enumerate(keypoints):
                frame_keypoints[f'kp_{i}_x'] = float(x)
                frame_keypoints[f'kp_{i}_y'] = float(y)
            
            all_keypoints.append(frame_keypoints)
            
            # Визуализация сглаженных точек
            if show:
                # Копируем кадр для аннотации
                annotated_frame = frame.copy()
                
                # Рисуем сглаженные ключевые точки
                for i, (x, y) in enumerate(keypoints):
                    if x > 0 and y > 0:
                        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, str(i), (int(x), int(y)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Рисуем соединения между точками
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Руки
                    (5, 11), (6, 12), (11, 12),  # Торс
                    (11, 13), (13, 15), (12, 14), (14, 16)  # Ноги
                ]
                
                for start, end in connections:
                    if (start < len(keypoints) and end < len(keypoints) and 
                        keypoints[start][0] > 0 and keypoints[start][1] > 0 and
                        keypoints[end][0] > 0 and keypoints[end][1] > 0):
                        start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
                        end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
                        cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)
                
                # Добавляем информацию о кадре
                cv2.putText(annotated_frame, f"Frame: {frame_num}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Smoothed Pose Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Отображение оригинальных результатов если нет обнаружений
        elif show:
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Frame: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Smoothed Pose Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Закрытие видео
    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    # Сохранение результатов в CSV
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total frames processed: {len(all_keypoints)}")
    else:
        print("No keypoints detected in the video.")

def main():
    parser = argparse.ArgumentParser(
        description="Human pose tracking using YOLOv8 Pose with smoothing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        required=True,
        help="Path to video file for analysis"
    )
    parser.add_argument(
        '--output',
        default="tracking_results.csv",
        help="Output file for results"
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="Show real-time analysis"
    )
    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help="Disable motion smoothing"
    )
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.show, not args.no_smoothing)

if __name__ == "__main__":
    main()