import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

def process_video(input_path, output_path, show=False):
    # Загрузка модели YOLOv8 Pose
    model = YOLO('yolov8s-pose.pt')  # или 'yolov8s-pose.pt', 'yolov8m-pose.pt' и т.д.
    
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
        
        # Получение ключевых точек для первого обнаруженного человека (можно модифицировать для нескольких людей)
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Сохранение результатов
            frame_keypoints = {
                'frame': frame_num,
                'time': frame_num / fps
            }
            
            # Добавление координат каждой ключевой точки
            for i, (x, y) in enumerate(keypoints):
                frame_keypoints[f'kp_{i}_x'] = x
                frame_keypoints[f'kp_{i}_y'] = y
            
            all_keypoints.append(frame_keypoints)
        
        # Отображение результатов в реальном времени (если нужно)
        if show:
            annotated_frame = results[0].plot()
            cv2.imshow('Pose Tracking', annotated_frame)
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
    else:
        print("No keypoints detected in the video.")

def main():
    parser = argparse.ArgumentParser(
        description="Human pose tracking using YOLOv8 Pose",
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
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.show)

if __name__ == "__main__":
    main()