import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
from datetime import datetime

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class ChairRiseAnalyzer:
    def __init__(self, show_processing=False, save_output=False):
        self.show_processing = show_processing
        self.save_output = save_output
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def analyze_video(self, video_path):
        """Анализирует видео выполнения пробы "Вставание с кресла"""
        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            return None, None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Не удалось открыть видео: {video_path}")
                return None, None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.save_output:
                output_path = self._get_output_path(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Результаты анализа
            results = {
                'actions': [],  # Все действия (вставания и приседания)
                'current_action': None,
                'partial_attempts': 0,  # Количество частичных попыток вставания
                'hands_crossed': False,
                'posture_score': 0,
                'errors': []
            }
            
            # Состояние анализа
            state = {
                'prev_hip_y': None,
                'is_standing': False,
                'movement_history': [],
                'frame_count': 0,
                'prev_wrist_pos': None,
                'last_action_time': None,
                'max_hip_y': 0,  # Максимальная высота бедер (для определения частичных попыток)
                'min_hip_y': 1,  # Минимальная высота бедер
                'current_movement': None  # Текущее движение ('up' или 'down')
            }
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                state['frame_count'] += 1
                
                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results_pose = self.pose.process(image)
                    
                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark
                        
                        # Анализ положения бедер
                        hip_y = self._get_hip_height(landmarks)
                        state['movement_history'].append(hip_y)
                        
                        # Обновляем min/max высоту бедер
                        state['max_hip_y'] = max(state['max_hip_y'], hip_y)
                        state['min_hip_y'] = min(state['min_hip_y'], hip_y)
                        
                        # Определяем направление движения
                        if state['prev_hip_y'] is not None:
                            if hip_y < state['prev_hip_y'] - 0.005:  # Порог для движения вверх
                                state['current_movement'] = 'up'
                            elif hip_y > state['prev_hip_y'] + 0.005:  # Порог для движения вниз
                                state['current_movement'] = 'down'
                        
                        state['prev_hip_y'] = hip_y
                        
                        # Определяем положение (сидит/стоит)
                        self._detect_actions(hip_y, results, state, fps)
                        
                        # Анализ частичных попыток вставания
                        self._detect_partial_attempts(hip_y, results, state)
                        
                        # Анализ положения рук
                        if not results['hands_crossed']:
                            results['hands_crossed'] = self._check_crossed_hands(landmarks)
                        
                        # Визуализация
                        if self.show_processing or self.save_output:
                            frame = self._draw_analysis(frame, results_pose, results, state)
                
                except Exception as e:
                    results['errors'].append(f"Frame {state['frame_count']}: {str(e)}")
                    continue
                
                if self.show_processing:
                    cv2.imshow('Chair Rise Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if self.save_output:
                    out.write(frame)
            
            # Завершаем последнее действие, если оно не завершено
            if results['current_action'] and not results['current_action']['completed']:
                results['current_action']['end_frame'] = state['frame_count']
                results['current_action']['completed'] = True
                results['actions'].append(results['current_action'])
            
            cap.release()
            if self.save_output:
                out.release()
                print(f"Результат сохранен в: {output_path}")
            
            if self.show_processing:
                cv2.destroyAllWindows()
            
            # Рассчет итоговых метрик
            score = self._calculate_score(results, fps)
            return score, results
            
        except Exception as e:
            print(f"Ошибка при обработке видео: {str(e)}")
            return None, None
    
    def _get_hip_height(self, landmarks):
        """Возвращает нормализованную высоту бедер"""
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        return (left_hip.y + right_hip.y) / 2
    
    def _detect_actions(self, hip_y, results, state, fps):
        """Определяет действия (вставание/приседание)"""
        # Пороговые значения (можно настроить)
        STAND_THRESHOLD = 0.55  # Ниже - стоит
        SIT_THRESHOLD = 0.65    # Выше - сидит
        MIN_ACTION_FRAMES = int(fps * 0.3)  # Минимальная длительность действия (0.3 сек)
        
        prev_standing = state['is_standing']
        state['is_standing'] = hip_y < STAND_THRESHOLD
        
        # Если состояние изменилось
        if prev_standing != state['is_standing']:
            current_time = state['frame_count'] / fps
            
            # Если было движение и прошло достаточно времени с последнего действия
            if (state['last_action_time'] is None or 
                (current_time - state['last_action_time']) > 0.5):
                
                # Завершаем предыдущее действие
                if results['current_action']:
                    results['current_action']['end_frame'] = state['frame_count']
                    results['current_action']['completed'] = True
                    results['actions'].append(results['current_action'])
                
                # Начинаем новое действие
                action_type = "stand" if state['is_standing'] else "sit"
                results['current_action'] = {
                    'type': action_type,
                    'start_frame': state['frame_count'],
                    'end_frame': None,
                    'completed': False,
                    'partial': False  # По умолчанию действие считается полным
                }
                state['last_action_time'] = current_time
    
    def _detect_partial_attempts(self, hip_y, results, state):
        """Обнаруживает частичные попытки вставания"""
        # Порог для определения частичной попытки (50% от полного диапазона движения)
        partial_threshold = state['min_hip_y'] + (state['max_hip_y'] - state['min_hip_y']) * 0.5
        
        # Если есть текущее действие "вставание" и движение вверх
        if (results['current_action'] and 
            results['current_action']['type'] == 'stand' and 
            state['current_movement'] == 'up'):
            
            # Если бедра поднялись выше порога, но не достигли положения стоя
            if (hip_y < partial_threshold and 
                hip_y > 0.55 and  # Абсолютный порог для положения стоя
                not results['current_action'].get('partial_marked')):
                
                results['current_action']['partial'] = True
                results['current_action']['partial_marked'] = True
                results['partial_attempts'] += 1
    
    def _check_crossed_hands(self, landmarks):
        """Проверяет, скрещены ли руки на груди"""
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Проверяем положение запястий относительно противоположных плеч
        left_wrist_near_right_shoulder = (
            abs(left_wrist.x - right_shoulder.x) < 0.15 and 
            abs(left_wrist.y - right_shoulder.y) < 0.2
        )
        right_wrist_near_left_shoulder = (
            abs(right_wrist.x - left_shoulder.x) < 0.15 and 
            abs(right_wrist.y - left_shoulder.y) < 0.2
        )
        
        return left_wrist_near_right_shoulder and right_wrist_near_left_shoulder
    
    def _draw_analysis(self, frame, results_pose, results, state):
        """Отрисовывает результаты анализа на кадре"""
        # Рисуем скелет
        mp_drawing.draw_landmarks(
            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Отображаем статус
        status = "STANDING" if state['is_standing'] else "SITTING"
        color = (0, 255, 0) if state['is_standing'] else (0, 0, 255)
        
        cv2.putText(frame, f"Status: {status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Отображаем количество действий
        stand_actions = len([a for a in results['actions'] if a['type'] == 'stand'])
        sit_actions = len([a for a in results['actions'] if a['type'] == 'sit'])
        cv2.putText(frame, f"Stands: {stand_actions}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Sits: {sit_actions}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Partial attempts: {results['partial_attempts']}", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Hands crossed: {'Yes' if results['hands_crossed'] else 'No'}", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Отображаем информацию о текущем действии
        if results['current_action']:
            action_time = (state['frame_count'] - results['current_action']['start_frame']) / 30.0
            action_text = f"Current: {results['current_action']['type']} ({action_time:.2f}s)"
            if results['current_action'].get('partial'):
                action_text += " (partial)"
            cv2.putText(frame, action_text, (20, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def _calculate_score(self, results, fps):
        """Рассчитывает итоговую оценку от 0 до 4 с учетом частичных попыток"""
        if not results['actions']:
            return 4  # Нет зафиксированных действий
        
        # Фильтруем только действия вставания
        stand_actions = [a for a in results['actions'] if a['type'] == 'stand']
        
        if not stand_actions:
            return 4  # Нет вставаний
        
        # Рассчитываем среднее время вставания только для полных вставаний
        full_stand_times = []
        for action in stand_actions:
            if action['completed'] and not action.get('partial'):
                duration = (action['end_frame'] - action['start_frame']) / fps
                full_stand_times.append(duration)
        
        # Если есть только частичные вставания
        if not full_stand_times:
            return 4  # Не смог полностью встать
        
        avg_stand_time = sum(full_stand_times) / len(full_stand_times)
        
        # Критерии оценки с учетом частичных попыток
        score = 0  # Начинаем с лучшей оценки
        
        # Штрафы за различные проблемы
        if avg_stand_time > 5.0:
            score += 3  # Очень медленно
        elif avg_stand_time > 3.0:
            score += 2  # Медленно
        elif avg_stand_time > 2.0:
            score += 1  # Немного медленно
        
        if results['partial_attempts'] > 0:
            score += min(2, results['partial_attempts'])  # Штраф за частичные попытки
        
        if not results['hands_crossed']:
            score += 1  # Руки не скрещены
        
        # Ограничиваем оценку диапазоном 0-4
        return min(4, score)
    
    def _get_output_path(self, video_path):
        """Генерирует путь для сохранения обработанного видео"""
        dir_name = "processed_videos"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(dir_name, f"{name}_processed_{timestamp}{ext}")

def main():
    parser = argparse.ArgumentParser(
        description='Анализатор пробы "Вставание с кресла" с использованием MediaPipe')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, 
                      help='Путь к папке с видео для обработки')
    group.add_argument('--video', type=str, 
                      help='Путь к конкретному видеофайлу для обработки')
    
    parser.add_argument('--output', type=str, default='results.csv',
                      help='Имя файла для результатов (по умолчанию: results.csv)')
    parser.add_argument('--show', action='store_true',
                      help='Показывать процесс обработки видео')
    parser.add_argument('--save', action='store_true',
                      help='Сохранять видео с разметкой')
    
    args = parser.parse_args()
    
    analyzer = ChairRiseAnalyzer(show_processing=args.show, save_output=args.save)
    results = []
    
    if args.video:
        print(f"\nАнализ видео: {args.video}")
        score, details = analyzer.analyze_video(args.video)
        if score is not None:
            results.append((os.path.basename(args.video), score))
            print("\nРезультаты анализа:")
            print(f"Оценка: {score}")
            print(f"Руки скрещены: {'Да' if details['hands_crossed'] else 'Нет'}")
            print(f"Частичные попытки вставания: {details['partial_attempts']}")
            
            # Выводим информацию о всех действиях
            stand_actions = [a for a in details['actions'] if a['type'] == 'stand']
            sit_actions = [a for a in details['actions'] if a['type'] == 'sit']
            
            print(f"\nВсего вставаний: {len(stand_actions)}")
            for i, action in enumerate(stand_actions, 1):
                duration = (action['end_frame'] - action['start_frame']) / 30.0
                status = "частичное" if action.get('partial') else "полное"
                print(f"Вставание {i}: {duration:.2f} сек ({status})")
            
            print(f"\nВсего приседаний: {len(sit_actions)}")
            for i, action in enumerate(sit_actions, 1):
                duration = (action['end_frame'] - action['start_frame']) / 30.0
                print(f"Приседание {i}: {duration:.2f} сек")
        else:
            print("Не удалось проанализировать видео")
    else:
        print(f"\nАнализ видео в папке: {args.input}")
        for filename in sorted(os.listdir(args.input)):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(args.input, filename)
                print(f"\nОбработка: {filename}")
                score, _ = analyzer.analyze_video(video_path)
                if score is not None:
                    results.append((filename, score))
                    print(f"Оценка: {score}")
                else:
                    print(f"Не удалось проанализировать видео: {filename}")
    
    if results:
        with open(args.output, 'w') as f:
            f.write("Video,Score,Stands,PartialAttempts\n")
            for video, score in results:
                f.write(f"{video},{score}\n")
        print(f"\nРезультаты сохранены в {args.output}")
    else:
        print("Нет результатов для сохранения")

if __name__ == "__main__":
    main()