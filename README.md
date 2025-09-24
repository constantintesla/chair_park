Ниже готовые файлы, которые можно сразу положить в корень репозитория.

---

### `README.md`
```markdown
# YOLO-pose Sit-to-Stand Analyzer  
**Соответствует протоколу 6.5.1.7 «Вставание с кресла (0-4)»**

## Кратко
Двух-шаговый пайплайн:  
1. `pose_tracker.py` — извлекает ключевые точки из видео (YOLOv8-pose) и сохраняет их в CSV.  
2. `movement_analysis.py` — ищет циклы «сидя-стоя-сидя», оценивает их по 5-бальной шкале (0 = норма, 4 = максимальная декомпенсация).

## Установка
```bash
git clone <repo>
cd <repo>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Использование
1. Скачайте веса YOLOv8-pose **один раз**:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
```

2. Извлечение ключевых точек:
```bash
python pose_tracker.py --input video.mp4 --output keypoints.csv --show
```
Флаг `--show` включает предпросмотр (нажмите `q` для выхода).

3. Анализ циклов:
```bash
python movement_analysis.py --input keypoints.csv --plot
```
Результат: `keypoints_movement.json` и график высоты таза с цветовыми полосами циклов.

## Пример JSON-отчёта
```json
{
  "summary": {
    "cycles_found": 4,
    "mean_t_up_sec": 1.12,
    "median_category": 0,
    "worst_category": 1,
    "uses_hands_any": false,
    "total_failed_attempts": 0
  },
  "cycles": [ {...}, {...}, {...}, {...} ]
}
```

## Критерии категорий (0-4)
| Категория | Описание |
|-----------|----------|
| 0         | Норма (встаёт ≤ 2 с, без рук, руки скрещены) |
| 1         | Лёгкое нарушение (≤ 5 с, без рук) |
| 2         | Умеренное (≤ 7 с, ≤ 2 попытки) |
| 3         | Выраженное (использует руки или > 2 попытки) |
| 4         | Максимальная декомпенсация (нужна посторонняя помощь) |

## Структура CSV-файла ключевых точек
Каждая строка = 1 кадр:  
`frame,time,kp_0_x,kp_0_y,kp_1_x,kp_1_y,...,kp_16_x,kp_16_y`

## Зависимости
См. `requirements.txt`

## Лицензия
MIT
```

---

### `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Модель и артефакты
yolov8*.pt
*.mp4
*.avi
*.mov
*.csv
*.json
!requirements.txt
!README.md
!pose_tracker.py
!movement_analysis.py

# OS
.DS_Store
Thumbs.db
```

Сохраните оба файла в корень проекта – и репозиторий готов к публикации.