# Safety Helmet Detector

Проект по детекции защитных касок (Safety Helmets) с использованием MLOps инструментов: PyTorch Lightning, YOLOv8, Hydra, MLflow и DVC.

## Архитектура моделей

В проекте реализовано две архитектуры:

1.  **YOLOv8** (Основаня модель) — используется как основное решение благодаря высокой скорости и точности. По умолчанию используется `yolov8m`.
2.  **Faster R-CNN** (Бейзлайн) — классическая двухстадийная архитектура на базе ResNet50, используется для сравнения результатов.

## Особенности проекта

- **Конфигурирование**: Hydra позволяет гибко управлять параметрами обучения, данных и моделей.
- **Логирование**: MLflow отслеживает метрики, параметры и сохраняет артефакты (модели).
- **Автоматизация**: Автоматическая конвертация обученных моделей в формат **ONNX**.
- **CLI**: Удобный интерфейс управления через `commands.py`.

## Setup

В этом проекте для управления зависимостями используется [Poetry](https://python-poetry.org/).

### 1. Установка проекта

```bash
git clone <your-repo-url>
cd safety-helmet-detector
poetry install
source $(poetry env info --path)/bin/activate
pre-commit install
```

### 2. Запуск MLflow

Перед началом обучения поднимите локальный сервер:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Training

По умолчанию запускается обучение **YOLOv8**.

### Основная модель (YOLOv8)

```bash
python -m safety_helmet_detection.commands train
```

### Бейзлайн (Faster R-CNN)

Чтобы запустить обучение бейзлайна, нужно переопределить конфиг модели:

```bash
python -m safety_helmet_detection.commands train model=fasterrcnn
```

### Полезные параметры

- `train.epochs=N` — количество эпох.
- `data.batch_size=N` — размер батча.
- `data.download=True` — скачать датасет автоматически перед обучением.
- `train.precision=32` — точность (по умолчанию 32 для стабильности на Mac MPS).

Пример:

```bash
python -m safety_helmet_detection.commands train train.epochs=20 data.batch_size=16
```

## Inference & Export

### Экспорт в ONNX

Проект автоматически экспортирует лучшую модель в ONNX после обучения. Вы также можете сделать это вручную:

```bash
python -m safety_helmet_detection.commands export \
    --checkpoint_path="outputs/yolo/train/weights/best.pt" \
    --model_type="yolo"
```

### Инференс (Stub)

```bash
python -m safety_helmet_detection.commands infer \
    --checkpoint_path="outputs/yolo/train/weights/best.pt" \
    --image_path="test.jpg"
```

## Структура проекта

- `src/` — исходный код.
- `configs/` — конфигурации Hydra.
- `safety-helmet-ds/` — данные (создается автоматически).
- `outputs/` — веса моделей, логи и ONNX файлы.
