# Safety Helmet Detector

Проект по детекции наличия защитных касок (Safety Helmets) у людей на предприятии.

## Цель проекта

Обеспечение безопасности людей на предприятии путем фиксации соблюдения технический требований и преждевременного предупреждения в случае их нарушения.

## Архитектура моделей

В проекте реализовано две архитектуры:

1.  **YOLOv8** (Основаня модель) — используется как основное решение благодаря высокой скорости и точности. По умолчанию используется `yolov8m`. (Так же предполагается инференс на потоковом видео, где одностадийная модель выигрывает по скорости бьейзлайн)
2.  **Faster R-CNN** (Бейзлайн) — классическая двухстадийная архитектура на базе ResNet50, используется для сравнения результатов.

## Особенности проекта

- **Конфигурирование**: Hydra позволяет гибко управлять параметрами обучения, данных и моделей.
- **Логирование**: MLflow отслеживает метрики, параметры и сохраняет артефакты (модели).
- **Автоматизация**: Автоматическая конвертация обученных моделей в формат **ONNX**.
- **CLI**: Удобный интерфейс управления через `commands.py`.

## Setup

В этом проекте для управления зависимостями используется [Poetry](https://python-poetry.org/).
Для обеспечения качества кода используются `pre-commit` хуки.

### 1. Установка инструментов

- Python 3.9+
- Poetry (инструкция: https://python-poetry.org/docs/#installation)
- Git

#### Poetry

- **macOS / Linux / WSL:**
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
- **Windows (PowerShell):**
  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
  ```
_Примечание: После poetry install может потребоваться добавить `~/.local/bin` в ваш `PATH`. Выполните:_
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Подробная инструкция: [Poetry docs](https://python-poetry.org/docs/#installation).


### 2. Установка проекта и активация окружения

```bash
git clone
cd safety-helmet-detector
poetry install
source $(poetry env info --path)/bin/activate
pre-commit install
```

#### Проверка корректность установки

```bash
pre-commit -a run
```

<picture>
  <img alt="" src="https://github.com/dmsezv/safety-helmet-detector/blob/main/readme_img/tests_acc.png">
</picture>

### 2. Данные

Для обучения моделей используется датасет из открытых источников, На данный момент он размещен на Google Drive и ссылка добавлена в конфиг файлах. Датасет содержит 3 класса, полностью размечен и подготовлен для обучения

```
"Helmet": 1
"Head": 2
"Person": 3,
```

Проект настроен на **автоматическую загрузку данных**.

> Tсли данные отсутствуют, система автоматически скачивает ZIP-архив с Google Drive и распакуетт его в папку `safety-helmet-ds` при первом старте обучения.

<picture>
  <img alt="" src="https://github.com/dmsezv/safety-helmet-detector/blob/main/readme_img/ds_load.png">
</picture>

### 3. Запуск MLflow

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

<picture>
  <img alt="" src="https://github.com/dmsezv/safety-helmet-detector/blob/main/readme_img/ml_flow_1.png">
</picture>
<picture>
  <img alt="" src="https://github.com/dmsezv/safety-helmet-detector/blob/main/readme_img/ml_flow_2.png">
</picture>

## Inference & Export

### Экспорт в ONNX

Проект автоматически экспортирует лучшую модель в ONNX после обучения. Вы также можете сделать это вручную:

```bash
python -m safety_helmet_detection.commands export \
    --checkpoint_path="outputs/yolo/train/weights/best.pt" \
    --model_type="yolo"
```

### Инференс (Stub)

Предполагается использовать инференс с потоковым видео и выводом результатов online. Для реализации используются инструменты ffmpeg, open CV

>Не реализован в рамках текущей задачи

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
