# Safety Helmet Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Профессиональное MLOps-решение для автоматического контроля наличия средств индивидуальной защиты (защитных касок) на производственных объектах.

## Цели и задачи проекта

**Основная цель:** Обеспечение безопасности людей на предприятии путем фиксации соблюдения технический требований и преждевременного предупреждения в случае их нарушения.

### Ключевые задачи:

1.  **Детекция в реальном времени:** Обеспечение высокой скорости обработки видеопотока (инференса) для оперативного реагирования.
2.  **Высокая точность:** Минимизация ложных срабатываний и пропусков нарушений в сложных промышленных условиях (низкая освещенность, задымленность, перекрытие объектов).
3.  **Гибкая конфигурация:** Возможность быстрой перенастройки параметров обучения и выбора моделей без изменения исходного кода.
4.  **Воспроизводимость:** Полное логирование экспериментов, версионирование данных и конфигураций.
5.  **Готовность к продакшну:** Автоматизированный экспорт моделей в оптимизированные форматы (ONNX).

## Архитектура моделей

В проекте реализовано две архитектуры:

1.  **YOLOv8 (Основная модель):**

    - Модель: `yolov8m`.
    - Назначение: Основное решение для промышленной эксплуатации.
    - Преимущества: Одностадийная архитектура (Single Shot) обеспечивает идеальный баланс точности и FPS, что критично для работы с потоковым видео.

2.  **Faster R-CNN (Бейзлайн):**

    - База: `ResNet50-FPN`.
    - Назначение: Контрольная точка для оценки качества.
    - Реализация: PyTorch Lightning.
    - Преимущества: Классическая двухстадийная архитектура.

## Особенности проекта

Проект построен на современном MLOps стеке:

| Инструмент              | Назначение                                              |
| :---------------------- | :------------------------------------------------------ |
| **PyTorch / Lightning** | Обучение нейронных сетей и оркестрация пайплайнов       |
| **Ultralytics**         | Современная реализация YOLOv8                           |
| **Hydra**               | Иерархическое управление конфигурациями (YAML)          |
| **MLflow**              | Трекинг экспериментов, метрик и версионирование моделей |
| **Poetry**              | Строгое управление зависимостями и окружением           |
| **DVC**                 | Версионирование данных и удаленное хранение в GDrive    |
| **ONNX**                | Формат для высокопроизводительного инференса            |
| **Ruff / Pre-commit**   | Контроль качества кода и статический анализ             |
| **CLI**                 | Удобный интерфейс управления через `commands.py`        |

## Setup

В этом проекте для управления зависимостями используется [Poetry](https://python-poetry.org/).
Для обеспечения качества кода используются `pre-commit` хуки.

### 1. Установка инструментов

- Python 3.9+
- Poetry (инструкция: [Poetry docs](https://python-poetry.org/docs/#installation)
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

### 2. Установка проекта и активация окружения

```bash
# Клонирование репозитория
git clone https://github.com/dmsezv/safety-helmet-detector.git
cd safety-helmet-detector

# Установка зависимостей
poetry install

# Активация виртуального окружения
source $(poetry env info --path)/bin/activate

# Настройка git-хуков (linting & formatting)
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

**Классы детектирования:**

```
- `Helmet`
- `Head`
- `Person`
```

Проект настроен на **автоматическую загрузку данных**.

> Если данные отсутствуют, система автоматически скачивает ZIP-архив с Google Drive и распакует его в папку `safety-helmet-ds` и подготовит для работы при первом старте обучения.

<picture>
  <img alt="" src="https://github.com/dmsezv/safety-helmet-detector/blob/main/readme_img/ds_load.png">
</picture>

### 3. Версионирование данных (DVC)

Проект может использовать **DVC** для управления версиями датасета. Сейчас актуальная версия датасета хранится в виде архива на GDrive и автоматически загружается при первом запуске обучения, но в дальнейшем можно будет использовать DVC для более удобного управления версиями при подклоючении своего хранилища. Для работы с ним все полностью подготовлено

**Основные команды DVC:**

- **Получить данные** (если папка `safety-helmet-ds` пуста):
  ```bash
  dvc pull
  ```
- **Загрузить новую версию** (после изменений в датасете):
  ```bash
  dvc add safety-helmet-ds
  dvc push
  git add safety-helmet-ds.dvc
  git commit -m "Update dataset"
  ```

### 4. Трекинг экспериментов (MLflow)

Перед запуском обучения поднимите сервер MLflow для визуализации процесса:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Результаты будут доступны по адресу: `http://localhost:8080`.

> Если обучаете модель удаленно то используйте `--host 0.0.0.0` и подключайтесь по адресу: `http://<ip сервера>:8080`

## Обучение моделей (Training)

Все управление осуществляется через единую точку входа — `commands.py`.

> По умолчанию запускается обучение **YOLOv8**.

### Обучение YOLOv8 (по умолчанию)

```bash
python -m safety_helmet_detection.commands train
```

### Обучение Faster R-CNN (Бейзлайн)

```bash
python -m safety_helmet_detection.commands train model=fasterrcnn
```

### Гибкая настройка через CLI

Благодаря Hydra, вы можете менять любой параметр «на лету»:

```bash
python -m safety_helmet_detection.commands train \
    train.epochs=50 \
    data.batch_size=32 \
    model.lr=0.001 \
    data.download=True
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

## Экспорт и инференс

### Экспорт в ONNX

После завершения обучения модель автоматически конвертируется в ONNX. Вы также можете выполнить экспорт вручную:

```bash
python -m safety_helmet_detection.commands export \
    --checkpoint_path="outputs/yolo/train/weights/best.pt" \
    --model_type="yolo"
```

### Инференс (Stub)

Предполагается использовать инференс с потоковым видео и выводом результатов online. Для реализации используются инструменты ffmpeg, open CV

> Не реализован в рамках текущей задачи

```bash
python -m safety_helmet_detection.commands infer \
    --checkpoint_path="outputs/yolo/train/weights/best.pt" \
    --image_path="test.jpg"
```

## Структура проекта

```text
.
├── configs/                # Конфигурационные файлы Hydra (YAML)
│   ├── data/               # Настройки датасета
│   ├── model/              # Описания архитектур моделей
│   └── train/              # Гиперпараметры обучения
├── src/                    # Исходный код проекта
│   └── safety_helmet_detection/
│       ├── commands.py     # Главный CLI интерфейс
│       ├── train.py        # Оркестрация обучения
│       ├── data/           # Загрузчики и обработка данных
│       └── models/         # Определения моделей
├── safety-helmet-ds/       # Директория данных (создается автоматически)
├── outputs/                # Артефакты: веса, логи, ONNX модели
└── pyproject.toml          # Зависимости Poetry
```

## Лицензия

Распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](LICENSE).
