# Safety Helmet Detector

Проект по детекции защитных касок (Safety Helmets) с использованием PyTorch Lightning, Hydra и DVC.

## Setup

В этом проекте для управления зависимостями и упаковки используется [Poetry](https://python-poetry.org/).

### Предварительные требования

- Python 3.9 или выше
- Установленный Poetry (инструкция по установке: https://python-poetry.org/docs/#installation)

### Установка окружения

1. **Клонируйте репозиторий и перейдите в папку проекта:**

   ```bash
   git clone <your-repo-url>
   cd safety-helmet-detector
   ```

2. **Установите зависимости:**
   Команда создаст виртуальное окружение и установит все необходимые пакеты.

   ```bash
   poetry install
   ```

3. **Активируйте виртуальное окружение:**

   ```bash
   poetry shell
   ```

4. **Настройте pre-commit хуки:**
   Это необходимо для автоматической проверки стиля кода (Ruff, Prettier) перед каждым коммитом.

   ```bash
   pre-commit install
   ```

   Проверить, что всё настроено корректно, можно запустив проверку на всех файлах:

   ```bash
   pre-commit run -a
   ```

   Должен отобразиться список проверок с зеленым статусом `Passed`.

## Data

Используется датасет [Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection/data).
Структура данных ожидается в папке `safety-helmet-ds` в корне проекта:

```
safety-helmet-ds/
├── images/
│   └── *.png
└── annotations/
    └── *.xml
```

В проекте есть функция для автоматической загрузки данных (mock-реализация). Чтобы ее использовать, передайте флаг при запуске обучения (см. раздел Train).

## Configuration (Hydra)

Конфигурация проекта управляется через [Hydra](https://hydra.cc/). Конфигурационные файлы находятся в папке `configs/`.

Основные группы параметров:

- `data`: параметры датасета, такие как размер батча, размер изображения.
- `model`: параметры модели (FasterRCNN/YOLO), количество классов, learning rate.
- `train`: параметры тренера (количество эпох, accelerator, devices).
- `logger`: настройки MLFlow.

Вы можете менять любой параметр из командной строки, не изменяя файлы конфигов.

## Train

Для запуска команд используется единая точка входа `commands.py`.

### Запуск обучения

**Базовый запуск:**

**Базовый запуск:**

```bash
python -m safety_helmet_detection.commands train
```

Эта команда:

1. Инициализирует модель и датасет на основе конфигов в `configs/`.
2. Запустит обучение с использованием PyTorch Lightning.
3. Будет логировать метрики в MLFlow (если сервер доступен).

### Примеры настройки через CLI

**Изменить количество эпох и размер батча:**

```bash
python -m safety_helmet_detection.commands train --train.epochs=20 --data.batch_size=8
```

**Включить автоматическое скачивание данных (если они не существуют):**

```bash
python -m safety_helmet_detection.commands train --data.download=True
```

**Изменить Learning Rate:**

```bash
python -m safety_helmet_detection.commands train --model.lr=0.001
```

### Логирование (MLFlow)

Для сбора метрик (потери, параметры) используется MLFlow.

1. Запустите локальный сервер MLFlow (в отдельном окне терминала):
   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```
2. Убедитесь, что в конфиге `configs/logger/mlflow.yaml` указан верный `tracking_uri` (по умолчанию `http://127.0.0.1:8080`).

После запуска обучения графики лоссов (Classification, Box Regression, Objectness, RPN Box Regression) будут доступны в веб-интерфейсе MLFlow.

## Inference

Для запуска предсказания модели на новых изображениях используется команда `infer`.
_(Примечание: В текущей версии это заглушка для демонстрации интерфейса)_

```bash
python -m safety_helmet_detection.commands infer --checkpoint_path="outputs/.../best.ckpt" --image_path="test_image.jpg"
```
