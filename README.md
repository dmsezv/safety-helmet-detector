# Safety Helmet Detector

Проект по детекции защитных касок (Safety Helmets) с использованием PyTorch Lightning, Hydra и DVC.

## Описание проекта

Проект решает задачу детекции объектов (Object Detection) на строительных площадках.
Целевые классы:

- `Helmet` (Каска)
- `Head` (Голова без каски)
- `Person` (Человек)

## Setup

В этом проекте для управления зависимостями используется [Poetry](https://python-poetry.org/).
Для обеспечения качества кода используются `pre-commit` хуки.

### 1. Предварительные требования

- Python 3.9+
- Poetry (инструкция: https://python-poetry.org/docs/#installation)
- Git

### 2. Установка окружения

Следуйте этим шагам для настройки чистого окружения:

1. **Клонируйте репозиторий:**

   ```bash
   git clone <your-repo-url>
   cd safety-helmet-detector
   ```

2. **Создайте и активируйте виртуальное окружение, установите зависимости:**

   ```bash
   poetry install
   poetry shell
   ```

3. **Установите pre-commit хуки:**
   Это обязательно для проверки качества кода перед коммитами.

   ```bash
   pre-commit install
   ```

4. **Проверьте работоспособность хуков:**
   Запустите проверку на всех файлах. Все проверки должны пройти успешно (Passed).
   ```bash
   pre-commit run -a
   ```

## Data

Данные хранятся на Google Drive. Проект настроен на автоматическое скачивание данных при необходимости.

Ссылка на датасет (для справки): [Google Drive Folder](https://drive.google.com/drive/folders/1Jc-z_kAirl-vE65zpc1U3e73oX0YBzIz?usp=sharing).
**Важно:** Ссылка уже сконфигурирована в `configs/data/default.yaml` и не требует ручного скачивания пользователем.

Для управления версионированием данных используется DVC (при наличии настроенного remote). В данном учебном примере реализована функция `download_data`, которая использует `gdown` для загрузки данных с Google Drive, если они отсутствуют локально.

Структура данных после скачивания (`safety-helmet-ds/`):

```
safety-helmet-ds/
├── images/
│   └── *.png
└── annotations/
    └── *.xml
```

## Train

Запуск тренировки осуществляется через CLI утилиту `commands.py`.

### Запуск обучения

Для запуска обучения с дефолтными параметрами:

```bash
python -m safety_helmet_detection.commands train
```

**Если данные еще не скачаны**, добавьте флаг `data.download=True`:

```bash
python -m safety_helmet_detection.commands train data.download=True
```

Система автоматически скачает датасет с Google Drive в папку `safety-helmet-ds` перед началом обучения.

### Конфигурация обучения

Вы можете переопределять любые параметры конфигурации (Hydra) через командную строку.

Примеры:

1. **Изменить количество эпох и размер батча:**

   ```bash
   python -m safety_helmet_detection.commands train train.epochs=20 data.batch_size=8
   ```

2. **Изменить Learning Rate:**

   ```bash
   python -m safety_helmet_detection.commands train model.lr=0.001
   ```

3. **Использовать GPU (если доступно):**
   ```bash
   python -m safety_helmet_detection.commands train train.accelerator=gpu train.devices=1
   ```

### Logging

Метрики обучения логируются в MLFlow.
Перед запуском обучения поднимите локальный сервер MLFlow:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Результаты будут доступны по адресу http://127.0.0.1:8080.
Логируются:

- Loss (train/val)
- Гиперпараметры
- Версия кода (Git commit)

## Inference

Для запуска инференса (предсказания) можно воспользоваться командой `infer`:

```bash
python -m safety_helmet_detection.commands infer --checkpoint_path="outputs/best_model.ckpt" --image_path="test_image.jpg"
```
