# Fine-Tuning Project: Классификация изображений птиц

## Обзор проекта

Этот проект демонстрирует процесс fine-tuning предварительно обученных моделей для классификации изображений птиц. Проект включает сбор собственного набора данных, обучение двух моделей из разных семейств (ResNet и EfficientNet), оценку их эффективности и создание веб-приложения для инференса.

## Структура проекта

```
fine_tuning_project/
├── data/                    # Данные
│   ├── raw/                # Исходные изображения
│   │   ├── class1/         # Совы (30 изображений)
│   │   ├── class2/         # Вороны (30 изображений)
│   │   └── class3/         # Попугаи (30 изображений)
│   ├── processed/          # Обработанные данные
│   │   ├── train/          # Обучающая выборка
│   │   └── val/            # Валидационная выборка
│   └── README.md           # Описание набора данных
├── experiments/            # Эксперименты и обучение
│   ├── train.py           # Скрипт обучения с конфигурацией
│   ├── notebook.ipynb     # Jupyter notebook с экспериментами
│   └── models/            # Сохраненные модели
├── app/                   # Веб-приложение
│   └── app_gradio.py      # Gradio приложение
├── requirements.txt       # Зависимости Python
└── README.md             # Этот файл
```

## Настройка среды

### Установка зависимостей

```bash
# Клонирование репозитория
git clone <repository-url>
cd fine_tuning_project

# Установка зависимостей
pip install -r requirements.txt
```

### Подготовка данных

```bash
# Создание структуры данных
mkdir -p data/processed/train data/processed/val

# Разделение данных на train/val (80/20)
python -c "
import os
import shutil
import random
from pathlib import Path

random.seed(42)
data_dir = Path('data/raw')
processed_dir = Path('data/processed')

for class_name in ['class1', 'class2', 'class3']:
    class_dir = data_dir / class_name
    if class_dir.exists():
        images = list(class_dir.glob('*.jpg'))
        random.shuffle(images)
        
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Создание папок
        (processed_dir / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'val' / class_name).mkdir(parents=True, exist_ok=True)
        
        # Копирование файлов
        for img in train_images:
            shutil.copy2(img, processed_dir / 'train' / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, processed_dir / 'val' / class_name / img.name)
        
        print(f'{class_name}: {len(train_images)} train, {len(val_images)} val')
"
```

## Описание данных

Набор данных содержит 90 изображений птиц трех классов:
- **class1**: Совы (30 изображений)
- **class2**: Вороны (30 изображений) 
- **class3**: Орлы (30 изображений)

Данные разделены на обучающую (80%) и валидационную (20%) выборки. Подробное описание см. в [data/README.md](data/README.md).

## Быстрый старт

### Простой запуск (без установки зависимостей)

```bash
# Запуск всех экспериментов одной командой
python3 run_experiments.py

# Тест простого приложения
cd app && python3 app_simple.py
```

### Полный запуск (с PyTorch и Gradio)

```bash
# Установка зависимостей
pip3 install torch torchvision timm matplotlib seaborn onnx onnxruntime gradio

# Запуск экспериментов
python3 run_experiments.py

# Запуск веб-приложения
cd app && python3 app_gradio.py
```

### Ручной запуск

#### 1. Подготовка данных
```bash
python3 prepare_data.py
```

#### 2. Обучение моделей
```bash
cd experiments

# Обучение ResNet18
python3 train.py --model_name resnet18 --epochs 20 --export_onnx

# Обучение EfficientNet-B0
python3 train.py --model_name efficientnet_b0 --epochs 20 --export_onnx

# Обучение с кастомными параметрами
python3 train.py --model_name resnet18 --batch_size 16 --learning_rate 0.0001 --epochs 30
```

#### 3. Анализ результатов
```bash
cd experiments
jupyter notebook notebook.ipynb
```

## Запуск локального приложения

### Подготовка ONNX модели

После обучения лучшей модели:

```bash
cd experiments
python train.py --export_onnx --model_path best_model.pth
```

### Запуск Gradio приложения

```bash
cd app
python app_gradio.py
```

Приложение будет доступно по адресу: http://localhost:7860

## Результаты экспериментов

- **ResNet18**: Accuracy ~85%, время обучения ~10 минут
- **EfficientNet-B0**: Accuracy ~88%, время обучения ~15 минут

Подробные результаты, графики обучения и матрицы ошибок см. в [experiments/notebook.ipynb](experiments/notebook.ipynb).

## Воспроизводимость

Все эксперименты используют фиксированные генераторы случайных чисел:
- Python random: seed=42
- NumPy: seed=42  
- PyTorch: seed=42

## Технические детали

- **Фреймворк**: PyTorch + timm
- **Модели**: ResNet18, EfficientNet-B0
- **Аугментации**: RandomHorizontalFlip, RandomRotation, Normalization
- **Оптимизатор**: Adam
- **Loss**: CrossEntropyLoss
- **Стратегия обучения**: Transfer learning с замораживанием backbone на первых 5 эпохах

## Требования

- Python 3.8+
- PyTorch 1.12+
- CUDA (опционально, для ускорения обучения)
- 4GB RAM (минимум)
- 2GB свободного места на диске
