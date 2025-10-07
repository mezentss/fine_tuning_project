#!/usr/bin/env python3
"""
Скрипт для подготовки данных - разделение на train/val выборки
"""

import os
import shutil
import random
from pathlib import Path

def prepare_data():
    """Разделяет данные на обучающую и валидационную выборки"""
    
    # Фиксируем seed для воспроизводимости
    random.seed(42)
    
    data_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    # Создаем папки для классов
    for split in ['train', 'val']:
        for class_name in ['class1', 'class2', 'class3']:
            (processed_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("Разделение данных на train/val выборки...")
    
    for class_name in ['class1', 'class2', 'class3']:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"Предупреждение: папка {class_dir} не найдена")
            continue
            
        # Получаем все изображения
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        if not images:
            print(f"Предупреждение: в папке {class_dir} нет изображений")
            continue
            
        # Перемешиваем и разделяем (80/20)
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Копируем файлы
        for img in train_images:
            shutil.copy2(img, processed_dir / 'train' / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, processed_dir / 'val' / class_name / img.name)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print("Данные успешно подготовлены!")

if __name__ == '__main__':
    prepare_data()
