#!/usr/bin/env python3
"""
Создание тестовых данных для проекта fine-tuning
"""

import os
import shutil
from PIL import Image, ImageDraw
import random

def create_test_image(width=224, height=224, color=(128, 128, 128), text="TEST"):
    """Создает простое тестовое изображение"""
    img = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(img)
    
    # Добавляем простой текст
    try:
        draw.text((10, 10), text, fill=(255, 255, 255))
    except:
        # Если не удается добавить текст, просто рисуем прямоугольник
        draw.rectangle([10, 10, 50, 50], fill=(255, 255, 255))
    
    return img

def create_test_dataset():
    """Создает тестовый датасет с 3 классами по 30 изображений"""
    
    # Удаляем старые данные
    if os.path.exists("data/raw"):
        shutil.rmtree("data/raw")
    if os.path.exists("data/processed"):
        shutil.rmtree("data/processed")
    
    # Создаем структуру папок
    os.makedirs("data/raw/class1", exist_ok=True)
    os.makedirs("data/raw/class2", exist_ok=True)
    os.makedirs("data/raw/class3", exist_ok=True)
    
    # Цвета для разных классов
    colors = [
        (255, 100, 100),  # Красноватый для class1
        (100, 255, 100),  # Зеленоватый для class2
        (100, 100, 255),  # Синеватый для class3
    ]
    
    # Создаем изображения для каждого класса
    for class_idx in range(1, 4):
        class_name = f"class{class_idx}"
        color = colors[class_idx - 1]
        
        print(f"Создание изображений для {class_name}...")
        
        for i in range(30):
            # Создаем изображение с небольшими вариациями
            base_color = list(color)
            # Добавляем случайные вариации цвета
            for j in range(3):
                base_color[j] = max(0, min(255, base_color[j] + random.randint(-50, 50)))
            
            img = create_test_image(
                width=224, 
                height=224, 
                color=tuple(base_color),
                text=f"{class_name}_{i+1}"
            )
            
            # Сохраняем изображение
            filename = f"{class_name}_{i+1:02d}.jpg"
            filepath = os.path.join("data/raw", class_name, filename)
            img.save(filepath, "JPEG")
    
    print("Тестовый датасет создан успешно!")
    print("Структура:")
    for class_name in ["class1", "class2", "class3"]:
        count = len(os.listdir(f"data/raw/{class_name}"))
        print(f"  {class_name}: {count} изображений")

if __name__ == "__main__":
    create_test_dataset()

