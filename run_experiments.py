#!/usr/bin/env python3
"""
Скрипт для автоматизации экспериментов по fine-tuning
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Выполняет команду и выводит результат"""
    print(f"Выполняется: {description}")
    print(f"Команда: {command}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Успешно выполнено")
            if result.stdout:
                print(f"Вывод:\n{result.stdout}")
        else:
            print(f"Ошибка при выполнении")
            print(f"Код ошибки: {result.returncode}")
            if result.stderr:
                print(f"Ошибки:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Исключение: {e}")
        return False

def main():
    print("Запуск экспериментов по fine-tuning")
    print("=" * 50)
    
    # Проверяем, что мы в правильной директории
    if not os.path.exists("experiments") or not os.path.exists("app"):
        print("Ошибка: Запустите скрипт из корневой директории проекта")
        sys.exit(1)
    
    # 1. Подготовка данных
    success = run_command("python3 prepare_data.py", "Подготовка данных (разделение на train/val)")
    if not success:
        print("Ошибка при подготовке данных. Продолжаем...")
    
    # 2. Обучение ResNet18
    print("\n" + "=" * 50)
    success = run_command("cd experiments && python3 train.py --model_name resnet18 --epochs 10 --export_onnx", 
                         "Обучение ResNet18")
    if not success:
        print("Продолжаем с EfficientNet...")
    
    # 3. Обучение EfficientNet-B0
    print("\n" + "=" * 50)
    success = run_command("cd experiments && python3 train.py --model_name efficientnet_b0 --epochs 10 --export_onnx", 
                         "Обучение EfficientNet-B0")
    if not success:
        print("Обучение EfficientNet завершилось с ошибкой")
    
    # 4. Инструкции по дальнейшим действиям
    print("\n" + "=" * 50)
    print("Для анализа результатов запустите Jupyter notebook:")
    print("cd experiments && jupyter notebook notebook.ipynb")
    print("\nДля запуска веб-приложения:")
    print("cd app && python3 app_gradio.py")
    print("\nВсе эксперименты завершены!")
    print("Проверьте папку experiments/models для сохраненных моделей")

if __name__ == "__main__":
    main()

