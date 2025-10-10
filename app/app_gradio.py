import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import sys

# Добавляем путь к родительской директории для импорта
sys.path.append('..')

class BirdClassifier:
    def __init__(self, model_path="../experiments/models/resnet18.onnx"):
        """
        Инициализация классификатора птиц
        
        Args:
            model_path: путь к ONNX модели
        """
        # Построим абсолютный путь к модели относительно корня проекта
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        default_model_path = os.path.join(project_root, "experiments", "models", "resnet18.onnx")
        # Если переданный путь не существует, используем дефолтный абсолютный
        resolved_model_path = model_path
        if not os.path.isabs(resolved_model_path):
            resolved_model_path = os.path.normpath(os.path.join(app_dir, model_path))
        if not os.path.exists(resolved_model_path):
            resolved_model_path = default_model_path

        self.model_path = resolved_model_path
        self.class_names = ["Совы", "Вороны", "Орлы"]
        self.class_descriptions = {
            "Совы": "Ночные хищные птицы с характерным лицевым диском и тихим полётом.",
            "Вороны": "Умные птицы семейства врановых, часто чёрного окраса.",
            "Орлы": "Крупные дневные хищники с мощным клювом и острым зрением."
        }
        
        # Загружаем ONNX модель
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            # Имя входного тензора читаем из самой модели
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"Модель успешно загружена: {self.model_path}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            # Пробуем альтернативный путь
            alt_path = "models/resnet18.onnx"
            try:
                alt_abs = os.path.join(project_root, alt_path)
                self.ort_session = ort.InferenceSession(alt_abs if os.path.exists(alt_abs) else alt_path)
                self.input_name = self.ort_session.get_inputs()[0].name
                print(f"Модель загружена по альтернативному пути: {alt_path}")
            except Exception as e2:
                print(f"Не удалось загрузить модель: {e2}")
                raise e2

    def preprocess(self, image):
        """
        Предобработка изображения для инференса
        
        Args:
            image: PIL Image
            
        Returns:
            np.array: предобработанное изображение
        """
        # Конвертируем в RGB и изменяем размер
        image = image.convert('RGB').resize((224, 224))
        
        # Конвертируем в numpy array и нормализуем
        img_data = np.array(image).astype('float32') / 255.0
        
        # Нормализация (ImageNet статистики)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_data = (img_data - mean) / std
        
        # Изменяем порядок каналов (H, W, C) -> (C, H, W)
        img_data = np.transpose(img_data, (2, 0, 1))
        
        # Добавляем batch dimension
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

    def classify(self, image):
        """
        Классификация изображения
        
        Args:
            image: PIL Image
            
        Returns:
            dict: результаты классификации
        """
        try:
            # Предобработка
            input_data = self.preprocess(image)
            
            # Инференс
            outputs = self.ort_session.run(None, {self.input_name: input_data})
            predictions = outputs[0][0]  # Убираем batch dimension
            
            # Softmax для получения вероятностей
            exp_predictions = np.exp(predictions - np.max(predictions))
            probabilities = exp_predictions / np.sum(exp_predictions)
            
            # Получаем предсказанный класс
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Формируем результаты
            results = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "description": self.class_descriptions[predicted_class],
                "all_probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities)
                }
            }
            
            return results
            
        except Exception as e:
            return {
                "error": f"Ошибка при классификации: {str(e)}",
                "predicted_class": "Ошибка",
                "confidence": 0.0
            }

# Создаем экземпляр классификатора
classifier = BirdClassifier()

def classify_image(image):
    """
    Функция для Gradio интерфейса
    
    Args:
        image: PIL Image
        
    Returns:
        str: результат классификации
    """
    if image is None:
        return "Пожалуйста, загрузите изображение"
    
    results = classifier.classify(image)
    
    if "error" in results:
        return results["error"]
    
    # Форматируем результат
    output = f"""
**Предсказанный класс:** {results['predicted_class']}
**Уверенность:** {results['confidence']:.2%}
**Описание:** {results['description']}

**Все вероятности:**
"""
    
    for class_name, prob in results['all_probabilities'].items():
        output += f"- {class_name}: {prob:.2%}\n"
    
    return output

# Создаем Gradio интерфейс
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Загрузите изображение птицы"),
    outputs=gr.Markdown(label="Результат классификации"),
    title="Классификация птиц",
    description="""
    Загрузите изображение птицы для автоматической классификации.
    
    Модель может распознать три типа птиц:
    - **Совы**
    - **Вороны**  
    - **Орлы**
    
    Модель обучена с использованием transfer learning на предобученных весах.
    """,
    examples=[
        # Здесь можно добавить примеры изображений
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Запуск Gradio приложения...")
    print("Приложение будет доступно по адресу: http://localhost:7860")
    interface.launch(
        server_name="127.0.0.1",
        server_port=None,  # авто-свободный порт
        share=True,
        show_error=True,
        inbrowser=True
    )
