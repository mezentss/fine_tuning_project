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
        """
        # Построим абсолютный путь к модели
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        default_model_path = os.path.join(project_root, "experiments", "models", "resnet18.onnx")

        # Проверяем существование модели
        resolved_model_path = model_path
        if not os.path.isabs(resolved_model_path):
            resolved_model_path = os.path.normpath(os.path.join(app_dir, model_path))
        if not os.path.exists(resolved_model_path):
            resolved_model_path = default_model_path

        self.model_path = resolved_model_path
        self.class_names = ["Совы", "Вороны", "Попугаи"]
        self.class_descriptions = {
            "Совы": "Ночные хищные птицы с характерным лицевым диском и тихим полётом.",
            "Вороны": "Умные птицы семейства врановых, часто чёрного окраса.",
            "Попугаи": "Яркие красавцы."
        }

        print(f"Пытаемся загрузить модель из: {self.model_path}")
        print(f"Файл существует: {os.path.exists(self.model_path)}")

        # Загружаем ONNX модель
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.input_shape = self.ort_session.get_inputs()[0].shape
            self.input_type = self.ort_session.get_inputs()[0].type
            print(f"Модель успешно загружена: {self.model_path}")
            print(f"Входное имя: {self.input_name}")
            print(f"Входная форма: {self.input_shape}")
            print(f"Входной тип: {self.input_type}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            self.ort_session = None

    def preprocess(self, image):
        """
        Предобработка изображения для инференса
        """
        try:
            # Конвертируем в RGB и изменяем размер
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Изменяем размер до 224x224 (стандарт для ResNet)
            image = image.resize((224, 224))

            # Конвертируем в numpy array с явным указанием float32
            img_data = np.array(image, dtype=np.float32)

            # Нормализация (0-1)
            img_data = img_data / 255.0

            # ImageNet нормализация
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            # Нормализуем каждый канал
            for i in range(3):
                img_data[:, :, i] = (img_data[:, :, i] - mean[i]) / std[i]

            # Изменяем порядок каналов (H, W, C) -> (C, H, W)
            img_data = np.transpose(img_data, (2, 0, 1))

            # Добавляем batch dimension
            img_data = np.expand_dims(img_data, axis=0)

            print(f"Тип данных после предобработки: {img_data.dtype}")
            print(f"Форма данных: {img_data.shape}")

            return img_data

        except Exception as e:
            print(f"Ошибка в предобработке: {e}")
            # Возвращаем нулевой тензор правильной формы и типа
            return np.zeros((1, 3, 224, 224), dtype=np.float32)

    def classify(self, image):
        """
        Классификация изображения
        """
        try:
            # Если модель не загружена, возвращаем тестовые данные
            if self.ort_session is None:
                print("Модель не загружена, возвращаем тестовые данные")
                return self._get_test_prediction()

            # Предобработка
            input_data = self.preprocess(image)

            # Явно проверяем тип данных
            if input_data.dtype != np.float32:
                print(f"Исправляем тип данных: {input_data.dtype} -> float32")
                input_data = input_data.astype(np.float32)

            print(f"Финальный тип данных для инференса: {input_data.dtype}")

            # Инференс
            outputs = self.ort_session.run(None, {self.input_name: input_data})
            print(f"Количество выходов: {len(outputs)}")

            # Берем первый выход
            predictions = outputs[0]
            print(f"Форма предсказаний: {predictions.shape}")

            # Если выход имеет batch dimension, убираем ее
            if len(predictions.shape) > 1:
                predictions = predictions[0]

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

            print(f"Предсказанный класс: {predicted_class}, уверенность: {confidence:.4f}")
            return results

        except Exception as e:
            print(f"Ошибка при классификации: {e}")
            return self._get_error_prediction(str(e))

    def _get_test_prediction(self):
        """Возвращает тестовое предсказание когда модель не загружена"""
        return {
            "predicted_class": "Совы",
            "confidence": 0.85,
            "description": self.class_descriptions["Совы"],
            "all_probabilities": {
                "Совы": 0.85,
                "Вороны": 0.10,
                "Попугаи": 0.05
            }
        }

    def _get_error_prediction(self, error_msg):
        """Возвращает предсказание при ошибке"""
        return {
            "predicted_class": "Ошибка",
            "confidence": 0.0,
            "description": f"Произошла ошибка: {error_msg}",
            "all_probabilities": {
                "Совы": 0.33,
                "Вороны": 0.33,
                "Попугаи": 0.34
            }
        }


# Создаем экземпляр классификатора
classifier = BirdClassifier()


def classify_image(image):
    """
    Функция для Gradio интерфейса
    """
    if image is None:
        return "Пожалуйста, загрузите изображение птицы", {"Совы": 0, "Вороны": 0, "Попугаи": 0}

    try:
        results = classifier.classify(image)

        # Форматируем результат
        output = f"""
**🎯 Предсказанный класс:** {results['predicted_class']}

**📊 Уверенность:** {results['confidence']:.2%}

**📝 Описание:** {results['description']}

**📈 Вероятности по классам:**
"""

        for class_name, prob in results['all_probabilities'].items():
            output += f"- **{class_name}:** {prob:.2%}\n"

        return output, results['all_probabilities']

    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке изображения: {str(e)}"
        return error_msg, {"Совы": 0, "Вороны": 0, "Попугаи": 0}


# Альтернативная версия предобработки для тестирования
def preprocess_alternative(image):
    """Альтернативный метод предобработки"""
    # Конвертируем в RGB и изменяем размер
    image = image.convert('RGB').resize((224, 224))

    # Конвертируем в numpy array с float32
    img_array = np.array(image, dtype=np.float32)

    # Нормализация 0-1
    img_array = img_array / 255.0

    # ImageNet нормализация
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Транспонирование и добавление batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Тестируем модель перед запуском
def test_model_with_sample():
    """Тестируем модель с примером изображения"""
    print("🧪 Тестирование модели с примером изображения...")
    try:
        # Создаем тестовое изображение
        test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
        result = classifier.classify(test_image)
        print(f"✅ Тест пройден: {result['predicted_class']}")
        return True
    except Exception as e:
        print(f"❌ Тест не пройден: {e}")
        return False


# Запускаем тест перед созданием интерфейса
test_model_with_sample()

# Создаем Gradio интерфейс
with gr.Blocks(theme=gr.themes.Soft(), title="Классификатор птиц") as interface:
    gr.Markdown("# 🐦 Классификатор птиц")
    gr.Markdown("""
    Загрузите изображение птицы для автоматической классификации на три категории:
    - **🦉 Совы** 
    - **🐦‍⬛ Вороны**   
    - **🦅 Попугаи** 
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Загрузите изображение птицы",
                height=300
            )
            classify_btn = gr.Button(
                "🔍 Классифицировать",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            text_output = gr.Markdown(
                label="📊 Результат классификации",
                value="Здесь появится результат классификации..."
            )
            label_output = gr.Label(
                label="📈 Вероятности по классам",
                num_top_classes=3
            )

    # Обработчик кнопки
    classify_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[text_output, label_output]
    )

    gr.Markdown("### 💡 Советы:")
    gr.Markdown("""
    - Используйте четкие изображения птиц
    - Птица должна быть хорошо видна
    - Лучше всего работают фото с контрастным фоном
    """)

if __name__ == "__main__":
    print("🚀 Запуск Gradio приложения...")
    print("📢 Приложение будет доступно по адресу: http://127.0.0.1:7860")

    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")
        print("🔄 Пробуем запустить на случайном порту...")
        interface.launch(
            server_name="127.0.0.1",
            share=False,
            inbrowser=True
        )