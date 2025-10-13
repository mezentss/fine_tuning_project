import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import functional as TF
import timm
from dataclasses import dataclass
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import onnx
import onnxruntime as ort
from PIL import Image, ImageFile, UnidentifiedImageError

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


@dataclass
class TrainConfig:
    model_name: str = 'resnet18'
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    num_classes: int = 3
    freeze_epochs: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir: str = '../data/processed'
    save_dir: str = 'models'
    export_onnx: bool = False
    model_path: str = None  # Изменено на None по умолчанию
    class_names: list = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['class1', 'class2', 'class3']


class FineTuneModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError('Model architecture not supported for classifier replacement')

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_predictions, all_labels


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Строит графики обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # График потерь
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # График точности
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Строит матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()


def export_to_onnx(model, input_shape=(1, 3, 224, 224), onnx_path='model.onnx'):
    """Экспортирует модель в ONNX формат"""
    model.eval()

    # Создаем пример входных данных
    dummy_input = torch.randn(input_shape)

    # Экспортируем в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Модель экспортирована в {onnx_path}")

    # Проверяем ONNX модель
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX модель валидна")
    except Exception as e:
        print(f"Ошибка при проверке ONNX модели: {e}")


def main(config: TrainConfig):
    # Создаем папку для сохранения моделей
    os.makedirs(config.save_dir, exist_ok=True)

    import torch
    from torchvision import transforms

    # Разрешаем чтение частично поврежденных файлов
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Безопасный загрузчик изображений: возвращает заглушку, если файл не читается
    def safe_pil_loader(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except (UnidentifiedImageError, OSError):
            # Возвращаем простое серое изображение 224x224, чтобы не падать на даталоаде
            return Image.new('RGB', (224, 224), color=(128, 128, 128))

    def random_segment_crop_pil(img: Image.Image, num_segments: int = 4) -> Image.Image:
        """Кадрирует случайный горизонтальный сегмент и возвращает к размеру 224x224.
        Работает на PIL, чтобы сохранять стандартный 4D-тензор после ToTensor()."""
        try:
            width, height = img.size
            # Защита от экстремальных размеров
            if num_segments < 1:
                num_segments = 1
            seg_w = max(1, width // num_segments)
            # Случайный индекс сегмента
            idx = random.randint(0, num_segments - 1)
            left = idx * seg_w
            right = width if idx == num_segments - 1 else (idx + 1) * seg_w
            # Кадрируем горизонтальный сегмент полностью по высоте
            img = img.crop((left, 0, right, height))
            # Возвращаем обратно к 224x224 для совместимости с сетями
            img = TF.resize(img, [224, 224])
            return img
        except Exception:
            # В случае проблем возвращаем исходное изображение
            return TF.resize(img, [224, 224])

    # Определяем трансформации
    train_transform = transforms.Compose([
        # Аугментация: случайный горизонтальный сегмент
        transforms.Lambda(lambda img: random_segment_crop_pil(img, num_segments=4)),
        # Дополнительные стандартные аугментации
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем данные
    train_dataset = datasets.ImageFolder(config.data_dir + '/train', transform=train_transform, loader=safe_pil_loader)
    val_dataset = datasets.ImageFolder(config.data_dir + '/val', transform=val_transform, loader=safe_pil_loader)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Обучающих примеров: {len(train_dataset)}")
    print(f"Валидационных примеров: {len(val_dataset)}")
    print(f"Классы: {train_dataset.classes}")

    device = config.device
    model = FineTuneModel(config.model_name, config.num_classes).to(device)
    print(f"Модель: {config.model_name}")
    print(f"Устройство: {device}")

    # Если указан путь к предобученной модели, загружаем ее
    if config.model_path and os.path.exists(config.model_path):
        print(f"Загружаем предобученную модель: {config.model_path}")
        model.load_state_dict(torch.load(config.model_path, map_location=device))
    else:
        print("Обучаем модель с нуля")

    # Freeze backbone initially
    for param in model.model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    if hasattr(model.model, 'fc'):
        for param in model.model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model.model, 'classifier'):
        for param in model.model.classifier.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # Списки для сохранения метрик
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    best_predictions = None
    best_labels = None

    print("\nНачинаем обучение...")
    for epoch in range(config.epochs):
        if epoch == config.freeze_epochs:
            print(f"\nРазмораживаем backbone на эпохе {epoch + 1}")
            for param in model.model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_predictions, val_labels = validate_one_epoch(model, val_loader, criterion, device)

        # Сохраняем метрики
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_predictions = val_predictions
            best_labels = val_labels
            model_path = os.path.join(config.save_dir, f'best_{config.model_name}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Новая лучшая модель сохранена: {model_path}")

    # Строим графики
    plot_path = os.path.join(config.save_dir, f'training_curves_{config.model_name}.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)

    # Строим матрицу ошибок
    if best_predictions is not None:
        cm_path = os.path.join(config.save_dir, f'confusion_matrix_{config.model_name}.png')
        plot_confusion_matrix(best_labels, best_predictions, config.class_names, cm_path)

        # Выводим отчет о классификации
        print("\nОтчет о классификации:")
        print(classification_report(best_labels, best_predictions, target_names=config.class_names))

    print(f"\nЛучшая точность: {best_acc:.4f}")

    # Экспорт в ONNX
    if config.export_onnx:
        model_path = os.path.join(config.save_dir, f'best_{config.model_name}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            onnx_path = os.path.join(config.save_dir, f'{config.model_name}.onnx')
            export_to_onnx(model, onnx_path=onnx_path)
        else:
            print("Модель не найдена для экспорта в ONNX")


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning script')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='Model name from timm')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Number of epochs to freeze backbone')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Data directory')
    parser.add_argument('--model_path', type=str, default=None,  # Добавлен аргумент model_path
                        help='Path to pre-trained model for fine-tuning')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_epochs=args.freeze_epochs,
        export_onnx=args.export_onnx,
        data_dir=args.data_dir,
        model_path=args.model_path  # Добавлено
    )

    main(config)