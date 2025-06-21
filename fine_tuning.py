import torch  
import torch.nn as nn  
from torchvision import (
    datasets,
    transforms,
)  
from torch.utils.data import (
    DataLoader,
)  
import torch.nn.functional as F  
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import numpy as np
from PIL import Image  
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt  
from sklearn.metrics import classification_report  
import torchvision.models as models
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Treat the data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Inverte horizontalmente a imagem com uma probabilidade de 0.5
    transforms.RandomRotation(45),     # Rotaciona a imagem aleatoriamente em até 45 graus
    transforms.RandomAutocontrast(),   # Ajusta o contraste da imagem aleatoriamente
    transforms.Resize((224, 224)),     # Dimensão 224 x 224(Tamanho padrão de redes famosas)
    transforms.ToTensor(),             # Converte para Tensor
    
])

dataset = datasets.ImageFolder(root='dataset', transform=transform)
class_names = dataset.classes
print(class_names)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
dataloaders = {'train': train_loader,
               'eval': test_loader}
dataset_sizes = {'train': len(train_dataset),
                 'eval': len(train_dataset)}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            num_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects = torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.sample_dict())

    print(f'Best model acc: {best_acc}')
    model.load_state_dict(best_model_wts)
    return model

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
for m in model.modules():
    if isinstance(m, nn.Module) and m.__class__.__name__ == 'Detect':
        detect_layer = m
        break
if detect_layer is None:
    raise ValueError("Could not find the 'Detect' layer in the YOLOv5 model. "
                     "Model structure might have changed or is unexpected.")
# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

original_detect_layer = detect_layer

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=5)

# # --- Recommendation ---
# If your goal is to use YOLOv5 for **object detection** (predicting bounding boxes AND classes), the most robust and recommended way to fine-tune it with your 4 classes is to **use the official YOLOv5 `train.py` script.**

# 1.  **Organize your data:** Ensure your images and labels are in the YOLOv5 format.
#     * Images in `dataset/images/train/`, `dataset/images/val/`
#     * Labels in `dataset/labels/train/`, `dataset/labels/val/` (each label is a `.txt` file for an image, containing `class_id x_center y_center width height` per object)
# 2.  **Create a `data.yaml` file:**
#     ```yaml
#     # data.yaml
#     train: /path/to/your/dataset/images/train/
#     val: /path/to/your/dataset/images/val/

#     # number of classes
#     nc: 4

#     # class names
#     names: ['area_degradada', 'construcao_irregular', 'iluminacao_pracaria', 'lixo']
#     ```
# 3.  **Run `train.py`:**
#     ```bash
#     git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) # if you don't have it
#     cd yolov5
#     pip install -r requirements.txt

#     # Then train
#     python train.py --img 640 --batch 16 --epochs 50 --data C:\Users\pc-de-caselli\Desktop\Campust-Party-Scripts\yolov5_ready_dataset\data.yaml --weights yolov5s.pt --name yolov5s_custom --cache
#     ```
#     * `--img 640`: Input image size.
#     * `--batch 16`: Batch size.
#     * `--epochs 50`: Number of training epochs.
#     * `--data path/to/your/data.yaml`: Path to your custom data configuration.
#     * `--weights yolov5s.pt`: Start from a pre-trained YOLOv5s model.
#     * `--name yolov5s_custom`: Name for your training run results.
#     * `--cache`: Caches images for faster training (if memory permits).

# This approach correctly handles the modification of the detection head, initializes weights properly, and manages the entire training pipeline designed for YOLOv5.

# If you are trying to *only* classify the entire image into one of the 4 categories (not detect objects), then YOLOv5 is likely an overkill and a standard classification model (like a fine-tuned ResNet) would be much more appropriate and easier to implement.