import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import time
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = 'dataset-balls/train'
test_dir = 'dataset-balls/test'




class AdvancedBallClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedBallClassifierCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Dropout(0.8), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def show_top_3_absolute_best(model, data_loader, class_names):
    print("\nScanning the entire test set for the top 3 most best predictions")
    model.eval()

    all_results = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)


            max_probs, predicted_indices = torch.max(probabilities, 1)

            for i in range(images.size(0)):
                highest_prob = max_probs[i].item() * 100
                pred_class = class_names[predicted_indices[i]]
                true_label = class_names[labels[i]]
                
                # Save just the essential info
                all_results.append({
                    'prob': highest_prob,
                    'pred_class': pred_class,
                    'true_label': true_label
                })


    all_results.sort(key=lambda x: x['prob'], reverse=True)


    print("\n TOP 3 ")
    for i in range(3):
        res = all_results[i]
        print(f"{i+1}. Predicted: {res['pred_class']} ({res['prob']:.4f}%) | True Label: {res['true_label']}")



if __name__ == '__main__':
    try:
        train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)


        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=3)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=3)


        num_classes = len(train_data.classes)
        class_names = train_data.classes
        print(f"Classes found: {class_names}")
    except FileNotFoundError:
        print("Error: Dataset directories not found. Please verify paths.")
        exit()

    model = AdvancedBallClassifierCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


    epochs = 30

    print(f"\nStarting Training for {epochs} epochs...")
    start_time = time.time()


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
       

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

           
            running_loss += loss.item()


        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")



    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\nTotal Runtime for Training: {total_time:.2f} minutes")


    print("\nStarting Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []



    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
           
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)


    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Ball Classification', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    show_top_3_absolute_best(model, test_loader, class_names)


