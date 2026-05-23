import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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

def predict_image(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = AdvancedBallClassifierCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 
    img = Image.open(image_path).convert('RGB')

    input_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
        max_prob, predicted_idx = torch.max(probabilities, 1)
        
        accuracy = max_prob.item() * 100
        predicted_class = class_names[predicted_idx.item()]
        
        print(f"Prediction: {predicted_class}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        print("\nAll probabilities:")
        probs_list = probabilities[0].cpu().numpy()
        for cls, prob in zip(class_names, probs_list):
            print(f"  {cls}: {prob*100:.2f}%")


if __name__ == '__main__':

    my_classes = 'american_football', 'baseball', 'basketball', 'football', 'golf_ball', 'tennis_ball' 
    
    photo_to_test = 'dataset-balls/my testing/shwayet_tennis.png' 
    
    saved_model = 'ballsCNN.pth'
    
    predict_image(photo_to_test, saved_model, my_classes)