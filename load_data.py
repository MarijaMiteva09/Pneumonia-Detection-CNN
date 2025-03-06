import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.RandomHorizontalFlip(),         
    transforms.RandomRotation(15),              
    transforms.ToTensor(),                     
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

data_dir = "./dataset"
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train() 

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_preds.float() / total_preds

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    torch.save(model.state_dict(), "pneumonia_model.pth")
    print("Model saved as pneumonia_model.pth")  

    print("Training complete.")

train_model(model, train_loader, criterion, optimizer, num_epochs=5)

def evaluate_model(model, test_loader):
    model.eval()  
    
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)
    
    accuracy = correct_preds.float() / total_preds
    print(f"Test Accuracy: {accuracy:.4f}")

evaluate_model(model, test_loader)

def save_gradients(grad):
    global gradients
    gradients = grad

def get_last_conv_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None

def generate_gradcam(model, image, target_class):
    model.eval()
    image.requires_grad_()
    
    image = image.unsqueeze(0).to(device)  

    last_conv_layer = get_last_conv_layer(model)
    global gradients  

    activations = None  

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output  

    def backward_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0]  

    hook_forward = last_conv_layer.register_forward_hook(forward_hook)
    hook_backward = last_conv_layer.register_backward_hook(backward_hook)
    
    output = model(image)

    model.zero_grad()
    output[0, target_class].backward()

    hook_forward.remove()  
    hook_backward.remove()

    if gradients is None:
        raise ValueError("Gradients were not captured. Make sure hooks are correctly set.")

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True) 

    weighted_feature_maps = weights * activations
    gradcam_heatmap = torch.sum(weighted_feature_maps, dim=1).relu()
    
    gradcam_heatmap = gradcam_heatmap.cpu().detach().numpy()
    gradcam_heatmap = cv2.resize(gradcam_heatmap[0], (224, 224))

    gradcam_heatmap -= np.min(gradcam_heatmap)
    gradcam_heatmap /= np.max(gradcam_heatmap)

    return gradcam_heatmap

def overlay_gradcam_on_image(image, gradcam_heatmap):
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap), cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return superimposed_img

def visualize_gradcam(model, image, label, target_class):
    gradcam_heatmap = generate_gradcam(model, image, target_class)
    superimposed_img = overlay_gradcam_on_image(image, gradcam_heatmap)
    
    plt.imshow(superimposed_img)
    plt.title(f"Predicted: {label}")
    plt.axis("off")
    plt.show()

image_batch, label_batch = next(iter(test_loader))  
image = image_batch[0] 
label = label_batch[0].item()  

class_label = test_dataset.classes[label]  

image = image.to(device)

output = model(image.unsqueeze(0))
target_class = torch.argmax(output, dim=1).item()

visualize_gradcam(model, image, class_label, target_class)
