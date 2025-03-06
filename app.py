import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  
model.fc = nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))  
model.to(device)
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return "🟢 Normal" if pred.item() == 0 else "🔴 Pneumonia"

def get_last_conv_layer(model):
    for name, module in reversed(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            return module
    return None

def generate_gradcam(image):
    image.requires_grad_()
    image = image.unsqueeze(0).to(device)

    last_conv_layer = get_last_conv_layer(model)
    gradients, activations = None, None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    hook_forward = last_conv_layer.register_forward_hook(forward_hook)
    hook_backward = last_conv_layer.register_backward_hook(backward_hook)

    output = model(image)
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    hook_forward.remove()
    hook_backward.remove()

    if gradients is None or activations is None:
        st.error("❌ Gradients not captured! Check model hooks.")
        return None

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    gradcam_heatmap = torch.sum(weights * activations, dim=1).relu()

    gradcam_heatmap = gradcam_heatmap.cpu().detach().numpy()[0]
    gradcam_heatmap = cv2.resize(gradcam_heatmap, (224, 224))
    gradcam_heatmap = (gradcam_heatmap - np.min(gradcam_heatmap)) / (np.max(gradcam_heatmap) + 1e-8)

    return gradcam_heatmap

def overlay_gradcam(image, heatmap):
    image = np.array(image.resize((224, 224)))  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)  

    return superimposed_img

st.title("🩺 Pneumonia Detection Web App")
st.write("Upload a **chest X-ray** image, and the model will predict whether it is **Normal or Pneumonia**.")

uploaded_file = st.file_uploader("📤 Upload Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    prediction = predict(image)
    st.markdown(f"## 🏥 **Diagnosis:** {prediction}")

    image_tensor = transform(image)
    heatmap = generate_gradcam(image_tensor)

    if heatmap is not None:
        heatmap_img = overlay_gradcam(image, heatmap)
        st.image(heatmap_img, caption="🔥 Grad-CAM Activation", use_column_width=True)
