🚀 Pneumonia Detection Using Deep Learning

This project detects Pneumonia from chest X-rays using a ResNet-18 model and Grad-CAM visualization. It includes a Streamlit web app for easy interaction.

📌 Features
✅ Pneumonia detection (Normal vs. Pneumonia)
✅ Grad-CAM heatmap for explainability
✅ Web app using Streamlit
✅ Transfer learning with ResNet-18

🏗️ Installation
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt

🏋️ Training the Model
python train.py
This saves the model as pneumonia_model.pth.

🎯 Evaluating the Model
python evaluate.py

🌍 Running the Web App
streamlit run app.py
Open http://localhost:8501/ in your browser.

🔬 Grad-CAM Visualization
python gradcam.py
This overlays Grad-CAM heatmaps on test images.

📜 Requirements
Python 3.8+
PyTorch, Torchvision
OpenCV, Streamlit
🤝 Contributing
Pull requests are welcome!

📜 License
MIT License

⭐ Star this repo if you find it useful! ⭐
