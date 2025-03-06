🚀 Pneumonia Detection Using Deep Learning
This project aims to detect Pneumonia from chest X-rays using a ResNet-18 model and Grad-CAM visualization for model interpretability. The app allows users to upload chest X-ray images and receive predictions on whether the image indicates Pneumonia or Normal. The app also uses Grad-CAM to visualize the regions of the image that the model is focusing on, providing an explanation for the predictions.

📌 Features
✅ Pneumonia Detection: Predicts whether a chest X-ray shows Pneumonia (1) or Normal (0).
✅ Grad-CAM Heatmap: Visualizes which parts of the image the model focused on to make predictions.
✅ Web App with Streamlit: Easy-to-use interface for uploading images and viewing results.
✅ Transfer Learning with ResNet-18: Fine-tuning of the pre-trained ResNet-18 model for accurate predictions.

📂 Files
app.py: The main Streamlit app script where users can interact with the model, upload chest X-ray images, and view predictions along with Grad-CAM heatmaps.
train.py: The script for training the ResNet-18 model using a chest X-ray dataset.
pneumonia_model.pth: The trained model file, saved after training (train.py).
dataset/: The directory containing the chest X-ray images used for training and testing the model.
requirements.txt: A file that lists all the necessary Python packages for the project.

🏗️ Installation
To run this app locally, follow these steps:

Clone the repository:
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

Install dependencies:
pip install -r requirements.txt

🎯 Running the Web App
After installation, you can run the Streamlit web app with:
streamlit run app.py
The app will be accessible at http://localhost:8501/ in your browser.

🏋️ Training the Model
To train the ResNet-18 model, use the following command:
python train.py
This will train the model on a chest X-ray dataset and save the trained model as pneumonia_model.pth.

🌍 Model Evaluation
To evaluate the model’s performance, use:
python evaluate.py
This script will display the model's accuracy on the test dataset.

🔬 Grad-CAM Visualization
To generate Grad-CAM visualizations for test images, use:
python gradcam.py
This will generate heatmaps showing the areas of the image the model focused on for its prediction.

Install them using:
pip install -r requirements.txt
🤝 Contributing
Feel free to fork the repository, create pull requests, or raise issues for improvements and bug fixes.


