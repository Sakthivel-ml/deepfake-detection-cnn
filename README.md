# Deepfake Detection using Basic CNN

A simple **CNN-based deepfake detection** model using **TensorFlow/Keras**.  
The model classifies face images as **real or fake**.

## 📌 Features
✔ Basic CNN with **3 convolutional layers**  
✔ Classifies images as **Real** or **Fake**  
✔ Uses **Binary Crossentropy Loss & Adam Optimizer**  
✔ Supports **image uploads for prediction**  

## 🚀 Installation & Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/Sakthivel-ml/deepfake-detection-cnn.git
   cd deepfake-detection-cnn
## 📊 Model Training
Run the following script to train the CNN model:
  ```python deepfake_cnn.py```

## 🔍 Prediction on New Images
Use the test script to classify an image as real or fake:
  ```python test_cnn.py --image test_image.jpg```

## 📂 Dataset
You need to place your real and fake images in the dataset/ folder.
Ensure the dataset is structured as:
 ``` dataset/
├── real/  (Real face images)
├── fake/  (Deepfake images)
```

## 🔧 Dependencies
TensorFlow
NumPy
OpenCV
Matplotlib

## 📜 License
This project is licensed under the MIT License.

---
```git add .
git commit -m "Added CNN deepfake detection project"
git push origin main
```



