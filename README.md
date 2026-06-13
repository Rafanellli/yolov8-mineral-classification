# 🪨 Mineral Classification using YOLOv8

## Project Overview
This project focuses on building a deep learning model to classify different types of rock minerals using the **YOLOv8 (You Only Look Once)** image classification framework. Identifying minerals accurately is a crucial step in geosciences and mining industries, and this project automates that process using Computer Vision.

The model is trained to recognize **7 distinct mineral classes**:
* Biotite
* Bornite
* Chrysocolla
* Malachite
* Muscovite
* Pyrite
* Quartz

## 📊 Dataset & Preprocessing
The dataset consists of raw mineral images that were thoroughly preprocessed and split into three specific sets to ensure the model learns effectively without overfitting:

* **Training Set (70%):** Used to train the YOLOv8 model.
* **Validation Set (15%):** Used for hyperparameter tuning and tracking model loss during the training phase.
* **Test Set (15%):** Unseen data used for the final evaluation of the model's performance.

An automated Python script leveraging `os` and `shutil` was created to randomly shuffle and distribute the raw images into the respective directories.

## 🚀 Model Training
* **Framework:** Ultralytics YOLOv8 (`yolov8n-cls.pt`)
* **Environment:** Google Colab (GPU accelerated)
* **Epochs:** 3 (Initial proof of concept training)

## 📈 Evaluation & Performance
The model was evaluated using the unseen validation dataset. The results indicate a strong ability to generalize and extract relevant visual features from the rocks.

* **Top-1 Accuracy:** 82.1% (The model's absolute first guess is correct 82.1% of the time).
* **Top-5 Accuracy:** 99.3% (The correct mineral class is within the model's top 5 predictions 99.3% of the time).

## 🛠️ Tech Stack
* **Language:** Python 3
* **Libraries:** Ultralytics (YOLO), NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
* **Tools:** Google Colab, Google Drive (for dataset storage)

#Test result
<img width="450" height="308" alt="image" src="https://github.com/user-attachments/assets/9ae4a137-43c5-46dd-80c3-9bd381d4dc3b" />



This picture above is a demonstration of how the model can predict the mineral which is shown in the picture



## 💻 How to Use
To run this project locally or in your own Colab environment:

1. Clone this repository.
2. Install the required dependencies:
   pip install ultralytics pandas scikit-learn matplotlib seaborn
3. Download the pre-trained weights (`best.pt`) from the releases/drive link (insert your link here).
4. Run inference on a new image:
   from ultralytics import YOLO
   model = YOLO('path/to/best.pt')
   results = model.predict('path/to/your/image.jpg')
   results[0].show()

## 📌 Future Improvements
* Increase the number of training epochs (e.g., 50 or 100) to further reduce training loss and improve Top-1 accuracy.
* Implement data augmentation techniques (rotation, flipping, brightness adjustment) to make the model more robust against different lighting conditions and camera angles.
