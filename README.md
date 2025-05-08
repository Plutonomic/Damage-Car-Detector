# 🚗 Damaged Car Image Detection and Preprocessing Pipeline

This project is an end-to-end pipeline that automates preprocessing of car images for **damage detection** using **OpenCV** and a custom **Convolutional Neural Network (CNN)**. The system helps streamline insurance claim processes, repair estimates, and AI-powered visual assessments.

## 🔧 Features

- 📸 Upload raw car images
- 🧼 Image preprocessing pipeline using OpenCV
- 🧠 Custom CNN model for damage classification
- 📊 Exploratory Data Analysis (EDA)
- 🌐 Streamlit Web App Interface

---

## 🗂 Project Structure

```

Damage_Car_Detector/
├── app.py                  # Streamlit app interface
├── cnn.py                  # CNN model definition, training, prediction
├── preprocessing.py        # Image preprocessing utilities using OpenCV
├── eda.py                  # Functions for image EDA
├── car\_damage\_model.h5     # Pre-trained CNN model (optional)
├── damaged_car_images/, not_damaged_car_images          # Folder with test car images
└── README.md

````

---

## 🖼 Sample Use Case

Upload a car image via the app. It goes through preprocessing and is then classified as:

- ✅ **Not Damaged**
- ❌ **Damaged**

This can be used in:

- 🏦 Insurance claim automation  
- 🔧 Car repair estimate apps  
- 📲 Smart garage systems

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Damage_car.git
cd Damage_car
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit tensorflow opencv-python numpy matplotlib seaborn
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 Dataset

You can train the CNN using a binary-labeled dataset with folders like:

```
dataset/
├── Damaged/
│   ├── img1.jpg
│   └── ...
└── Not_Damaged/
    ├── img2.jpg
    └── ...
```

> ✅ You can download such datasets from [Kaggle](https://www.kaggle.com/datasets) or annotate your own.

---

## 📊 Model Summary

* Input shape: `(256, 256, 3)`
* Layers:

  * 3 Convolution + MaxPooling
  * Flatten → Dense (128) → Dropout → Dense (1, Sigmoid)
* Loss: `binary_crossentropy`
* Optimizer: `Adam`

---

## 💡 Future Improvements

* 📈 Train on larger datasets
* 🏷 Use multi-class classification (minor, major, total loss)
* ☁️ Deploy via cloud (e.g., Streamlit Cloud, Heroku)
* 🔄 Integrate real-time mobile capture

---

## 🧑‍💻 Author

* Dann Berlin – [@Plutonomic](https://github.com/Plutonomic)

---

## 📄 License

This project is licensed under the MIT License.
#See the [LICENSE](LICENSE) file for details.
