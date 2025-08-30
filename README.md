
# Student Performance Prediction - End-to-End Machine Learning Pipeline

## 📌 Project Overview
This project is an **End-to-End Machine Learning Pipeline** designed to predict **student performance (math score)** based on various features such as gender, ethnicity, parental education, lunch, and test preparation.  
It includes **modular coding practices**, **logging**, **exception handling**, **data pipelines**, and **model training with evaluation**.

The best performing model in this project is **ElasticNet Regression**, achieving an **R² score of 0.88**.

<img width="1920" height="1321" alt="image" src="https://github.com/user-attachments/assets/01eea25b-c20f-4efd-8ce5-d96be7df74ff" />

## 🚀 Features Implemented
- ✅ Data Ingestion Pipeline (train/test split + raw storage)
- ✅ Data Transformation Pipeline (encoding categorical features + scaling numerical features)
- ✅ Model Training with multiple algorithms
- ✅ Hyperparameter Tuning
- ✅ Model Evaluation & Comparison
- ✅ Best Model Selection (ElasticNet with R² = 0.88)
- ✅ Modular Project Structure with OOP
- ✅ Logging & Exception Handling
- ✅ CI/CD Ready (can be deployed with Docker/Heroku/AWS)

---

## 🏗 Project Structure
```

End-to-End-Project-Using-Pipeline-Machinelearning/
│── .gitignore
│── README.md
│── requirements.txt
│── setup.py
│
├── artifacts/ # Stores datasets and trained models
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│
├── src/ # Core source code
│   ├── **init**.py
│   ├── logger.py # Logging module
│   ├── exception.py # Custom exception handler
│   │
│   ├── components/ # Pipeline Components
│   │   ├── data\_ingestion.py
│   │   ├── data\_transformation.py
│   │   ├── model\_trainer.py
│   │
│   ├── pipeline/ # Training & Prediction pipelines
│       ├── training\_pipeline.py
│       ├── prediction\_pipeline.py
│
├── notebook/ # Jupyter notebooks (EDA, trials, etc.)
│   ├── EDA.ipynb
│   ├── model\_trials.ipynb

````

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/qosain-bukhari/End-To-end-Project-Using-Pipeline-Machinelearning.git
cd End-To-end-Project-Using-Pipeline-Machinelearning
````

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Run the Training Pipeline

```bash
python src/pipeline/training_pipeline.py
```

### 4️⃣ Run the Prediction Pipeline

```bash
python src/pipeline/prediction_pipeline.py
```

## 📊 Model Comparison

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | 0.86     |
| Ridge Regression  | 0.87     |
| Lasso Regression  | 0.85     |
| ElasticNet        | 0.88     |

✅ **Best Model → ElasticNet Regression**

---

## 🙌 Credits

This project is inspired by **Krish Naik’s End-to-End ML tutorials** and extended with additional improvements.

## ✨ Author

👨‍💻 **Qosain Bukhari**
Data Science & Machine Learning


