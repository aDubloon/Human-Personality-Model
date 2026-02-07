# 🧠 Human Personality Model

A machine learning application that predicts whether someone is an **Extrovert** or **Introvert** based on behavioral survey responses. Built with scikit-learn and deployed as an interactive Streamlit web application.

## 📋 Features

- **Personality Prediction**: Classify individuals as Extrovert or Introvert
- **Interactive Web Interface**: User-friendly Streamlit application
- **Offline Model Support**: Pre-trained model serialized as pickle files for fast inference
- **Model Performance Metrics**: View accuracy, confusion matrix, and classification reports
- **Real-time Predictions**: Get instant personality predictions with confidence scores

## 🎯 Model Details

- **Algorithm**: K-Nearest Neighbors (KNN) Classifier
- **Accuracy**: 95.88%
- **Input Features**: Behavioral survey responses
- **Output**: Personality classification (Extrovert/Introvert)
- **Model Format**: Pickle files (`model.pkl`, `feature_names.pkl`)

## 📦 Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 1.5.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- numpy >= 1.24.0

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/aDubloon/Human-Personality-Model.git
cd Human-Personality-Model
```

2. **Create a virtual environment** (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Making Predictions

1. Navigate to the **"Make Prediction"** page
2. Fill in your responses to the survey questions
3. Click the **"🔮 Predict Personality"** button
4. View your personality classification and confidence score

### Viewing Model Performance

1. Navigate to the **"Model Performance"** page
2. View model accuracy, confusion matrix, and classification metrics

## 📁 Project Structure

```
Human-Personality-Model/
├── app.py                              # Streamlit web application
├── train_model.py                      # Script to train and save model
├── model.pkl                           # Pre-trained KNN classifier
├── feature_names.pkl                   # Feature names for model
├── behavioursurvey.csv                 # Training dataset
├── behaviour ml model vihaan (1).ipynb # Jupyter notebook with analysis
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🔄 Model Training

To retrain the model with new data:

```bash
python train_model.py
```

This will:
- Load the behavioral survey data
- Preprocess and encode features
- Train the KNN classifier
- Save the model as `model.pkl`
- Save feature names as `feature_names.pkl`
- Display model accuracy

## 📊 Survey Features

The model uses the following behavioral features:
- **Stage Fear**: Binary feature (Yes/No)
- **Drained After Socializing**: Binary feature (Yes/No)
- **Additional Behavioral Metrics**: Various numerical survey responses (0-10 scale)

## 🌐 Deployment

This application is optimized for offline deployment:

1. **Pickle Files**: Pre-trained model stored as `model.pkl` for fast loading without retraining
2. **Streamlit Compatible**: Can be deployed on Streamlit Cloud, Heroku, or any Python hosting
3. **Lightweight**: No additional model files or large dependencies needed beyond requirements.txt

### Deploy to Streamlit Cloud

```bash
streamlit run app.py
# Then use `streamlit login` and follow the deployment steps
```

## 📈 Performance Metrics

- **Accuracy**: 95.88%
- **Test Sample Size**: Automatically calculated from train_test_split (80-20 ratio)
- **Algorithm**: K-Nearest Neighbors with default settings

## 🛠️ Future Improvements

- Add cross-validation for better model evaluation
- Implement hyperparameter tuning
- Add more personality classification models (Random Forest, SVM, etc.)
- Create model comparison visualization
- Add data visualization of feature importance

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Vihaan**
- Email: vihaanskk@outlook.com
- GitHub: [aDubloon](https://github.com/aDubloon)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For questions or issues, please create an issue in the GitHub repository.

---

**Last Updated**: February 6, 2026
