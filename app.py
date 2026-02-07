import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Page configuration
st.set_page_config(page_title="Personality Prediction Model", layout="wide")

# Title
st.title("🧠 Personality Prediction Model")
st.markdown("Predict whether someone is an Extrovert or Introvert based on survey responses")

# Load model from pickle file
@st.cache_resource
def load_model_from_pickle():
    # Load pre-trained model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load feature names
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    # Load test data for performance metrics
    df = pd.read_csv("behavioursurvey.csv")
    df.dropna(inplace=True)
    df["Personality"].replace({"Extrovert": 0, "Introvert": 1}, inplace=True)
    df["Stage_fear"].replace({"No": 0, "Yes": 1}, inplace=True)
    df["Drained_after_socializing"].replace({"No": 0, "Yes": 1}, inplace=True)
    
    x = df.drop(columns=["id", "Personality"])
    y = df["Personality"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_pred, Y_test)
    
    return model, feature_names, accuracy, (X_test, Y_test, Y_pred)

model, feature_names, accuracy, test_data = load_model_from_pickle()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Make Prediction", "Model Performance"])

if page == "Make Prediction":
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survey Questions")
        
        # Create input fields for each feature
        inputs = {}
        for feature in feature_names:
            if feature in ["Stage_fear", "Drained_after_socializing"]:
                inputs[feature] = st.selectbox(
                    f"{feature.replace('_', ' ').title()}?",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=feature
                )
            else:
                # For other numerical features, create a slider
                inputs[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=0,
                    max_value=10,
                    value=5,
                    key=feature
                )
    
    with col2:
        st.subheader("Prediction Result")
        
        # Make prediction
        if st.button("🔮 Predict Personality", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame([inputs])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display result
            personality = "🎯 Extrovert" if prediction == 0 else "🎯 Introvert"
            st.success(f"Predicted Personality: {personality}")
            
            # Display confidence
            confidence = max(probability) * 100
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
            # Display probabilities
            st.write("**Probability Distribution:**")
            col_ext, col_int = st.columns(2)
            with col_ext:
                st.metric("Extrovert Probability", f"{probability[0]*100:.2f}%")
            with col_int:
                st.metric("Introvert Probability", f"{probability[1]*100:.2f}%")

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    X_test, Y_test, Y_pred = test_data
    
    # Accuracy
    col1, col2 = st.columns(2)
    with col1:
        acc = accuracy_score(Y_pred, Y_test)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    with col2:
        st.metric("Total Test Samples", len(Y_test))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Extrovert', 'Introvert'],
                yticklabels=['Extrovert', 'Introvert'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(Y_test, Y_pred, 
                                  target_names=['Extrovert', 'Introvert'],
                                  output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("This model predicts personality type based on behavioral survey responses.")
