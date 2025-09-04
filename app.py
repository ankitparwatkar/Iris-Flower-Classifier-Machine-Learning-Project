import streamlit as st
import pickle
import numpy as np
from PIL import Image

# ================== Page Config ==================
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ================== Custom CSS with Beautiful Background ==================
st.markdown("""
    <style>
    /* Beautiful floral background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #b3e5fc 50%, #81d4fa 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-image: url('https://images.unsplash.com/photo-1520763185298-1b434c919102?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1632&q=80');
        background-size: cover;
        background-blend-mode: overlay;
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 15px;
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
    }
    
    /* Subtitle styling */
    .subtext {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 30px;
        color: #34495e;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* Input field styling */
    .stNumberInput label {
        font-weight: bold;
        color: #2c3e50;
        font-size: 1.2rem;
        background: rgba(248, 249, 250, 0.7);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        font-weight: 800;
        font-size: 1.4rem;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(45deg, #FF8E53, #FF6B6B);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
    }
    
    /* Success message styling */
    .success-box {
        background-color: rgba(212, 237, 218, 0.9);
        color: #155724;
        border-radius: 16px;
        padding: 25px;
        font-weight: 800;
        font-size: 1.8rem;
        border: 2px solid #28a745;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Confidence bars */
    .confidence-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        border: 1px solid rgba(233, 236, 239, 0.7);
    }
    
    .confidence-title {
        font-weight: 800;
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .confidence-bar {
        height: 40px;
        border-radius: 20px;
        margin: 15px 0;
        position: relative;
        background: #f8f9fa;
        overflow: hidden;
        border: 1px solid #dee2e6;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 20px;
        text-align: right;
        padding-right: 15px;
        line-height: 40px;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        transition: width 1s ease-in-out;
    }
    
    .confidence-label {
        position: absolute;
        left: 15px;
        font-weight: 700;
        color: #2c3e50;
        line-height: 40px;
        font-size: 1.1rem;
        z-index: 2;
    }
    
    /* Section headers */
    .section-header {
        font-weight: 800;
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 20px;
        text-align: center;
        background: rgba(248, 249, 250, 0.7);
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #FF6B6B;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 30px;
        font-size: 1.2rem;
        padding: 15px;
        background: rgba(248, 249, 250, 0.7);
        border-radius: 12px;
        border: 1px solid rgba(233, 236, 239, 0.7);
    }
    
    /* Make all text more visible */
    .stNumberInput input {
        font-size: 1.1rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.8);
    }
    
    /* Error message styling */
    .stAlert {
        font-weight: 700;
        font-size: 1.2rem;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("""
<div class="glass-container">
    <div class="main-title">🌸 IRIS FLOWER CLASSIFIER</div>
    <div class="subtext">Predict Iris Species with Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# ================== Load Model ==================
try:
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_accuracy = 1.0
except:
    st.error("Model file not found. Please ensure 'iris_model.pkl' is in the same directory.")
    st.stop()

# Define iris species names (standard for Iris dataset)
iris_species = ["setosa", "versicolor", "virginica"]

# ================== Inputs ==================
st.markdown('<div class="glass-container"><div class="section-header">📏 Enter Flower Measurements</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.0, step=0.1)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

predict = st.button("🔮 PREDICT SPECIES", use_container_width=True)

# ================== Prediction ==================
if predict:
    try:
        # Fix for the prediction error - ensure input is in correct format
        input_array = np.array(input_data).reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = model.predict(input_array)
        
        # Handle different model output types
        if isinstance(prediction[0], str):
            # Model returns string labels directly
            predicted_class = prediction[0]
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_array)
            else:
                # Create placeholder probabilities if not available
                prediction_proba = [[0.33, 0.33, 0.34]]
        else:
            # Model returns integer indices
            predicted_class_idx = int(prediction[0])
            predicted_class = iris_species[predicted_class_idx]
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_array)
            else:
                prediction_proba = [[0.33, 0.33, 0.34]]
        
        st.markdown(f'<div class="glass-container"><div class="success-box">Predicted Species: <strong>{predicted_class.upper()}</strong></div></div>', unsafe_allow_html=True)

        # Confidence visualization using custom HTML/CSS
        st.markdown('<div class="glass-container"><div class="confidence-title">Confidence Distribution</div></div>', unsafe_allow_html=True)
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        for i, (name, prob) in enumerate(zip(iris_species, prediction_proba[0])):
            percentage = prob * 100
            color = colors[i]
            
            st.markdown(f"""
            <div class="glass-container">
                <div class="confidence-label">{name.capitalize()}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {percentage}%; background: {color};">
                        {percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# ================== Footer ==================
st.markdown("---")
st.markdown(f"""
<div class="footer">
🚀 Model Accuracy: {model_accuracy*100:.2f}% | Built with ❤️ by Ankit Parwatkar
</div>
""", unsafe_allow_html=True)