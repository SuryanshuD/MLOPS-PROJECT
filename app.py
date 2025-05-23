import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import train_and_evaluate_models

# Set the title and description
st.set_page_config(page_title="Walmart Sales MLOps", layout="centered")
st.title("🛒 Walmart Weekly Sales Model Trainer")
st.markdown("This app compares different regression models to predict log-transformed weekly sales.")

# Button to trigger model comparison
if st.button("Run Model Comparison"):
    with st.spinner("Training models..."):
        # Call the function to train and evaluate models
        results = train_and_evaluate_models()
        st.success("Training complete!")

        # Display model results (MSE and R²)
        st.subheader("Model Comparison - Test Set")
        
        # Display the results as text
        mse_values = []
        r2_values = []
        model_names = []

        for model, (mse, r2) in results.items():
            st.write(f"**{model}**: MSE = `{mse:.4f}`, R² = `{r2:.4f}`")
            mse_values.append(mse)
            r2_values.append(r2)
            model_names.append(model)

        # Plotting the bar chart for MSE and R²
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot MSE
        ax[0].barh(model_names, mse_values, color='skyblue')
        ax[0].set_title("Mean Squared Error (MSE)")
        ax[0].set_xlabel("MSE")
        ax[0].set_ylabel("Models")

        # Plot R²
        ax[1].barh(model_names, r2_values, color='lightgreen')
        ax[1].set_title("R² Score")
        ax[1].set_xlabel("R²")
        ax[1].set_ylabel("Models")

        # Display the plots
        st.pyplot(fig)
