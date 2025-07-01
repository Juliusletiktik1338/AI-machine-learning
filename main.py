# main.py - Conceptual Python Code for Crop Yield Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Preprocessing ---
def load_data(filepath="crop_yield_data.csv"):
    """
    Loads the crop yield dataset.
    In a real scenario, this would involve loading data from various sources
    (CSV, APIs for weather, satellite imagery, etc.) and merging them.
    For this conceptual example, we assume a single CSV.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please create a dummy CSV or provide a real one.")
        # Create a dummy DataFrame for demonstration if file not found
        data = {
            'Region': ['A', 'B', 'A', 'C', 'B'],
            'CropType': ['Wheat', 'Corn', 'Wheat', 'Rice', 'Corn'],
            'AvgTemp': [25, 28, 24, 30, 27],
            'TotalRainfall': [150, 120, 160, 200, 130],
            'SoilPH': [6.5, 7.0, 6.8, 6.0, 7.2],
            'NDVI': [0.7, 0.8, 0.75, 0.6, 0.85],
            'Yield_kg_per_hectare': [3000, 4500, 3200, 5000, 4800]
        }
        return pd.DataFrame(data)

def preprocess_data(df):
    """
    Preprocesses the raw data:
    - Handles categorical features (One-Hot Encoding).
    - Scales numerical features.
    - Splits data into features (X) and target (y).
    """
    print("\nStarting data preprocessing...")

    # Separate target variable
    X = df.drop('Yield_kg_per_hectare', axis=1)
    y = df['Yield_kg_per_hectare']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=np.number).columns

    print(f"Categorical columns: {categorical_cols.tolist()}")
    print(f"Numerical columns: {numerical_cols.tolist()}")

    # One-Hot Encode categorical features
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) # drop_first to avoid multicollinearity
        print(f"Shape after one-hot encoding: {X.shape}")

    # Scale numerical features
    scaler = StandardScaler()
    # Fit and transform only numerical columns that exist in X after one-hot encoding
    # (numerical_cols might contain columns that were dropped if they were also categorical,
    # but here we assume they are distinct)
    cols_to_scale = [col for col in numerical_cols if col in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        print(f"Numerical columns scaled: {cols_to_scale}")
    else:
        print("No numerical columns found to scale.")

    print("Data preprocessing complete.")
    return X, y, scaler # Return scaler for inverse transformation if needed

# --- 2. Model Definition (Neural Network) ---
def build_nn_model(input_dim):
    """
    Builds a simple Multi-Layer Perceptron (MLP) neural network.
    """
    model = Sequential([
        # Input layer and first hidden layer
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3), # Dropout for regularization to prevent overfitting

        # Second hidden layer
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer (single neuron for regression)
        Dense(1)
    ])

    # Compile the model
    # Adam optimizer is a good default choice for many tasks
    # Mean Squared Error (MSE) is a common loss function for regression
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("\nNeural Network model built successfully.")
    model.summary()
    return model

# --- 3. Model Training ---
def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    Trains the neural network model.
    """
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1 # Show training progress
    )
    print("Model training complete.")
    return history

# --- 4. Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    print("\nEvaluating model on test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.2f}")
    print(f"Test MAE (Mean Absolute Error): {mae:.2f}")
    return loss, mae

# --- 5. Prediction and Visualization ---
def make_predictions(model, X_new, scaler=None):
    """
    Makes predictions using the trained model.
    X_new should be preprocessed in the same way as training data.
    """
    print("\nMaking predictions...")
    predictions = model.predict(X_new).flatten()
    return predictions

def plot_training_history(history):
    """
    Plots the training and validation loss/MAE over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_test, predictions):
    """
    Plots actual vs. predicted values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield (kg/hectare)')
    plt.ylabel('Predicted Yield (kg/hectare)')
    plt.title('Actual vs. Predicted Crop Yields')
    plt.grid(True)
    plt.show()

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Load data (replace with your actual data path)
    df = load_data("crop_yield_data.csv")

    # If the dummy data was created, let's save it to a CSV for consistency
    if df.shape[0] < 10 and not pd.io.common.file_exists("crop_yield_data.csv"):
        df.to_csv("crop_yield_data.csv", index=False)
        print("Dummy data saved to 'crop_yield_data.csv'.")
        # Reload to ensure consistent loading path for demonstration
        df = load_data("crop_yield_data.csv")

    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

    # Build the neural network model
    model = build_nn_model(X_train.shape[1])

    # Train the model
    history = train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Make predictions on the test set
    predictions = make_predictions(model, X_test)

    # Visualize results
    plot_training_history(history)
    plot_predictions_vs_actual(y_test, predictions)

    print("\n--- Project Overview ---")
    print("This script demonstrates a conceptual AI solution for SDG 2 (Zero Hunger) by predicting crop yields.")
    print("It uses a Multi-Layer Perceptron (MLP) neural network, a supervised learning technique for regression.")
    print("Key steps include data preprocessing, model building, training, evaluation, and visualization.")
    print("To make this a full project, you would need:")
    print("1. Real, comprehensive agricultural and environmental datasets.")
    print("2. More sophisticated feature engineering (e.g., time-series features for weather).")
    print("3. Hyperparameter tuning for the neural network.")
    print("4. A user-friendly interface for inputting new data and displaying predictions.")
    print("5. Robust error handling and deployment considerations.")
