## This file is responsible for training the neural network model using the preprocessed data.
#  It includes functions for
#   a. loading the data, 
#   b. building the model, 
#   c. training it,
#   d. saving the trained model for future use.
#   e. Evaluating the model's performance on the test set.

import numpy as np
import joblib

from preprocess import load_data, create_sequences, FEATURES, preprocess
from model import build_model

def main():
    ## Step 1: prepare the data

    X_train, X_test, y_train, y_test = preprocess()

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)

    ## Step 2: Build the model

    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
        )
    
    ## Step 3: Model summary

    model.summary()

    ## Step 4: Train the model
    model.fit(X_train, y_train,
              epochs=10, 
              batch_size=32, 
              validation_data = (X_test, y_test)
              )
    
    ## Step 5: Save the trained model for future use
    model.save('models/vegetable_model.h5')

    print("Model training completed and saved to 'models/vegetable_model.h5'.")

    ## Step 6: Make predictions on the test set and evaluate the model's performance
    predictions = model.predict(X_test)

    ## Inverse transform the predictions and actual values to get the original scale
    y_scaler = joblib.load('models/y_scaler.pkl')

    predictions = y_scaler.inverse_transform(predictions)
    y_test = y_scaler.inverse_transform(y_test)

    ## Step 7: Calculate the Mean Squared Error (MSE) to evaluate the model's performance
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f"Root Mean Squared Error (RMSE) on test set: {rmse:.2f}")

if __name__ == "__main__":
    main()