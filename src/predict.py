## This file preforms inference.  Inference means using the trained model to make predictions on new data. 
# It includes functions for loading the trained model, preprocessing the input data, and making predictions.

import numpy as np
import joblib

from tensorflow.keras.models import load_model
from preprocess import load_data, FEATURES

def predict(commodity_select = 2):

    model = load_model('models/vegetable_model.h5')

    ## Load the scaler object for the variables
    X_scaler = joblib.load('models/X_scaler.pkl')
    y_scaler = joblib.load('models/y_scaler.pkl')

    ## load the data
    df = load_data()

    COMMODITY_MAPPING = {commodity: i for i, commodity in enumerate(df['Commodity'].unique())}
    UNIT_MAPPING = {unit: i for i, unit in enumerate(df['Unit'].unique())}

    df['Commodity'] = df['Commodity'].map(COMMODITY_MAPPING)
    df['Unit'] = df['Unit'].map(UNIT_MAPPING)

    last_data = df[df['Commodity'] == commodity_select][FEATURES].tail(7).values  ## get the last 7 days of data for prediction

    ## Preprocess the input data
    X = X_scaler.transform(last_data)  ## scale the input data

    ## Reshape the input data to match the model's expected input shape
    X = X.reshape((1,
                   X.shape[0], 
                   X.shape[1]))  ## reshape to (1, 7, number_of_features) for a single sample with 7 time steps
    
    ## Make the prediction
    prediction = model.predict(X)

    ## Inverse transform the prediction to get the original scale
    prediction = y_scaler.inverse_transform(prediction)

    print(f"Predicted vegetable price for the next day: {prediction[0][0]:.2f}")

def predict_from_array(arr):
    """ 
    Model makes a prediction based on the input array.
      The input array should be a 2D array with shape (7, number_of_features), 
      representing the last 7 days of data for prediction. i.e. 7x11
    """
    model = load_model('models/vegetable_model.h5')

    ## Load the scaler object for the variables
    X_scaler = joblib.load('models/X_scaler.pkl')
    y_scaler = joblib.load('models/y_scaler.pkl')
  
    ##convert incoming data to 2D array
    arr = arr.reshape((7, -1))  ## reshape to (7, number_of_features)
    
    ## Scale the input data
    arr=  X_scaler.transform(arr)  ## scale the input data

    ## Convert back to RNN format
    arr = arr.reshape(1, 
                       7, 
                       len(FEATURES)
                    )  ## reshape to (1, 7, number_of_features) for a single sample with 7 time steps
     
    ## Make the prediction
    prediction = model.predict(arr)

    ## Convert the prediction back to the original scale
    prediction = y_scaler.inverse_transform(prediction)

    return float(prediction[0][0])  ## return the predicted bike rental count for the next day as a float

if __name__ == "__main__":
    predict()  ## call the predict function to make a prediction based on the last 7 days of data