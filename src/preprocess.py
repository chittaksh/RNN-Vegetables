## This file is for preparing the raw data before giving it to the neural network. It includes functions for loading, cleaning, and transforming the data into a suitable format for training and evaluation.

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib  ## used for saving the scaler object for later use

## Features: Columns that are used as input to the neural network.

FEATURES = [
    "SN", 
    "Commodity", 
    # "Date", 
    "Unit", 
    "Minimum", 
    "Maximum", 
    "Average"]

COMMODITY_MAPPING = {}
UNIT_MAPPING = {}


## Load the dataset

def load_data(path = "data/kalimati_tarkari_dataset.csv"):

    df = pd.read_csv(path)

    ## Convert the date column to datetime format from string format
    df['Date'] = pd.to_datetime(df['Date'])

    ## Sort the dataframe by date to ensure that the data is in chronological order.
    df = df.sort_values(by='Date')

    global COMMODITY_MAPPING
    global UNIT_MAPPING

    COMMODITY_MAPPING = {commodity: i for i, commodity in enumerate(df['Commodity'].unique())}
    UNIT_MAPPING = {unit: i for i, unit in enumerate(df['Unit'].unique())}

    # print("Commodity Mapping:", COMMODITY_MAPPING)
    # print("Unit Mapping:", UNIT_MAPPING)

    return df


def create_sequences(X, y, window= 7):
    ## Create a sequence of data for time series prediction. This function takes the dataframe, the number of time steps to look back, and the features to be used as input. 
    ##  It returns the input sequences and the corresponding target values.
    ## We will create a sliding window of 7 days data to predict the next day's bike rental count.

    Xs = []
    ys = []

    ## Move the 7-day window across the data to create sequences
    for i in range(len(X) - window):

        ## Take 7 consecutive days of data as input features
        Xs.append(X[i:i + window])

        ## Take the next day's bike rental count as the target value
        ys.append(y[i + window])
    
    return np.array(Xs), np.array(ys)

## Preprocess the data 

def preprocess():
    df = load_data()

    ## Step 1: Convert the Commodity column to numerical values using mapping. This is necessary because neural networks can only work with numerical data.
    df['Commodity'] = df['Commodity'].map(COMMODITY_MAPPING)
    df['Unit'] = df['Unit'].map(UNIT_MAPPING)

    # print(df.head())  ## Print the first 5 rows of the dataframe to check if the data is loaded and processed correctly.

    ## Step 2: divide the dataset into training and testing sets
    ## we cannot use train_test_split here because we need to maintain the temporal order of the data.
    ##  Instead, we will split the data into training and testing sets based on a specific date.

    split = int(len(df) * 0.8)  ## Use 80% of the data for training and 20% for testing

    train_df = df[:split]
    test_df = df[split:]

    ## Step 3: Scale the features using MinMaxScaler to bring all the features into the range [0, 1].
    
    X_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(train_df[FEATURES])
    X_test = X_scaler.transform(test_df[FEATURES])

    ## Step 4: Scale the target variable (bike rental count) using MinMaxScaler.

    y_scaler = MinMaxScaler()

    y_train = y_scaler.fit_transform(train_df[['Average']])
    y_test = y_scaler.transform(test_df[['Average']])

    ## Step 5: Create sequences of data for time series prediction using the create_sequences function defined above.

    X_train, y_train = create_sequences(X_train, y_train, window=7)
    X_test, y_test = create_sequences(X_test, y_test, window=7)

    ## Step 6: Save the scalers for later use during inference.
    joblib.dump(X_scaler, 'models/X_scaler.pkl')
    joblib.dump(y_scaler, 'models/y_scaler.pkl')

    return X_train, X_test, y_train, y_test