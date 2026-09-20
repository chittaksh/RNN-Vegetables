## Simple streamlit app to predict Vegetable Price counts using a pre-trained model
## Streamlit is an open-source app framework which allows you to create interactive web applications with minimal effort.

import streamlit as st

from predict import predict, predict_from_array
from preprocess import load_data, FEATURES

## Page configuration for the Streamlit app

st.set_page_config(
    page_title="Vegetable Price Prediction",
    page_icon="&#xf06c;",
    layout="wide"
)

## Title
st.title("Vegetable Price Prediction App &#xf06c;")
### Description
st.write(
    """
    This app predicts the number of Vegetable Prices for the next day based on the last 7 days of data.
    The model is trained on historical Vegetable Price data and uses a Recurrent Neural Network (RNN) for prediction.
    You need to provide the last 7 days of data for the following features:
    - Temperature (temp)
    - Feels Like Temperature (atemp)
    - Humidity (hum)
    - Windspeed (windspeed)
    
    The app will then predict the Vegetable Price for the next day based on the provided data.
    """
)

## load the dataset to display the last 7 days of data for user reference
df = load_data()

st.session_state.commodity_select = st.session_state.get("commodity_select", df['Commodity'].unique()[0])  ## default to the first commodity in the list

## adda dropdown here to select the commodity for which the prediction is to be made
st.selectbox(
    "Select Commodity for Prediction",
    options=df['Commodity'].unique(),
    key="commodity_select"
)

## Show the last 7 days of data for user reference
st.subheader("Last 7 Days of Data")
st.dataframe(df[df['Commodity'] == st.session_state.commodity_select][FEATURES].tail(7))
print(st.session_state.commodity_select)

## Prediction section

if st.button("Predict Vegetable Prices for the Next Day"):
    latest = df[df['Commodity'] == st.session_state.commodity_select][FEATURES].tail(7).values  ## get the last 7 days of data for prediction

    ## convert into numpy array and reshape to (7, number_of_features)
    input_data = latest.reshape((7, -1))  ## reshape to (7, number_of_features)

    ## Call the trained model to make a prediction based on the last 7 days of data
    prediction = predict_from_array(input_data)

    ## Display the prediction result
    st.subheader("Predicted Vegetable Price for the Next Day")
    st.metric(
        "Predicted Vegetable Price", 
        f"{prediction:,.0f}")

## Explanation section

st.subheader("How the Prediction Works")
st.write(
    """
    The prediction is made using a Recurrent Neural Network (RNN) model that has been trained on historical Vegetable Price data.
    The model takes the last 7 days of data as input and predicts the Vegetable Price count for the next day.
    The features used for prediction include temperature, feels like temperature, humidity, and windspeed.
    The model has been trained to capture the temporal patterns in the data and make accurate predictions based on past observations.
    """
)