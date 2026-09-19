## This file contains the code for building and training the neural network model for bike rental prediction.
## we are using a simpleRNN.
## RNN: Recurrent Neural Network is a type of neural network that is well-suited for sequential data, such as time series data. It has a memory component that allows it to retain information from previous time steps, making it effective for tasks like predicting future values based on past observations.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.losses import MeanSquaredError


def build_model(input_shape):
    ## Build a simple RNN model for time series prediction. The model takes the input shape of the data and returns a compiled model.
    ## The model consists of a SimpleRNN layer followed by a Dense layer for output. 
    # #The loss function used is Mean Squared Error (MSE).
    
    model = Sequential()  ## it means that layers are stacked sequentially, one after the other. Each layer has weights that are learned during training.
    ## first RNN layer:
        ## 64 units: The number of neurons in the RNN layer. More units can capture more complex patterns but may lead to overfitting.
        ## activation='relu': The activation function used in the RNN layer. ReLU (Rectified Linear Unit) is a common choice for activation functions in neural networks.
        ## input_shape=input_shape: The shape of the input data. It should match the shape  of the input sequences created during preprocessing. The input shape is typically (window_size, number_of_features).
   
    model.add(SimpleRNN(64, return_sequences= True, input_shape=input_shape))

    ## Second RNN layer:
        ## 32 units: The number of neurons in the second RNN layer. It is smaller than the first layer, which can help in reducing overfitting and capturing more abstract features.

    model.add(SimpleRNN(32))

    ## Output layer:
        ## Dense(1): A fully connected layer with a single neuron. This layer produces the final output of the model, which is the predicted bike rental count for the next day.
    model.add(Dense(1))

    ## Compile the model with Mean Squared Error loss and Adam optimizer.
    model.compile(loss=MeanSquaredError(), optimizer='adam')

    return model