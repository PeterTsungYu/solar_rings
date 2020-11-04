import numpy as np
import re
from tensorflow import concat, split, math, square, constant
from tensorflow.keras.backend import squeeze, reshape
from tensorflow.keras import Input, models, optimizers, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, BatchNormalization, MaxPooling3D, TimeDistributed, Flatten, Dense, Concatenate, Multiply, Add 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint


def TimeCNN(inputs, hidden=4, multi=2):
    # data=[pixel_w, pixel_h, channels]
    # hidden: total number of hidden layers
        ## at least 2 hidden layers # even number
    # multi: times of the filter
    
    port = re.match(r"[a-z]+", inputs.name).group(0)
    #print(port)
    
    conv_args = {
        "kernel_size":3,
        "activation": "tanh", # for -1 to 1 coef
        "kernel_initializer": "glorot_uniform",
        "padding": "same", # for element-wise featmap
        "data_format": "channels_first"
    }
    
    filters = inputs.shape[2]*multi
    x = TimeDistributed(Conv2D(filters=filters, **conv_args), name=f'{port}CNN_1')(inputs)

    for i in range(hidden-2):
        filters = filters*multi
        x = TimeDistributed(Conv2D(filters=filters, **conv_args), name=f'{port}CNN_{i+2}')(x)
            
    outputs = TimeDistributed(Conv2D(filters=1, **conv_args), name=f'{port}_FeatMap')(x)
    
    return outputs


def ConvLSTM(inputs, multi=2):
    
    lstm_args = {
        "kernel_size":3,
        "stateful":False, # for decoder and encoder
        "activation":"tanh", # for -1 to 1 coef
        "recurrent_activation":"hard_sigmoid",
        "kernel_initializer":"glorot_uniform",
        "padding":"same", # for element-wise featmap
        "data_format": "channels_first"
    }
    
    filters = inputs.shape[2]*multi
    x = ConvLSTM2D(filters=filters, return_sequences=True, **lstm_args, name='coef_LSTM_1')(inputs)
    
    filters = filters*multi
    outputs = ConvLSTM2D(filters=filters, return_sequences=True, **lstm_args, name='coef_LSTM_2')(x)
    
    return outputs


def factor(inputs, raw, port_len):
    
    lstm_args = {
        "kernel_size":3,
        "stateful":False,
        "activation":"tanh", # for -1 to 1 coef
        "recurrent_activation":"hard_sigmoid",
        "kernel_initializer":"glorot_uniform",
        "padding":"same", # for element-wise featmap
        "data_format": "channels_first"
    }
    
    conv_args = {
        "kernel_size":3,
        "activation": "sigmoid", # for 0 to 1 coef
        "kernel_initializer": "glorot_uniform",
        "padding": "same", # for element-wise featmap
        "data_format": "channels_first"
    }
    
    target = re.match(r"[a-z]+", raw.name).group(0)
    
    x = ConvLSTM2D(filters=port_len, return_sequences=False, **lstm_args, name=f'{target}_port_FeatMap')(inputs)
    maps = split(x, num_or_size_splits=port_len, axis=1) # split axis: channel
    coef = Add(name=f'{target}_Addcoef')(maps)
    coef = Conv2D(filters=1, **conv_args, name=f'{target}_coef_CNN')(coef)
    #coef = squeeze(coef, axis=1)
    outputs = Multiply(name=f'{target}_ground_pred')([coef, raw])
    
    return outputs

def input_window(data, timesteps=3, future=1, stride=1):
    x = []
    i = 0
    while (i + timesteps + future) <= len(data):
        x.append(data[i:i + timesteps])
        i += stride
        
    x = np.array(x)
    
    return x


def predict_window(data, timesteps=3, future=1, stride=1):
    y = []
    i = timesteps
    while (i + future) <= len(data):
        y.append(data[i:(i + future)])
        i += stride
        
    y = np.array(y)
    
    return y