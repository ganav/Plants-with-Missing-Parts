
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer

def res_block(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model) 
    model = add([gen, model])
    return model


def cnn(shape,n_class):
    #record gradient between F_k and Y_c
    
    inp = Input(shape = shape)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "valid")(inp)
    model = Conv2D(filters = 64, kernel_size = 4, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 4, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "valid")(model)

    for index in range(3):
        model = res_block(model, 3, 128, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    A_k = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = "valid",name='last_conv')(model)
           
    F_k = GlobalAveragePooling2D(name='gap')(A_k)

    dense_layer = Dense(n_class,name='last_dense')
    Y_c=dense_layer(F_k)
    model = Activation('softmax')(Y_c)

    model2 = Model(inputs = inp, outputs = model)
    optimizer = Adam(learning_rate=0.00001, decay=0.000001)
    model2.compile(loss=['categorical_crossentropy'],
        optimizer=optimizer,metrics=['accuracy'])
    return model2
