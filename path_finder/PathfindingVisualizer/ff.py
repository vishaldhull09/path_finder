import os
os.environ['KERAS_BACKEND'] ="tensorflow"



import keras
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




from keras.models import Sequential


import numpy as np
import tensorflow as tf
np.random.seed(2)




if tf.test.is_built_with_cuda():
      print("The installed version of TensorFlow includes GPU support.")
else:
      print("The installed version of TensorFlow does not include GPU support.")








import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  Convolution2D, MaxPooling2D, BatchNormalization,UpSampling2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt





import os
os.chdir('/home/akashdeep/Desktop/')
get_ipython().system('pwd')




get_ipython().system('ls')

import globalvars
 




import tensorflow as tf




from keras.callbacks import TensorBoard,  ModelCheckpoint,EarlyStopping





model=Sequential()
from keras.layers import LeakyReLU

import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform



import pickle


def data_create():
    import argparse                      
    import pickle

    args_file = 'important'    
    args = pickle.load(open(args_file, 'rb'))
    (x,y) = args[0],args[1]
    x = np.array(x,dtype=np.float64)
    y = np.array(y,dtype=np.float64)
    x /= 255
    y /= 255
    x_train = x[:14163]
    y_train = y[:14163]
    x_test = x[14163:]
    y_test = y[14163:]
    y_test = np.reshape(y_test, (-1, 256, 256, 1))
    x_test = np.reshape(x_test, (-1, 256, 256, 1))
    x_train = np.reshape(x_train, (-1, 256, 256, 1))
    y_train = np.reshape(y_train, (-1, 256, 256, 1))
    return x_train, y_train, x_test, y_test


from keras.callbacks import TensorBoard,  ModelCheckpoint,EarlyStopping
import keras.callbacks as callbacks
import time




def create_model(x_train, y_train, x_test, y_test):
   
   
    model=Sequential()
    model.add(Convolution2D({{choice([8,16,32,64])}}, kernel_size = {{choice([(3,3),(5,5)])}}, padding='same', input_shape=(256, 256, 1)) )
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0,0.6)}}))
    model.add({{choice([MaxPooling2D(pool_size=(2,2),strides=None,padding='valid',data_format=None)])}})


    model.add(Convolution2D({{choice([8,16,32,64])}}, kernel_size={{choice([(3,3),(5,5)])}},padding='same'))
    model.add(BatchNormalization())        
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0,0.6)}}))
    model.add({{choice([MaxPooling2D(pool_size=(2,2),strides=None,padding='valid',data_format=None)])}})



    model.add(Convolution2D({{choice([8,16,32,64])}}, kernel_size = {{choice([(3,3),(5,5)])}},padding='same'))
    model.add(BatchNormalization())      
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0,0.7)}}))

    model.add(Convolution2D({{choice([8,16,32,64])}}, kernel_size = {{choice([(3,3),(5,5)])}},padding='same'))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0,0.6)}}))
    model.add({{choice([UpSampling2D((2,2))])}})


    model.add(Convolution2D({{choice([8,16,32,64])}}, kernel_size = {{choice([(3,3),(5,5)])}},padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add({{choice([UpSampling2D((2,2))])}})

    model.add(Convolution2D(1, kernel_size = {{choice([(3,3),(5,5)])}},padding='same',activation={{choice(['relu', 'sigmoid'])}}))
    from keras.utils import multi_gpu_model
    model2 = multi_gpu_model(model, gpus=2)
    model2.compile(loss='mean_squared_error', metrics=['accuracy'],
                   optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    globalvars.globalVar += 1
    
    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))
    filepath = "weights_mlp_hyperas" + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    result = model2.fit(x_train, y_train,
                        batch_size={{choice([32,64, 128])}},
                        epochs=200,
                        verbose=2,
                        validation_split=0.1,callbacks=[checkpoint,tensorboard])

    import numpy

    h1 = result.history
    acc_ = numpy.asarray(h1['acc'])
    loss_ = numpy.asarray(h1['loss'])
    val_loss_ = numpy.asarray(h1['val_loss'])
    val_acc_ = numpy.asarray(h1['val_acc'])
 
    acc_and_loss = numpy.column_stack((acc_, loss_, val_acc_, val_loss_))
    save_file_mlp = 'mlp_run_' + '_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_mlp, 'w') as f:
            numpy.savetxt(save_file_mlp, acc_and_loss, delimiter=" ")

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Final validation accuracy:', acc)
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model2}

   

   


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                      data=data_create,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials(),
                                     )
    X_train, Y_train, X_test, Y_test = data_create()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
