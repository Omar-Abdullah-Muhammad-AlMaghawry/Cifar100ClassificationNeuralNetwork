# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense,Dropout
from keras import backend as K
from keras.engine.training import Model
class DeepLearning :
  @staticmethod
  def build(numChannels, imgRow, imgCol,numClasses, activation="relu",weightPath=None):
    #init the model
    model = Sequential();
    inputShape = (imgRow,imgCol, numChannels)
    #if we are using channel first
    if K.image_data_format() == "channels_first":
      inputShape = (numChannels, imgRow, imgCol)
    #1st layer 
    model.add(Conv2D(filters=96,kernel_size= (5,5),strides=(3,3),padding='same',input_shape=inputShape))#,strides=(2,2)
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))#,strides=(2,2)
    # model.add(Dropout(0.9))#1st try non
    #2nd layer 
    model.add(Conv2D(filters=256,kernel_size= (3,3),padding='same',input_shape=inputShape))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    # model.add(Dropout(0.6))#1st try 0.6 , tried 0.9
    #3rd layer 
    model.add(Conv2D(filters=384,kernel_size= (3,3),padding='same',input_shape=inputShape))
    model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))#,strides=(2,2)
    # model.add(Dropout(0.8))#1st try non    
    #4th layer 
    model.add(Conv2D(filters=384,kernel_size= (1,1),padding='same',input_shape=inputShape))
    model.add(Activation(activation))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))#,strides=(2,2)
    # model.add(Dropout(0.8)) #1st try non
    #5th layer 
    model.add(Conv2D(filters=256,kernel_size= (1,1),padding='same',input_shape=inputShape))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))#,strides=(2,2)
    # model.add(Dropout(0.8)) #1st try non


    #6th fully connected
    model.add(Flatten())
    model.add(Dense(5120))#5120
    model.add(Activation(activation))
    model.add(Dropout(0.3))#1st try 0.5

    #7th fully connected
    model.add(Dense(4096))
    model.add(Activation(activation))
    model.add(Dropout(0.3))#1st try 0.5

    #8th fully connected with soft max classifier
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    #load the model if it's required
    if weightPath is not None:
      model.load_weights(weightPath)
    
    return model


