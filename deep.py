from DeepLearning import DeepLearning
from sklearn.model_selection import train_test_split
from keras.datasets import cifar100
from keras.optimizers import gradient_descent_v2,adam_v2,Optimizer,TFOptimizer
from tensorflow.keras.optimizers import Adam  # - Works
from keras.utils import np_utils
from keras import backend as K
from keras.losses import categorical_crossentropy
import numpy as np
import argparse
import cv2
from tensorflow.python.framework.error_interpolation import interpolate
from keras import optimizers
# from google.colab.patches import cv2_imshow ############################################added for only colab
#construct args parser and parse yhe args
argP = argparse.ArgumentParser()
argP.add_argument("-s",
                  "--savemodel",
                  type=int,
                  default = -1,
                  help="(optional) whether or not model should be saved to disk")
argP.add_argument("-l",
                  "--loadmodel",
                  type=int,
                  default = -1,
                  help="(optional) whether or not model should be load from the disk")
argP.add_argument("-w",
                  "--weights",
                  type=str,
                  default="",
                  help="(optional) path to weights file")
args=vars(argP.parse_args())
print(args["weights"])

print(args["savemodel"])

print(args["loadmodel"])

#download the dataset
print("[INFO] Downloading Cifar100 ...")
(trainData,trainLabels),(testData,testLabels)= cifar100.load_data()

#reshape the data matrix depend on the channel first or last
# num_samples x depth x rows x columns or # num_samples x rows x columns x depth
print(trainData.shape)

# if K.image_data_format == "channels_first":
#   trainData.reshape(trainData.shape[0],3,32,32)
#   testData.reshape(trainData.shape[0],3,32,32)
# else:
#   trainData.reshape(trainData.shape[0],32,32,3)
#   testData.reshape(trainData.shape[0],32,32,3)

#noemalize and scale range [0,255] to [0,1]
trainData = trainData.astype("float32")/255.0
testData = testData.astype("float32")/255.0

#transform from training and testing labels to vectors [0:classes] thats for cross_entorpy loss fn
trainLabels = np_utils.to_categorical(trainLabels,100)
testLabels = np_utils.to_categorical(testLabels,100)

#initlize the optmizer and model
print("[INFO] Compiling mode ...")
opt= Adam(lr=0.0001)
# opt = GCD(0.01)
model = DeepLearning.build(3, 32, 32,100, activation="relu",
                           weightPath = args["weights"] if args["loadmodel"] > 0 else None
                           )
model.compile(optimizer = opt,loss="categorical_crossentropy",metrics=["accuracy"])

#train the model
#load the model if there pre-trained model
print(testData.shape)
print(testLabels.shape)

if args["loadmodel"]<0:
  print("[INFO] training ...")
  model.fit(trainData,trainLabels,batch_size=120,epochs=20,verbose=1)
  print(testData.shape)
  print(testLabels.shape)

  #lets see the accuracy for the test mode
  print("[INFO] testing ...")
  (loss,accuracy)= model.evaluate(testData,testLabels,batch_size=120,verbose = 1)
  print("[INFO] accuracy: {:.2f}%".format(accuracy*100))

#checck if we want save the weight
if args["savemodel"]  >0:
  print("[INFO] saving the model ...")
  model.save_weights(args["weights"],overwrite=True)

#done
names =['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
#randomly chose some pics
for i in np.random.choice(np.arange(0,len(testLabels)), size = (100,)):
  # classify the pics
  probality = model.predict(testData[np.newaxis,i])
  #the res will be vect so we give us the index of the max
  pred = probality.argmax(axis=1)

  #show the img and the res on in it
  #so oppesit every thing
  if K.image_data_format() == "channel first":
    image = (testData[i][0]*255).astype("uint8") ###
  else:
    image = (testData[i]*255).astype("uint8")

  #merge the channel into the img
  image = cv2.merge([image])
  #
  #resize the img from 32 *32 to 256*256
  image = cv2.resize(image,(256,256),interpolation = cv2.INTER_LINEAR)

  #write the result on the output img
  image = cv2.putText(image,names[pred[0]],(15,15),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0),2, cv2.LINE_AA)
  print("[INFO] predicted: {}, Actual : {} ".format(pred[0],np.argmax(testLabels[i])))
  cv2.imshow("catagory",image)
  cv2.waitKey(0)