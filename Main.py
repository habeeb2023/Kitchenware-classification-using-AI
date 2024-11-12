import tensorflow as tf
from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.processing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# NAME = "Cats-vs-dog-cnn-64x2-15e{}".format(int(time.time()))



#gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Sessions(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#normalizing data (scaling the image)
X = X/255.0

dense_layers = [1]
layer_sizes = [64]
conv_layers = [2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = '{}-conv-{}-node-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:])) # convulusional layer, window, input shape
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3))) # convulusional layer, window, input shape
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten()) # converts 3D feature maps to 1D feature vectors

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",  #categorical
                        optimizer="adam",
                        metrics=["accuracy"])

            model.fit(X, y, batch_size=32, epochs=25, validation_split=0.3, callbacks=[tensorboard])

model.save('64x3-CNN.model')