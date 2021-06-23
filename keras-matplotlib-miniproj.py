
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as  pd

#inp0
#take the train and test images you want to train the data to learn
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images.shape

scaled_train_img = train_images/255
scaled_test_img = test_images/255

#create a model 

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)),
                        keras.layers.Dense(128, activation = 'relu'),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(10, activation = 'softmax')])
#the output is a unit from 0 to 10

model.compile(optimizer = 'Adadelta',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'],)

x = model.fit(scaled_train_img, train_labels, epochs = 5, validation_data = (scaled_test_img, test_labels))
#epochs tells the program how many times the training has to run

#inp1
x_history = pd.DataFrame(x.history)
plt.figure()
plt.subplot(2,2,1)
plt.plot(x_history['loss'])
plt.plot(x_history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()

#inp2
plt.figure()
plt.subplot(222)
plt.plot(x_history['accuracy'])
plt.plot(x_history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()

#inp3
x_history = pd.DataFrame(x.history)
plt.figure()
plt.subplot(2,2,1)
plt.plot(x_history['loss'])
plt.plot(x_history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()

#inp4
plt.figure()
plt.subplot(222)
plt.plot(x_history['accuracy'])
plt.plot(x_history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()


#inp5



