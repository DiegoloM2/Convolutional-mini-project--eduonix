from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
x_train = train_images.astype('float32')/255
x_test = test_images.astype('float32')/255

#remember that with covolutional models you need to reshape the data
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000, 28, 28, 1)




#create a model 
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

          
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(128))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

x = model.fit(x_train, train_labels, epochs = 3, validation_data = (x_test, test_labels), batch_size = 128)

x_evaluation = model.summary()

x_evaluation
