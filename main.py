from pickletools import optimize
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()


xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])  

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

#model.fit(tf.expand_dims(xTrain, axis=1), yTrain, epochs=3)
model.fit(xTrain, yTrain, epochs=1)

loss, accuracy = model.evaluate(xTest, yTest)

model.save('digits.model')

i = 1#for i in range(1,9):
img = np.array(cv.imread(f'own_numbers/{i}.png'))

#img = xTest[0]

print(img.shape)

prediction = model.predict(img)
print(f'guess its {np.argmax(prediction)}')
plt.imshow(img, cmap=plt.cm.binary)
plt.show()