from pickletools import optimize
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4)
])  

model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(xTrain, yTrain, epochs=3)

loss, accuracy = model.evaluate(xTest, yTest)

model.save('digits.model')

#for i in range(1,9):
#print(i)
img = cv.imread(f'own_numbers/1.png')
img = np.invert(np.array(img))
prediction = model.predict(img)
print(f'guess its {np.argmax(prediction)}')
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()

