from pickle import NONE
from pickletools import optimize
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg") # set the backend  


mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])  

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

model.fit(xTrain, yTrain, epochs=25)

loss, accuracy = model.evaluate(xTest, yTest)

model.save('digits.model')
'''
model = tf.keras.models.load_model('digits.model')

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

for i in range(10):

    img = np.invert(np.array([((cv.imread(f'own_numbers/{i}.png')))[:,:,0]]))#reading inverting and making it an array. more shit idk what do
    imgForGraphics = cv.imread(f'own_numbers/{i}.png')#only reading
    prediction = model.predict(img)
    print(f'guess its {np.argmax(prediction)}')

    #plotin
    plt.imshow(imgForGraphics, cmap=plt.cm.binary)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+100+100")
    plt.show()
    plt.bar(np.arange(len(prediction.ravel())), prediction.ravel())
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+100+100")
    plt.show()