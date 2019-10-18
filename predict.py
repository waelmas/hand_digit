import tensorflow as tf
import matplotlib.pyplot as pyplot
from tensorflow.keras.models import load_model


#for testing on an image from minst dataset
print()
print()
print()
test_img=input("Pick a number from 1 to 10000:  ")

image_index = int(test_img)
img_rows=28
img_cols=28

# load model
model = load_model('model.h5')

#load minst dataset to pick an image for prediction
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')

#show the image trying to predict
pyplot.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pyplot.show()
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))

print("Model summary")
model.summary()

print("Predicted output:")
print(pred.argmax())

