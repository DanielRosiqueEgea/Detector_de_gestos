
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg19 import VGG19
from keras.models import Model
# from keras.preprocessing import image
from tensorflow.keras.layers import Input, Lambda ,Dense ,Flatten ,Dropout
# import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 as cv
from utils_file import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

train_dir = "test_accuracy"

class_indices, index_to_class = get_class_indices()
progbar = True
images, labels = load_images(directory = train_dir,uniq_labels=class_indices.keys(),progbar=progbar)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)

n = len(class_indices.keys())
train_n = len(x_train)
test_n = len(x_test)

print("Total number of gestures: ", n)
print("Number of training images: " , train_n)
print("Number of testing images: ", test_n)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


mostrar_ejemplos_matplotlib(x_train, y_train, index_to_class)


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
# X_eval = X_eval.astype('float32')/255.0

# print("Se utiliza el modelo VGG16:")
# vgg16 = VGG16(input_shape= (64,64,3),include_top=False,weights='imagenet')

# #no entrenar de nuevo el modelo
# for layer in vgg16.layers:
#     layer.trainable = False


# model = vgg16.output#head mode
# model = Flatten()(model)#adding layer of flatten
# model = Dense(units=256, activation='relu')(model)
# model = Dropout(0.6)(model)
# model = Dense(units=n, activation='softmax')(model)

# print("Se compila el modelo:")
# model = Model(inputs = vgg16.input , outputs = model)
model = load_model("modelos/model_vgg16.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print("Se grafica el modelo:")
# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )

print("Sentencia de entrenamiento:")
history = model.fit(x_train, y_train, epochs =5, batch_size = 64,validation_data=(x_test,y_test))

model_path = "modelos/"
model_name = 'model_vgg16_with_my_data.h5'
model.save(os.path.join(model_path, f"{model_name}"), save_format="h5")
print("Modelo guardado como: ", model_path)

score = model.evaluate(x = x_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(score[1]*100, 3), '%')

plot_history(history)

y_test_pred = model.predict(x_test,batch_size=64,verbose=0)
cm = confusion_matrix(y_test, y_test_pred, labels=list(class_indices.keys()))
mostrar_matriz_confusion(cm)
