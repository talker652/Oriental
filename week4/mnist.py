import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255  
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
x_train = tf.stack((x_train, x_train, x_train), axis = -1)
x_test =  tf.stack((x_test, x_test, x_test), axis = -1)

x_test = x_train[-2000:,...]
y_test = y_train[-2000:]

from tensorflow.keras import datasets, layers, models, losses, Model
base_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (32,32,3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)  
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x) 
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(10, activation = 'softmax')(x)
head_model = Model(inputs = base_model.input, outputs = predictions)

head_model.compile(optimizer = 'adam', loss = losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = head_model.fit(x_train, y_train, batch_size = 64, epochs = 10, validation_data = (x_test, y_test))