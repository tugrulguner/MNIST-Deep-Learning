import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv') #It works fine in kaggle notebook in this form
train = train.reindex(np.random.permutation(train.index))
label = train.label.values
train.drop(['label'], axis = 1, inplace = True)
X_train = train/255
X_train = X_train.values.reshape(len(X_train),28,28,1)

model = keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='sigmoid'),
      tf.keras.layers.Dense(64, activation='sigmoid'),
      tf.keras.layers.Dense(10),
])


model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

history = model.fit(X_train, label, validation_split = 0.1, batch_size = 30, epochs = 100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
    

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test = test/255
test = test.values.reshape(len(test),28,28,1)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test)

ids = []
for i, z in enumerate(predictions):
    ids.append(np.argmax(z))
ids = pd.DataFrame(ids, columns=['Label'])
ids['ImageId'] = ids.index+1
ids = ids[['ImageId','Label']]

ids.to_csv('/kaggle/working/predict.csv', index = False)
