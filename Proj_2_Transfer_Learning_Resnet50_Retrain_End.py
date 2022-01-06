import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import seaborn as sns

#Image paths
projectPath = r'D:'
picPath = projectPath + r'\chest_xray'

#Model paramaters
img_height = 32
img_width = 32
batch_size = 64
epochs = 40

#Train loading
train_ds = tf.keras.utils.image_dataset_from_directory(
  picPath + r'\train',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  subset = 'training',
  validation_split = 0.2,
  label_mode='categorical')

#Validation loading
val_ds = tf.keras.utils.image_dataset_from_directory(
  picPath + r'\train',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  subset = 'validation',
  validation_split = 0.2,
  label_mode='categorical')

#Test loading
test_ds = tf.keras.utils.image_dataset_from_directory(
  picPath + r'\test',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=1000,
  label_mode='categorical',
  shuffle = False)

#Categories
class_names = train_ds.class_names
print(class_names)

#Model layers architecture
base_model = tf.keras.applications.resnet50.ResNet50(
    include_top=False, 
    weights='imagenet', 
    input_shape=(img_height, img_width, 3), 
    input_tensor=tf.keras.Input(shape=(img_height, img_width, 3))
)

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:168]:
  layer.trainable =  False

#Base model description print
base_model.summary()

#Adding model layers architecture
model = keras.Sequential (
    [
         layers.Input(shape=(img_height, img_width, 3)),
         layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
         base_model,
         layers.Flatten(),
         layers.Dropout(0.2),
         layers.Dense(64, activation='relu'),
    	 layers.Dropout(0.2),
         
    	 # last hidden layer i.e.. output layer
    	 layers.Dense(2, activation='softmax'),
    ]
)

#Model compilation
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Final model description print
model.summary()

#Log file for tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Model training with train and validation
earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode = 'auto', patience=5, restore_best_weights = True)
r = model.fit(train_ds, validation_data=val_ds , batch_size=batch_size, epochs=epochs, callbacks=[earlyCallback, tensorboard_callback])

#Model evaluate calculate
results = model.evaluate(test_ds, batch_size=batch_size)
print("test loss, test acc:", results)

#Predictions calculate
predictions = model.predict(test_ds)
predictions = np.argmax(predictions, axis=1)

#Confusion calculate and graph
test_y = next(test_ds.as_numpy_iterator())[1]
test_y = np.argmax(test_y, axis = 1)

confusion = tf.math.confusion_matrix(predictions, test_y)
print(confusion)
ax= plt.subplot()
sns.heatmap(confusion, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['NORMAL', 'PNEUMONIA']);
ax.yaxis.set_ticklabels(['NORMAL', 'PNEUMONIA']);


#Precision and Recall calculations
print("Precision: " + str(confusion.numpy()[0,0] / (confusion.numpy()[0,0] + confusion.numpy()[0,1])));
print("Recall: " + str(confusion.numpy()[0,0] / (confusion.numpy()[0,0] + confusion.numpy()[1,0])));