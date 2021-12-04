import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import seaborn as sns
from tensorboard.plugins.hparams import api as hp

#Image paths
projectPath = r'C:\Users\yakir\Projects\Python\Studies\Deep_learning\Project_1'
picPath = projectPath + r'\chest_xray_AIO_2'
logdir = "logs/hparamas"

#Model paramaters
img_height = 32
img_width = 32
batch_size = 64
epochs = 30

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

HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.8))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

#Model layers architecture
def create_model(hparams):
    model = keras.Sequential (
        [
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(8, kernel_size=(2, 2), activation="relu"),
            layers.BatchNormalization(),
    		layers.Flatten(),
    		layers.Dropout(hparams[HP_DROPOUT]),
    
    		# Hidden layer
    		layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
    		layers.Dropout(hparams[HP_DROPOUT]),
    				 
    		# last hidden layer i.e.. output layer
    		layers.Dense(2, activation='softmax'),
        ]
    )
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    earlyCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode = 'auto', patience=5, restore_best_weights = True)
    model.fit(train_ds, validation_data=val_ds , batch_size=batch_size, epochs=epochs, callbacks=[earlyCallback])
    loss, accuracy = model.evaluate(test_ds, batch_size=batch_size)

    return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = create_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        
session_num = 0

for dropout_rate in np.arange(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 0.1):
    hparams = {
        HP_DROPOUT: dropout_rate,
        }

    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1