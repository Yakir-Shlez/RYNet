import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import seaborn as sns
from tensorboard.plugins.hparams import api as hp

#Image paths
projectPath = r'D:'
picPath = projectPath + r'\chest_xray'
logdir = "logs/hparamas"

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

HP_LEARN_RATE = hp.HParam('learning_rate',
                          hp.Discrete([0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_LEARN_RATE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

def create_model(hparams):
    #Import a model to work with
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(img_height, img_width, 3), 
        input_tensor=tf.keras.Input(shape=(img_height, img_width, 3))
    )
    
    #Disable train for imported model
    base_model.trainable = False
    
    #Base model description print
    base_model.summary()
    
    #Adding model layers architecture
    model = keras.Sequential (
        [
             layers.Input(shape=(img_height, img_width, 3)),
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
    opt = keras.optimizers.Adam(learning_rate=hparams[HP_LEARN_RATE])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
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

for learning_rate in HP_LEARN_RATE.domain.values:
    hparams = {
        HP_LEARN_RATE: learning_rate,
        }

    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1