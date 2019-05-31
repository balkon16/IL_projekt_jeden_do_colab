from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, \
                                    Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle

import sys
import os
import pickle

from objects_for_training import *

model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embeddings_dim, \
                    input_length=dict_to_export['max_len']))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',\
                metrics=['acc', 'mse'])
model.summary()

# prepare a path for the best model
file_name_w_ext =  os.path.basename(sys.argv[0])
file_name = file_name_w_ext.split('.')[0]
save_name_path = './saved_models/' + file_name + '.h5'

history_name_path = './histories/' + file_name + '_history.p'

# the best model (val_loss-wise) will be saved as a .h5 file
mc = ModelCheckpoint(save_name_path,
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True,
                     verbose=1)

# fit the model and save the accuracy and other metrics to a history object
history = model.fit(x=dict_to_export['X_tr_tokenized'],
          y=dict_to_export['y_tr'],
          epochs=100,
          verbose=1,
          batch_size=350,
          validation_data=(dict_to_export['X_val_tokenized'], \
                            dict_to_export['y_val']),
          class_weight=dict_to_export['class_weights'],
          callbacks=[mc])

# save history to a pickle file
history_name_path = './histories/' + file_name + '_history.p'
with open(history_name_path, 'wb') as file:
    pickle.dump(history.history, file, protocol=pickle.HIGHEST_PROTOCOL)


# check the accuracy on the test set
# the output of the model.evaluate will be stored as a text file
with open('./test_set_accuracy_outputs/' + file_name+ '.txt', 'w') as f:
    print(model.evaluate(x=dict_to_export['X_ts_tokenized'], \
                            y=dict_to_export['y_test']), file=f)
