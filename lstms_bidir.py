import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#from keras.utils import plot_model

import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words,skip_top=20)

max_words = 80
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print('Build model...')
model = Sequential()
model.add(Embedding(top_words, 128))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
#plot_model(model, to_file='/output/model.png')

# Fit the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, verbose=2, validation_data=(X_test, y_test))
#model.save_weights('weights.h5')
# Final evaluation of the model
score, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
savefig('acc.png')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
savefig('loss.png')
#plt.show()
