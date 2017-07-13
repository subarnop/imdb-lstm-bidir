import os
import numpy
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

# Import configuration
import config
C = config.Config()

# Fix random seed for reproducibility
seed = 1338
numpy.random.seed(seed)

# Load the dataset but only keep the top n words, zero the rest
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=C.top_words, skip_top=C.skip_top)

max_words = C.max_words
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print('Building sequential model..')
model = Sequential()
model.add(Embedding(C.top_words, C.embedding_output_dim))
model.add(Bidirectional(LSTM(
    C.embedding_output_dim, dropout=C.dropout[0], recurrent_dropout=C.recurrent_dropout[0], return_sequences=True)))
model.add(Bidirectional(LSTM(
    C.bidir_input_dims[0], dropout=C.dropout[1], recurrent_dropout=C.recurrent_dropout[1], return_sequences=True)))
model.add(Bidirectional(LSTM(
    C.bidir_input_dims[1], dropout=C.dropout[2], recurrent_dropout=C.recurrent_dropout[2], return_sequences=True)))
model.add(Bidirectional(LSTM(
    C.bidir_input_dims[2], dropout=C.dropout[3], recurrent_dropout=C.recurrent_dropout[3])))
model.add(Dense(C.dense_units[0], activation='relu'))
model.add(Dense(C.dense_units[1], activation='sigmoid'))

# Try using different optimizers and different optimizer configs
model.compile(loss=C.loss,
              optimizer=C.optimizer,
              metrics=[C.metrics[0]])
print(model.summary())

if not os.path.exists(C.save_filename):
    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        C.save_filename,
        monitor='val_acc',
        verbose=C.verbosity,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=C.callback_period)

    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=C.epochs, batch_size=C.batch_size, verbose=C.verbosity,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint])

    # List all data in history
    print(history.history.keys())

    # Plot history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    savefig('acc.png')

    # Plot history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    savefig('loss.png')
else:
    # Load previously saved weights and evaluate the model
    model.load_weights(C.save_filename)

    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)
