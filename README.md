# imdb-lstm-bidir

An LSTM is a Long Short Term Memory, a special veriant of Recurrent Neural Networks mainly used for Language modeling and sequence predictions.
Bidirectional lstm is a network in which the data can make both both forwrd flow and backward flow at the same time.
I this experiment we stacked up bidirectional lstms one above another in three layers to obtain the sentiment expressed in a text.
'''
Test accuracy::
Conventional LSTM      : 80.92%
Deep LSTM              : 81.32%
Deep Bidirectional LSTM: 83.83%
