class Config:
    def __init__(self):
        self.top_words = 20000
        self.skip_top = 20
        self.max_words = 80

        self.embedding_output_dim = 128
        self.bidir_input_dims = [64, 32, 16]
        self.dense_units = [4, 1]
        self.dropout = [0.2, 0.2, 0.2, 0.2]
        self.recurrent_dropout = [0.2, 0.2, 0.2, 0.2]

        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        self.epochs = 100
        self.batch_size = 128
        self.verbosity = 1

        self.save_filename = 'weights.h5'
        self.callback_period = 5
