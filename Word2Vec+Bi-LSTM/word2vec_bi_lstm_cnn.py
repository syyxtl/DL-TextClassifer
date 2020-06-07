import tensorflow as tf
import numpy as np


class WBLSTMCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_sizes, wordEmbedding, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer #Embedding 层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W = tf.Variable( tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)   


        # Create a convolution + maxpool layer for each filter size #卷积层 池化层
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(hidden_sizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, self.embedded_chars, dtype=tf.float32, scope="bi-lstm" + str(idx))        
                    self.embedded_chars = tf.concat(outputs, 2)

        pooled_outputs = self.embedded_chars[:, 0, :]
        num_filters_total = hidden_sizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        self.h_pool_flat = tf.reshape(pooled_outputs, [-1, num_filters_total])  # reshape成全连接层的输入维度

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions #最后的BP层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
