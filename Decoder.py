import tensorflow as tf
import tensorflow.contrib.slim as slim


class Decoder:
    def __init__(self, net, weight_decay=0.0, dropout_keep_prob=0.8, batchnorm=True,
                 batchnorm_decay=0.999, is_training=True, scope='Decoder'):
        """

        :param net: is the input network. A tensor of size [

        :param weight_decay: is the L2 regularization strength hyperparameter

        :param dropout_keep_prob: the probability of keeping neurons for dropout layer

        :param batchnorm: a boolean to enable or disable batch normalization

        :param batchnorm_decay: a float number to represent the decay rate for the moving average of batch normalization

        :param is_training: a boolean used for both [batchnorm and dropout] layers which should be true during training and false during testing

        :param scope: is the name of the variable scope of the network

        """
        # L2 Regularization
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        # Glorot initializer is the default initializer
        initializer = tf.contrib.layers.xavier_initializer()
        # Batch normalization is used if enabled
        normalizer_fn = slim.batch_norm if batchnorm else None
        normalizer_params = {'is_training': is_training, 'center': True, 'scale': True,
                             'decay': batchnorm_decay} if batchnorm else None

        with tf.variable_scope(scope):
            # Create an arg scope for the fully connected layers
            with slim.arg_scope([slim.fully_connected], weights_initializer=initializer,
                                weights_regularizer=regularizer, normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params):
                # Create an arg scope for batch_norm layer
                with slim.arg_scope([slim.batch_norm]):
                    # Create an arg scope for the dropout layer
                    with slim.arg_scope([slim.dropout], is_training=is_training,
                                        keep_prob=dropout_keep_prob):

                        self.input = net

                        self.end_point = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu)
                        self.end_point = slim.dropout(self.end_point)

                        self.end_point = slim.fully_connected(self.end_point, 512, activation_fn=tf.nn.leaky_relu)
                        self.end_point = slim.dropout(self.end_point)

                        self.end_point = slim.fully_connected(self.end_point, 512, activation_fn=tf.nn.leaky_relu)

                        self.mid_point_1 = tf.concat([self.end_point, self.input], 2)

                        self.end_point = slim.fully_connected(self.mid_point_1, 512, activation_fn=tf.nn.tanh)
                        self.end_point = slim.dropout(self.end_point)

                        self.end_point = slim.fully_connected(self.end_point, 512, activation_fn=tf.nn.leaky_relu)
                        self.end_point = slim.dropout(self.end_point)

                        self.end_point = slim.fully_connected(self.end_point, 512, activation_fn=tf.nn.tanh)

                        self.mid_point_2 = tf.concat([self.end_point, self.mid_point_1], 2)

                        self.end_point = slim.fully_connected(self.mid_point_2, 512, activation_fn=tf.nn.leaky_relu)
                        self.end_point = slim.dropout(self.end_point)

                        self.end_point = slim.fully_connected(self.end_point, 1, activation_fn=tf.nn.tanh)


# Test drive for decoder
if __name__ == '__main__':
    tf.reset_default_graph()
    # Creating a dummy placeholder
    x = tf.placeholder(tf.float32, [None, 4000, 2051])  # 2051 latent variables + 3 coordinates
    # Class instantiation
    print("Building Decoder...")
    decoder = Decoder(x)
    # Success message
    print("Decoder is built successfully.")
    print(decoder.mid_point_1.shape, decoder.mid_point_2.shape, decoder.end_point.shape)

