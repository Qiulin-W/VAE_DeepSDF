import tensorflow.contrib.slim as slim
import tensorflow as tf


@slim.add_arg_scope
def inverted_residual_block(net, in_filters, out_filters, expansion_factor, stride, kernel_size=(3, 3)):
    """
    Inverted Residual Block specified in MobileNet-V2 paper
    ----
    Args:
    ----
    :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

    :param in_filters: number of input feature map filters

    :param out_filters: number of output feature map filters

    :param expansion_factor: multiplier to increase number of filters (proposed in the paper)

    :param stride: output stride for the depthwise convolution

    :param kernel_size: is the height and width of the convolution kernel.

    ----
    Return:
    ----
    Returns a block that consists of [conv2d, depthwise_conv2d, conv2d] with residual connection if input stride equals 1 or without if input stride equals 2
    """
    if stride not in [1, 2]:
        raise ValueError("Strides should be either 1 or 2")

    res = slim.conv2d(net, in_filters * expansion_factor, kernel_size, stride=1,
                      activation_fn=tf.nn.leaky_relu)

    res = slim.separable_conv2d(res, None, kernel_size, 1, stride=stride, activation_fn=tf.nn.leaky_relu)

    res = slim.conv2d(res, out_filters, kernel_size, stride=1, activation_fn=None)

    if stride == 2:
        return res
    else:
        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if in_filters != out_filters:
            net = slim.conv2d(net, out_filters, stride=1, kernel_size=1, activation_fn=None)
        return tf.add(res, net)


@slim.add_arg_scope
def inverted_residual_block_sequence(net, in_filters, out_filters, num_units: int, expansion_factor=6, initial_stride=2,
                                     kernel_size=3):
    """
    A group of inverted residual blocks
    ----
    Args:
    ----
    :param net: is the input network. A tensor of size [batch, height, width, channels] or [batch, channels, height, width] depending on the data format [NHWC or NCHW]

    :param in_filters: number of input feature map filters

    :param out_filters: number of output feature map filters

    :param num_units: is the number of blocks in the sequence

    :param expansion_factor: multiplier to increase number of filters (proposed in the paper)

    :param initial_stride: output stride for the depthwise convolution

    :param kernel_size: is the height and width of the convolution kernel.

    ----
    Return:
    ----
    Returns a sequence of blocks that consists of [conv2d, depthwise_conv2d, conv2d] with residual connection if input stride equals 1 or without if input stride equals 2
    """
    net = inverted_residual_block(net, in_filters, out_filters, expansion_factor, initial_stride, kernel_size)

    for i in range(num_units - 1):
        net = inverted_residual_block(net, in_filters, out_filters, expansion_factor, 1, kernel_size)

    return net


###################################################################################################################
def create_arg_scope(weight_decay=0.0, dropout_keep_prob=0.8, batchnorm=True, batchnorm_decay=0.999,
                     is_training=True, data_format='NCHW'):
    """
    This creates an argument scope to pass the same parameters for the same layers to avoid having duplicate code
    ----
    Args:
    ----
    :param weight_decay: is the L2 regularization strength hyperparameter

    :param dropout_keep_prob: the probability of keeping neurons for dropout layer

    :param batchnorm: a boolean to enable or disable batch normalization

    :param batchnorm_decay: a float number to represent the decay rate for the moving average of batch normalization

    :param is_training: a boolean used for both [batchnorm and dropout] layers which should be true during training and false during testing

    :param data_format: 'NCHW' or 'NHWC'. It's proved that 'NCHW' provides a performance boost for GPUs. However, 'NHWC' is the only one that can be used for CPU computations

    ----
    Return:
    ----
    A scope after construction with the passed arguments
    """
    # L2 Regularization
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    # Glorot initializer is the default initializer
    initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
    initializer_fc = tf.contrib.layers.xavier_initializer()
    # Batch normalization is used if enabled
    normalizer_fn = slim.batch_norm if batchnorm else None
    normalizer_params = {'is_training': is_training, 'center': True, 'scale': True,
                         'decay': batchnorm_decay} if batchnorm else None

    # Create an arg scope for the layers conv2d and separable_conv2d
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], data_format=data_format,
                        weights_initializer=initializer_conv2d,
                        weights_regularizer=regularizer, normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):
        with slim.arg_scope([slim.fully_connected], weights_initializer=initializer_fc,
                            weights_regularizer=regularizer, normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):
            # Create an arg scope for batch_norm layer
            with slim.arg_scope([slim.batch_norm], data_format=data_format):
                # Create an arg scope for the dropout layer
                with slim.arg_scope([slim.dropout], data_format=data_format, is_training=is_training,
                                    keep_prob=dropout_keep_prob) as sc:
                    return sc