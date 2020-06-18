import tensorflow as tf


def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# Function to create a convolutional layer
def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters,
                               _padding='SAME',
                               maxpool=True
                               ):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=_padding)
    layer += biases
    ## We shall be using max-pooling.
    if maxpool:
        layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding=_padding)

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer


# Function to create a Flatten Layer
def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()
    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()
    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer


# Function to create a Fully - Connected Layer
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True,
                    dropoutRate=0.2
                    ):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    dropped = tf.nn.dropout(input, rate=dropoutRate)
    layer = tf.matmul(dropped, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer




