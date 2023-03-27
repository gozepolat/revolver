"""# Input layers.
model.add(tf.keras.layers.Flatten(input_shape=x_train_normalized.shape[1:]))
model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.002)
))


# Hidden layers.
model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.002)
))

# Output layers.
model.add(tf.keras.layers.Dense(
    units=10,
    activation=tf.keras.activations.softmax
))"""
import tensorflow as tf
import numpy as np

GLOBAL_PARAMS = {}


def scoped_dense_matrix(prefix, ni, no, use_bias=False, kernel_regularizer=None, activation=None):
    key = '_'.join([str(k) for k in (prefix, ni, no)])

    if key not in GLOBAL_PARAMS:
        GLOBAL_PARAMS[key] = tf.keras.layers.Dense(units=no,
                                                   activation=activation,
                                                   use_bias=use_bias,
                                                   kernel_regularizer=kernel_regularizer)

    return GLOBAL_PARAMS[key]


hyperparams = {
    'hidden_size': 8,
    'hidden_head_count': 512,
    'num_classes': 10,
    'num_samples': 8,
    'approximation_depth': 6,
    'hidden_depth': 1,
    'head_count': 8,
    'vertical_dropout': 0.5
}


def make_conv_block(x, n_channels, kernel_size=3, strides=2, use_batch_norm=True, transposed=False):
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    conv = layers.Conv2D

    if transposed:
        conv = layers.Conv2DTranspose

    x = conv(n_channels, kernel_size, activation="relu", strides=strides, padding="same", use_bias=not use_batch_norm)(x)

    return x


def make_residual_block(x, n_channels, use_layer_norm=False):
    residual = x

    x = make_conv_block(x, n_channels, strides=1)
    x = make_conv_block(x, n_channels, strides=1)

    if residual.shape[-1] != x.shape[-1]:
        x = make_conv_block(x, residual.shape[-1], kernel_size=1, strides=1)

    x = x + residual

    if use_layer_norm:
        x = layers.LayerNormalization()(x)

    return x


class ConvBlock(tf.keras.Layer):
    def __init__(self, n_channels, kernel_size=3, strides=2, use_batch_norm=True, transposed=False, activation="silu",
                 **kwargs):
        super().__init__(**kwargs)

        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()

        conv = layers.Conv2D

        if transposed:
            conv = layers.Conv2DTranspose

        self.conv = conv(n_channels, kernel_size, activation=activation, strides=strides, padding="same",
                 use_bias=not use_batch_norm)

    def call(self, x):
        if self.use_bn:
            x = self.bn(x)
        return self.conv(x)


class ResidualBlock(tf.keras.Layer):
    def __init__(self, in_channels, out_channels, use_layer_norm=False):
        self.conv_block1 = ConvBlock(out_channels, kernel_size=3, strides=1,
                                     use_batch_norm=True, transposed=False, activation="silu")
        self.conv_block2 = ConvBlock(out_channels, kernel_size=3, strides=1,
                                     use_batch_norm=True, transposed=False, activation="silu")
        if in_channels != out_channels:
            self.convdim = ConvBlock(in_channels, kernel_size=1, strides=1,
                                     use_batch_norm=True, transposed=False, activation="silu")
        if residual.shape[-1] != x.shape[-1]:
        x = make_conv_block(x, residual.shape[-1], kernel_size=1, strides=1)


class FlattenBlock(tf.keras.Layer):
    def __init__(self, use_bias=False):
        if not use_bias:
            self.bn = tf.keras.layers.BatchNormalization()

        self.conv_block = tf.keras.layers.Conv2d(n_channels,
                                                 3,
                                                 activation="silu",
                                                 strides=2, padding="same",
                                                 use_bias=not use_batch_norm)
        self.

        self.pooling = layers.GlobalAveragePooling2D()
        self.out_dim = 2048


class TangentMlp(tf.keras.Model):
    def __init__(self, hyperparams=hyperparams,
                 **kwargs):
        super(TangentMlp, self).__init__(**kwargs)

        self.flatten = FlattenBlock()
        self.hidden_size = hyperparams['hidden_size']
        self.hidden_head_count = hyperparams['hidden_head_count']
        self.head_count = hyperparams['head_count']
        self.num_classes = hyperparams['num_classes']
        self.num_samples = hyperparams['num_samples']
        self.approximation_depth = hyperparams['approximation_depth']
        self.hidden_depth = hyperparams['hidden_depth']
        self.vertical_dropout = hyperparams['vertical_dropout']

        self.dense_layers = [scoped_dense_matrix(str(i), ni=self.flatten.out_dim,
                                                 no=self.hidden_size, use_bias=True,
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.002),
                                                 activation=tf.keras.activations.relu,
                                                 ) for i in range(self.head_count)
                             ]

        self.layer_norm = tf.keras.layers.LayerNormalization()
        # self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.hidden_layers = [
            scoped_dense_matrix(str(i),
                                ni=self.hidden_size,
                                no=self.hidden_size,
                                # lower kernel regularizer
                                kernel_regularizer=tf.keras.regularizers.l2(0.0002)
                                ) for i in range(self.hidden_head_count)
        ]

        self.final_hidden_layer = scoped_dense_matrix('final', ni=self.hidden_size,
                                                      no=self.hidden_size, use_bias=False,
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.0002),
                                                      activation=tf.keras.activations.relu, )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        # force initialization
        i = tf.random.uniform((1, self.hidden_size))
        for layer in self.hidden_layers:
            tf.print(layer(i).shape)

        self.output_layer = tf.keras.layers.Dense(
            units=self.num_classes,
            #activation=tf.keras.activations.softmax
        )

    @staticmethod
    def hidden_exponent(layer, hidden_size, approximation_depth):
        I = tf.eye(hidden_size)
        w = I
        div = 1.
        result = I
        for i in range(approximation_depth):
            div *= (i + 1)
            w = tf.linalg.matmul(layer.weights, w)
            result += result + w / div
        return result

    @staticmethod
    def random_choice(input_tensors, num_samples):
        """select some samples from the input tensors

        input_tensors: a regular array of tensors to pick from
        num_samples: number of samples to return (may have duplicates)
        """
        indices = tf.random.categorical(tf.math.log([[1. / len(input_tensors)] * len(input_tensors)]),
                                        num_samples)
        return tf.gather(input_tensors, indices)

    def call_hidden(self, x, weights):
        # exp weights

        # tf.print("weights", weights[0].shape)
        # randomly select n samples (replace=True)
        def step(x_i):
            heads = tf.squeeze(self.random_choice(weights, num_samples=self.num_samples))
            # tf.print('heads', heads.shape)
            # tf.print("heads shape", heads.shape, "heads", heads)
            w = tf.scan(tf.linalg.matmul, heads)[-1]
            # tf.print("scan", w.shape)
            w = tf.matmul(tf.expand_dims(x_i, 0), w)
            # #tf.print('matmul', w.shape)
            return tf.squeeze(w)

        # tf.print("step 0", step(x[0]))
        return tf.vectorized_map(step, x)

    def make_exp_weights(self):
        weights = []
        for i in range(self.hidden_head_count):
            # result = self.hidden_exponent(self.hidden_layers[i],
            #                              self.hidden_size,
            #                              self.approximation_depth)
            result = tf.linalg.expm(self.hidden_layers[i].weights)
            weights.append(result)
        return weights

    def maybe_call_hidden(self, x, weights, training):
        def identity():
            return x

        if training:
            def hidden_block():
                y = self.call_hidden(x, weights)
                y = tf.nn.relu(y)
                # tf.print('hidden', x.shape)
                return y

            x = tf.cond(tf.random.uniform((1,)) > self.vertical_dropout,
                        hidden_block, identity)
            return x

        return x

    def call(self, images, training=False):
        x = self.flatten(images)
        # tf.print("flatten", x)
        xs = [layer(x) for layer in self.dense_layers]
        # tf.print("dense1", x.shape)
        weights = self.make_exp_weights()

        xs = [self.maybe_call_hidden(x, weights, training) for x in xs]
        x = tf.concat(xs, -1)
        x = self.layer_norm(x)
        # x = tf.math.reduce_mean(xs, axis=0)
        # x = self.batch_norm1(x)
        x = self.final_hidden_layer(x)
        x = self.batch_norm2(x)
        # tf.print("hidden", x)
        # TODO one more layer?
        x = self.output_layer(x)
        return x

    def call2(self, images):
        x = self.flatten(images)
        x = self.dense_layer(x)

        result = x
        for i in range(self.hidden_depth):
            x = self.hidden_layer(x)
            result += x / np.math.factorial(i + 1)
        x = tf.nn.relu(x)

        x = result(x)
        x = self.output_layer(x)
        return x

    def call_residual(self, x):
        for i in range(self.hidden_depth):
            r = x
            x = self.call_hidden1(x)
            x = tf.nn.relu(x)
            x = self.call_hidden1(x)
            x = tf.nn.relu(x)
            x += r

        return x

