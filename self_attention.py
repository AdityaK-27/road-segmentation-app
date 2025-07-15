import tensorflow as tf
from tensorflow.keras import layers, models
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.query_conv = layers.Conv2D(self.filters // 8, (1, 1), padding="same")
        self.key_conv = layers.Conv2D(self.filters // 8, (1, 1), padding="same")
        self.value_conv = layers.Conv2D(self.filters, (1, 1), padding="same")

    def call(self, x):
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        q_reshaped = tf.reshape(q, [tf.shape(x)[0], -1, self.filters // 8])
        k_reshaped = tf.reshape(k, [tf.shape(x)[0], -1, self.filters // 8])
        v_reshaped = tf.reshape(v, [tf.shape(x)[0], -1, self.filters])

        attention = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)
        attention_output = tf.matmul(attention, v_reshaped)
        attention_output = tf.reshape(attention_output, tf.shape(x))

        return tf.add(attention_output, x)

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"filters": self.filters})
        return config