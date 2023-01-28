import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dr=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dr)
        self.dropout2 = layers.Dropout(dr)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(
        self, maxlen, vocab_size, embed_dim, layer_norm=True, dropout_rate=0.0
    ):
        self.maxlen = maxlen
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
            embeddings_initializer=tf.keras.initializers.Constant(
                get_pos_encoding_matrix(self.maxlen, embed_dim)
            ),
        )
        self.layernorm = (
            layers.LayerNormalization(epsilon=1e-12, dtype="float32")
            if layer_norm
            else None
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        # self.maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        embedding = x + positions
        if self.layernorm is not None:
            embedding = self.layernorm(embedding)
        embedding = self.dropout(embedding)
        return embedding