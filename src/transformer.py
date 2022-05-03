import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine import data_adapter
# from tensorflow.python.keras.engine.training import _minimize
# from tensorflow.python.util import nest
# from tensorflow.python.ops import array_ops, math_ops
# from tensorflow.python.keras.utils import losses_utils
import pdb

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, split_head=None, global_heads=None, fill_cont=None):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, split_head, global_heads, fill_cont)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, graph_mask=None):

        attn_output, attn_weights_block1 = self.mha(x, x, x, mask,  graph_mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights_block1

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, split_head, global_heads, fill_cont):
        # d_model: hidden dimension
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='mhaq')
        self.wk = tf.keras.layers.Dense(d_model, name='mhak')
        self.wv = tf.keras.layers.Dense(d_model, name='mhav')

        self.dense = tf.keras.layers.Dense(d_model, name='mha_dense')
        self.split_head = split_head
        self.global_heads = global_heads
        self.fill_cont = fill_cont

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,graph_mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        if self.split_head:
            scaled_attention, attention_weights = local_global_attention(
                q, k, v, mask, graph_mask, self.global_heads, self.fill_cont)
        else:
            scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask, graph_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def local_global_attention(q, k, v, mask, graph_mask, global_heads, fill_cont):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    # q = q / tf.math.sqrt(head_dim)

    # split the heads to global and local
    q_local = q[:,:-global_heads,:,:]
    q_global = q[:,-global_heads:,:,:]
    k_local = k[:,:-global_heads,:,:]
    k_global = k[:,-global_heads:,:,:]
    v_local = v[:,:-global_heads,:,:]
    v_global = v[:,-global_heads:,:,:]

    output_global, attention_weights_global = scaled_dot_product_attention(q_global, k_global, v_global, mask, graph_mask)
    output_local, attention_weights_local = local_attention(q_local, k_local, v_local, mask, graph_mask, fill_cont)

    return output, attention_weights

def local_attention(q, k, v, mask, graph_mask, fill_cont):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    bsz, seq_len, num_heads, head_dim = q.shape
    if tf.executing_eagerly():
        tf.debugging.assert_equal(
            seq_len % (fill_cont * 2),
            0,
            message=f"Sequence length should be multiple of {fill_cont * 2}. Given {seq_len}",
        )
        tf.debugging.assert_equal(
            q.shape,
            k.shape,
            message=f"Shape of query and key should be equal, but got query: {shape_list(query)} and key: {shape_list(key)}",
        )

    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    q = q / tf.math.sqrt(dk)
    # overlapping chunk of size 2*fill_cont overlapping fill_cont
    chunks_count = seq_len // fill_cont - 1
    
    
    q = tf.reshape(tf.transpose(q, perm=[0,2,1,3]), [bsz*num_heads, seq_len, head_dim])
    k = tf.reshape(tf.transpose(k, perm=[0,2,1,3]), [bsz*num_heads, seq_len, head_dim])
    
    chunk_q = _chunk(q, fill_cont) # bsz*n_head, chunks, 2w, head_dim
    chunk_k = _chunk(k, fill_cont) # bsz*n_head, chunks, 2w, head_dim


    chunk_q = tf.cast(chunk_q, dtype=chunk_k.dtype)
    chunk_attn = tf.einsum("bcxd,bcyd->bcxy", chunk_q, chunk_k) # bsz*n_head, chunks, 2w, 2w

    # convert diagonals into columns
    paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
    diagonal_chunked_attention_scores = _pad_and_transpose_last_two_dims(chunk_attn, paddings)

    # allocate space for the overall attention matrix where the chunks are combined. The last dimension
    # has (fill_cont * 2 + 1) columns. The first (fill_cont) columns are the fill_cont lower triangles (attention from a word to
    # fill_cont previous words). The following column is attention score from each word to itself, then
    # followed by fill_cont columns for the upper triangle.

    # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
    # - copying the main diagonal and the upper triangle
    # TODO: This code is most likely not very efficient and should be improved
    diagonal_attn_scores_up_triang = tf.concat(
        [
            diagonal_chunked_attention_scores[:, :, :fill_cont, : fill_cont + 1],
            diagonal_chunked_attention_scores[:, -1:, fill_cont:, : fill_cont + 1],
        ],
        axis=1,
    )

    # - copying the lower triangle
    diagonal_attn_scores_low_triang = tf.concat(
        [
            tf.zeros(
                (bsz * num_heads, 1, fill_cont, fill_cont),
                dtype=diagonal_chunked_attention_scores.dtype,
            ),
            diagonal_chunked_attention_scores[:, :, -(fill_cont + 1) : -1, fill_cont + 1 :],
        ],
        axis=1,
    )
    diagonal_attn_scores_first_chunk = tf.concat(
        [
            tf.roll(
                diagonal_chunked_attention_scores,
                shift=[1, fill_cont],
                axis=[2, 3],
            )[:, :, :fill_cont, :fill_cont],
            tf.zeros(
                (bsz * num_heads, 1, fill_cont, fill_cont),
                dtype=diagonal_chunked_attention_scores.dtype,
            ),
        ],
        axis=1,
    )
    first_chunk_mask = (
        tf.tile(
            tf.range(chunks_count + 1)[None, :, None, None],
            (bsz * num_heads, 1, fill_cont, fill_cont),
        )
        < 1
    )
    diagonal_attn_scores_low_triang = tf.where(
        first_chunk_mask,
        diagonal_attn_scores_first_chunk,
        diagonal_attn_scores_low_triang,
    )

    # merging upper and lower triangle
    diagonal_attention_scores = tf.concat(
        [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1
    )

    # separate batch_size and num_heads dimensions again
    diagonal_attention_scores = tf.transpose(
        tf.reshape(
            diagonal_attention_scores,
            (bsz, num_heads, seq_len, 2 * fill_cont + 1),
        ),
        (0, 2, 1, 3),
    )

    diagonal_attention_scores = _mask_invalid_locations(diagonal_attention_scores, fill_cont)


    # # add the mask to the scaled tensor.
    # if mask is not None:
    #     scaled_attention_logits += (mask * -1e9)
    # if graph_mask is not None:
    #     # head_size = q.shape[1]
    #     # graph_mask = tf.expand_dims(graph_mask, axis=1)
    #     # graph_mask = tf.tile(graph_mask, tf.constant([1,head_size,1,1],tf.int32))
    #     graph_mask = graph_mask[:,tf.newaxis, :,:]
    #     scaled_attention_logits += (-1e9 * (1-graph_mask))

    # # softmax is normalized on the last axis (seq_len_k) so that the scores
    # # add up to 1.
    # attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def _mask_invalid_locations(input_tensor, window_overlap):
    # create correct upper triangle bool mask
    mask_2d_upper = tf.reverse(
        tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
        axis=[0],
    )

    # pad to full matrix
    padding = tf.convert_to_tensor(
        [[0, input_tensor.shape[1] - window_overlap], [0, input_tensor.shape[3] - window_overlap - 1]]
    )

    # create lower mask
    mask_2d = tf.pad(mask_2d_upper, padding)

    # combine with upper mask
    mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

    # broadcast to full matrix
    mask_4d = tf.tile(mask_2d[None, :, None, :], (input_tensor.shape[0], 1, 1, 1))

    # inf tensor used for masking
    inf_tensor = -float("inf") * tf.ones_like(input_tensor)

    # mask
    input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

    return input_tensor

def _pad_and_transpose_last_two_dims(x, paddings):
    """pads rows and then flips rows and columns"""
    x = tf.pad(
        x, paddings
    )  # padding value is not important because it will be overwritten
    batch_size, chunk_size, seq_length, hidden_dim = x.shape
    x = tf.reshape(x, (batch_size, chunk_size, hidden_dim, seq_length))

    return x


def _chunk(x,w):
    ''' convert into overlapping chunks, chunk size=2w, overlap size = w'''
    # split chunks
    bsz, seq_len, head_dim = x.shape
    num_out_chunks = seq_len // w -1

    # define frame size and stride
    frame_hop_size = w * head_dim
    frame_size = 2 * w * head_dim
    x = tf.reshape(x, [bsz, seq_len * head_dim])

    # chunk with overlap
    chunked_x = tf.signal.frame(x, frame_size, frame_hop_size)
    chunked_x = tf.reshape(chunked_x, (bsz, num_out_chunks, 2*w, head_dim))

    return chunked_x

def scaled_dot_product_attention(q, k, v, mask, graph_mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


    # add the mask to the scaled tensor.
    
    if graph_mask is not None:
        # head_size = q.shape[1]
        # graph_mask = tf.expand_dims(graph_mask, axis=1)
        # graph_mask = tf.tile(graph_mask, tf.constant([1,head_size,1,1],tf.int32))
        scaled_attention_logits += (-1e3 * (graph_mask[:,tf.newaxis, :,:]+mask))
    else:
        scaled_attention_logits += (mask * -1e3)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=tf.nn.leaky_relu),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

def create_padding_mask(seq):
    # 25 is the zero padding
    seq = tf.cast(tf.math.equal(seq, 25), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # get positional encoding for sequence
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)