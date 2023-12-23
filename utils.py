import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


def positional_encoding(length, depth):
    """
    Method uses concatenation instead of interleaving
    """
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis] # (length, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth # (1, depth) formula i/dmodel where i is index
    angle_rates = 1 / (10000**(2*depths))
    angle_rads = positions*angle_rates
    encodings = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(encodings, tf.float32)

def positional_encoding_original(length, depth):
    pass
#TODO-> implement with interleaving like original paper PE(sin,2i) PE(cos,2i+1)


def masked_loss(label, pred):
    mask = label != 0
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_function(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, dtype=pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


class PositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) #mask_zero -> mask all zero enteries(dont add in computation)
        self.pos_encodings = positional_encoding(2048, d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embeddings.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embeddings(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encodings[tf.newaxis, :length, :] # adds upto the length of scentence (max is 2048)
        return x


class CustomScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.pow(step, -0.5)
        arg2 = step*(self.warmup_steps**-1.5)
        return tf.pow(self.d_model, -0.5) * tf.minimum(arg1, arg2)
    


# if __name__=="__main__":
#     learning_rate = CustomScheduler(128)
#     x = tf.range(40000, dtype=tf.float32)
#     y = learning_rate(x)
#     plt.plot(x, y)
#     plt.show()
    #     pos_encoding = positional_encoding(2048, 512) 
#     pos_encoding.shape
#     plt.pcolormesh(pos_encoding.numpy().T, cmap="RdBu")
#     plt.xlabel('Positions')
#     plt.ylabel('Depth')
#     plt.colorbar()
#     plt.show()
# pos_encoding/=tf.norm(pos_encoding, axis=1, keepdims=True)
# p = pos_encoding[1000]
# dots = tf.einsum('pd,d -> p', pos_encoding, p) # p->position d->dimension (dot product)
# plt.subplot(2,1,1)
# plt.plot(dots)
# plt.ylim([0,1])
# plt.plot([950, 950, float('nan'), 1050, 1050],
#         [0,1,float('nan'),0,1], color='k', label='Zoom')
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(dots)
# plt.xlim([950, 1050])
# plt.ylim([0,1])
