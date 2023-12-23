import tensorflow as tf
from utils import PositionalEmbeddings

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )        

        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x
    
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x
        )
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x
    
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x

class FeedForwad(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(rate=dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout=dropout_rate
        )
        self.feedforward = FeedForwad(
            d_model=d_model,
            dff=dff
        )
    
    def call(self, x):
        x = self.self_attention(x)
        x = self.feedforward(x)
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, num_heads, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_attention = CausalSelfAttention(
                            num_heads=num_heads, 
                            key_dim=d_model,
                            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
                            num_heads=num_heads,
                            key_dim=d_model,
                            dropout=dropout_rate)
        self.feedforward = FeedForwad(
                            d_model=d_model,
                            dff=dff,
                            dropout_rate=dropout_rate)
    
    def call(self, x, encoder_output):
        x = self.causal_attention(x=x)
        x = self.cross_attention(x=x, context=encoder_output)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.feedforward(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers=7, d_model, num_heads,
                dff, vocab_size, dropout_rate=0.1):
        super().__init__()        
        self.d_model = d_model
        self.num_layers = num_layers
        self.pe_layer = PositionalEmbeddings(vocab_size=vocab_size, d_model=d_model)
        self.encoder_layers = [EncoderLayer(num_heads=num_heads,
                                            d_model=d_model,
                                            dff=dff,
                                            dropout_rate=dropout_rate) 
                                for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x):
        x = self.pe_layer(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers=7, d_model, num_heads, 
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = [DecoderLayer(
                            num_heads=num_heads,
                            d_model=d_model,
                            dff=dff,
                            dropout_rate=dropout_rate)
                            for _ in range(num_layers)
                            ]
        self.pe = PositionalEmbeddings(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.last_attn_scores = None
    def call(self, x, context):
        x = self.pe(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, context)
        self.last_attn_scores = self.decoder_layers[-1].last_attn_scores
        return x        
        

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers=7, d_model, dff, num_heads,
                 input_vocab_size, target_vocab_size, dropout_rate):
        super().__init__()
        self.encoder= Encoder(num_layers=num_layers,
                             d_model=d_model,
                             dff=dff,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             vocab_size=input_vocab_size)
        
        self.deocder= Decoder(num_layers=num_layers,
                             d_model=d_model,
                             dff=dff,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             vocab_size=target_vocab_size)
        self.final_layer = tf.keras.layers.Dense(units=target_vocab_size)


    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.deocder(x=x, context=context)
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass
        
        return logits