import tensorflow as tf
from hyperparams import Hyperparams as hp
# from data_load import get_batch_data, load_de_vocab, load_en_vocab
from train_data import load_train_data
from modules import *
import time
from tqdm import tqdm

# tf.debugging.set_log_device_placement(True)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(hp.num_units, hp.num_heads)
        self.ffn = feedforward(hp.num_units, hp.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(hp.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(hp.dropout_rate)

    def call(self, x, training, mask):

        attn_output= self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self,input_vocab_size):
        super(Encoder, self).__init__()
        self.num_units = hp.num_units
        self.num_layers=hp.num_blocks

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, hp.num_units)
        self.pos_encoding = positional_encoding(input_vocab_size, hp.num_units)
        self.enc_layers = [EncoderLayer() for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(hp.dropout_rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.num_units, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(hp.num_units, hp.num_heads)
        self.mha2 = MultiHeadAttention(hp.num_units, hp.num_heads)

        self.ffn = feedforward(hp.num_units, hp.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(hp.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(hp.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(hp.dropout_rate)


    def call(self, x, enc_output, training, 
               look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3
    
    
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size):
        super(Decoder, self).__init__()
        self.num_units = hp.num_units
        self.num_layers=hp.num_blocks
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, hp.num_units)
        self.pos_encoding = positional_encoding(target_vocab_size, hp.num_units)

        self.dec_layers = [DecoderLayer() for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(hp.dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.num_units, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x= self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_vocab_size)

        self.decoder = Decoder(target_vocab_size)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output= self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=8000):
        super(CustomSchedule, self).__init__()

        self.num_units = hp.num_units
        self.num_units = tf.cast(self.num_units, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.num_units) * tf.math.minimum(arg1, arg2)
    

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
    

if __name__ == '__main__':                
    # Load vocabulary    
    train_dataset,input_vocab_size, target_vocab_size=load_train_data()
#     train_dataset = train_dataset.cache()
    train_dataset = train_dataset.padded_batch(hp.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    transformer = Transformer(input_vocab_size, target_vocab_size)
#     learning_rate = CustomSchedule(hp.num_units)
    learning_rate=0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    #checkpoints
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
#     train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),tf.TensorSpec(shape=(None, None), dtype=tf.int64),]

#     @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in tqdm(range(hp.num_epochs)):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in tqdm(enumerate(train_dataset)):
            train_step(inp, tar)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    
    print("Done training") 
