import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
from setting import hyperparams as hp
from nltk.translate.bleu_score import corpus_bleu



# positional encoding
def get_angles(pos, i, num_nodes):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(num_nodes))
    return pos * angle_rates

def positional_encoding(position, num_nodes):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(num_nodes)[np.newaxis, :], num_nodes)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

#mask padding tokens and future tokens
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(scr, tar):
    enc_padding_mask = create_padding_mask(scr)

    dec_padding_mask = create_padding_mask(scr)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_nodes, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes

        assert num_nodes % self.num_heads == 0

        self.depth = num_nodes // self.num_heads

        self.wq = tf.keras.layers.Dense(num_nodes)
        self.wk = tf.keras.layers.Dense(num_nodes)
        self.wv = tf.keras.layers.Dense(num_nodes)

        self.dense = tf.keras.layers.Dense(num_nodes)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, num_nodes)
        k = self.wk(k)  # (batch_size, seq_len, num_nodes)
        v = self.wv(v)  # (batch_size, seq_len, num_nodes)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.num_nodes))  # (batch_size, seq_len_q, num_nodes)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, num_nodes)

        return output, attention_weights

def point_wise_feed_forward_network(num_nodes, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(num_nodes)  # (batch_size, seq_len, num_nodes)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_nodes, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_nodes, num_heads)
        self.ffn = point_wise_feed_forward_network(num_nodes, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, num_nodes)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, num_nodes)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, num_nodes)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, num_nodes)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_nodes, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_nodes, num_heads)
        self.mha2 = MultiHeadAttention(num_nodes, num_heads)

        self.ffn = point_wise_feed_forward_network(num_nodes, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
               look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, num_nodes)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, num_nodes)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, num_nodes)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, num_nodes)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, num_nodes)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, num_nodes)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_nodes, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, num_nodes)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.num_nodes)


        self.enc_layers = [EncoderLayer(num_nodes, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, num_nodes)
        x *= tf.math.sqrt(tf.cast(self.num_nodes, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, num_nodes)
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_nodes, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, num_nodes)
        self.pos_encoding = positional_encoding(maximum_position_encoding, num_nodes)

        self.dec_layers = [DecoderLayer(num_nodes, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, num_nodes)
        x *= tf.math.sqrt(tf.cast(self.num_nodes, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, num_nodes)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_nodes, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, num_nodes, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, num_nodes, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, num_nodes)

        # dec_output.shape == (batch_size, tar_seq_len, num_nodes)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, num_nodes, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.num_nodes = num_nodes
        self.num_nodes = tf.cast(self.num_nodes, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.num_nodes) * tf.math.minimum(arg1, arg2)

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions= transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    
def translate_batch(inp,out,transformer):
    out=tf.cast(out,tf.int32)
    for i in range(hp.maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, out)

    # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer(inp, out, False, enc_padding_mask, combined_mask, dec_padding_mask)

        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        out = tf.concat([out, predicted_id], axis=-1)

    return out


if __name__ == '__main__': 
    
    samples, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_samples, val_samples = samples['train'], samples['validation']

    try:
        coder_scr = tfds.features.text.SubwordTextEncoder.load_from_file('vocab_scr.pt')
        coder_tar = tfds.features.text.SubwordTextEncoder.load_from_file('vocab_tar.en')
        print('restore')
    except:
        coder_tar = tfds.features.text.SubwordTextEncoder.build_from_corpus((tar.numpy() for scr, tar in train_samples), target_vocab_size=2**13)
        coder_scr = tfds.features.text.SubwordTextEncoder.build_from_corpus((scr.numpy() for scr, tar in train_samples), target_vocab_size=2**13)
        coder_tar.save_to_file('vocab_tar.en')
        coder_scr.save_to_file('vocab_scr.pt')
        print('save')

    def encode(lang1, lang2):
        lang1 = [coder_scr.vocab_size] + coder_scr.encode(
          lang1.numpy()) + [coder_scr.vocab_size+1]

        lang2 = [coder_tar.vocab_size] + coder_tar.encode(
          lang2.numpy()) + [coder_tar.vocab_size+1]

        return lang1, lang2

    def select_sents(x, y):
        return tf.logical_and(tf.size(x) <= hp.maxlen, tf.size(y) <= hp.maxlen)

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    train_dataset = train_samples.map(tf_encode)  #encode sentence
    train_dataset = train_dataset.filter(select_sents)  #select sentences not longer than max length

    #get batch
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.padded_batch(hp.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)    
    
    #get validation data
    real=[]
    for scr, tar in val_samples:
        real.append([bytes.decode(tar.numpy()).split()])
    val_dataset = val_samples.map(tf_encode)
    val_dataset = val_dataset.padded_batch(hp.batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    input_vocab_size = coder_scr.vocab_size + 2
    target_vocab_size = coder_tar.vocab_size + 2

    learning_rate = CustomSchedule(hp.num_nodes)

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

    transformer = Transformer(hp.num_layers, hp.num_nodes, hp.num_heads, hp.dff, input_vocab_size, target_vocab_size, pe_input=input_vocab_size, 
                              pe_target=target_vocab_size,rate=hp.dropout_rate)


    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, hp.checkpoint_path, max_to_keep=20)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


    for epoch in range(hp.epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

        ckpt_save_path = ckpt_manager.save()
        
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs'.format(time.time() - start))

        #validation bleu
        pred=[]
        for (inp, tar) in val_dataset:
            tar_real = tar[:, :1]
            out=translate_batch(inp,tar_real,transformer)
            for sents in out:
                bool_idx = tf.equal(sents, coder_tar.vocab_size+1)
                i=tf.where(bool_idx).numpy().tolist()
                if i:
                    i=i[0][0]
                    sents=sents[1:i]
                pred.append(coder_tar.decode([i for i in sents if i < coder_tar.vocab_size]).split())
        score = corpus_bleu(real, pred)
        print("Validation Bleu Score ={} \n".format(score*100))
    
