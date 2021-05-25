from train import Transformer, translate_batch
from setting import hyperparams as hp
import tensorflow_datasets as tfds
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu




if __name__ == '__main__': 
    samples, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    test_data=samples['test']
    
    coder_scr = tfds.features.text.SubwordTextEncoder.load_from_file(hp.vocab_scr_path)
    coder_tar = tfds.features.text.SubwordTextEncoder.load_from_file(hp.vocab_tar_path)
    
    real=[]
    for scr, tar in test_data:
        real.append([bytes.decode(tar.numpy()).split()])
        
    def encoder(scr, tar):
        scr = [coder_scr.vocab_size] + coder_scr.encode(
          scr.numpy()) + [coder_scr.vocab_size+1]

        tar = [coder_tar.vocab_size] + coder_tar.encode(
          tar.numpy()) + [coder_tar.vocab_size+1]

        return scr, tar

    def tf_encoder(scr, tar):
        code_scr, code_tar = tf.py_function(encoder, [scr, tar], [tf.int64, tf.int64])
        code_scr.set_shape([None])
        code_tar.set_shape([None])

        return code_scr, code_tar

    test_data = test_data.map(tf_encoder)
    test_data = test_data.padded_batch(hp.batch_size)
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    input_vocab_size = coder_scr.vocab_size + 2
    target_vocab_size = coder_tar.vocab_size + 2

    optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    transformer = Transformer(input_vocab_size, target_vocab_size)

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, hp.checkpoint_path, max_to_keep=20)

    if ckpt_manager.latest_checkpoint:
#         print(ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    pred=[]
    for (inp, tar) in test_data:
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
    
    print("Test Bleu Score = " + str(score*100))
