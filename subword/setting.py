class hyperparams:
    
    epochs =20
    batch_size = 64
    num_layers = 4
    num_nodes = 256   #The num of hidden units in one layer
    dff = 1024
    num_heads = 4

    maxlen = 40    #Max length of sentence
    dropout_rate = 0.1
    checkpoint_path = "./checkpoints_head4"
    
    vocab_scr_path='vocab_pt'
    vocab_tar_path='vocab_en'
