def get_config():
    config = {}
    config['bert_config'] = '../chinese_L-12_H-768_A-12/bert_config.json'
    config['do_lower_case'] = True
    config['init_checkpoint'] = "muti_model_bert3_4_0115/"
    config['max_seq_length'] = 128
    config['vocab_file'] = "../chinese_L-12_H-768_A-12/vocab.txt"
    return config