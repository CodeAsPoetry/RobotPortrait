def get_config():
    config = {}
    config['bert_config'] = '../bert_3/chinese_L-3_H-576_A-3/bert_config.json'
    config['do_lower_case'] = True
    config['bi_init_checkpoint'] = "bi_model_bert3_0115/"
    config['muti_init_checkpoint'] = "muti_model_bert3_4_0115/"
    config['max_seq_length'] = 64
    config['vocab_file'] = "../bert_3/chinese_L-3_H-576_A-3/vocab_4380.txt"
    return config