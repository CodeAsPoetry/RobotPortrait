import tensorflow as tf

from bot_img.run_classifier_bi_attr import InputExample, convert_examples_to_features

from bot_img.run_classifier_bi_attr import create_model as bi_create_model

from bot_img.run_classifier_muti_attr import create_model as muti_create_model

import bot_img.tokenization as tokenization
import bot_img.bi_modeling as bi_modeling
import bot_img.muti_modeling as muti_modeling

import bot_img.configurations as cf

import pandas as pd


class BotImageModel(object):

    def __init__(self):
        # 建立两个 graph
        self.g1 = tf.Graph()
        self.g2 = tf.Graph()

        # 为每个 graph 建创建一个 session
        self.sess1 = tf.Session(graph=self.g1)
        self.sess2 = tf.Session(graph=self.g2)

        self.bi_label_list = ['0', '1']
        self.bi_num_labels = len(self.bi_label_list)

        self.muti_label_list = [str(i) for i in range(40)]
        self.muti_num_labels = len(self.muti_label_list)

        self.configuration = cf.get_config()

        self.max_seq_len = self.configuration['max_seq_length']
        self.bi_bert_config_file = self.configuration['bert_config']
        self.muti_bert_config_file = self.configuration['bert_config']

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.configuration['vocab_file'], do_lower_case=True)

        self.bi_init_checkpoint = self.configuration['bi_init_checkpoint']
        self.muti_init_checkpoint = self.configuration['muti_init_checkpoint']

        self._init_session()

    def _init_session(self):
        with self.sess1.as_default():
            with self.sess1.graph.as_default():
                self.graph = tf.get_default_graph()

                self.bi_input_ids_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="input_ids")
                self.bi_input_mask_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="input_mask")
                self.bi_segment_ids_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="segment_ids")
                self.bi_labels_ph = tf.placeholder(
                    tf.int32, [None, ], name="labels_ph")

                self.bi_bert_config = bi_modeling.BertConfig.from_json_file(self.bi_bert_config_file)

                _, _, self.bi_logits, self.bi_prediction = bi_create_model(
                    self.bi_bert_config, False, self.bi_input_ids_ph, self.bi_input_mask_ph,
                    self.bi_segment_ids_ph, self.bi_labels_ph, self.bi_num_labels, False)

                saver1 = tf.train.Saver()
                saver1.restore(self.sess1, tf.train.latest_checkpoint(self.bi_init_checkpoint))

                print('Successfully load the model_1!')

        with self.sess2.as_default():
            with self.sess2.graph.as_default():

                self.graph = tf.get_default_graph()

                self.muti_input_ids_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="input_ids")
                self.muti_input_mask_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="input_mask")
                self.muti_segment_ids_ph = tf.placeholder(
                    tf.int32, [None, self.max_seq_len], name="segment_ids")
                self.muti_labels_ph = tf.placeholder(
                    tf.int32, [None, ], name="labels_ph")

                self.muti_bert_config = muti_modeling.BertConfig.from_json_file(self.muti_bert_config_file)

                _, _, self.muti_logits, self.muti_prediction = muti_create_model(
                    self.muti_bert_config, False, self.muti_input_ids_ph, self.muti_input_mask_ph,
                    self.muti_segment_ids_ph, self.muti_labels_ph, self.muti_num_labels, False)

                saver2 = tf.train.Saver()
                saver2.restore(self.sess2, tf.train.latest_checkpoint(self.muti_init_checkpoint))

                print('Successfully load the model_2!')

    def bi_predict(self, sent):
        prediction = None
        result = {}
        if not sent:
            return 0
        examples = [InputExample(0, sent, label='0')]
        features = convert_examples_to_features(examples, self.bi_label_list, self.max_seq_len, self.tokenizer)
        input_ids = [feature.input_ids for feature in features]
        input_mask = [feature.input_mask for feature in features]
        segment_ids = [feature.segment_ids for feature in features]
        label_ids = [feature.label_id for feature in features]

        with self.sess1.as_default():
            with self.sess1.graph.as_default():
                feed_dict = {self.bi_input_ids_ph: input_ids, self.bi_input_mask_ph: input_mask,
                             self.bi_segment_ids_ph: segment_ids, self.bi_labels_ph: label_ids}
                prediction = self.sess1.run([self.bi_prediction], feed_dict)
        if not prediction or type(prediction) != list:
            return None

        if prediction[0][0][0] < prediction[0][0][1]:
            result['attr'] = 'yes'
        else:
            result['attr'] = 'no'

        return result

    def muti_predict(self, sent):
        attr_label = ['中文名', '英文名', '昵称', '出生日期', '生肖', '年龄', '星座', '血型',
                      '性别', '优点', '缺点', '性格', '学历', '职业', '工作时长', '充电时长',
                      '行走速度', '越障能力', '国籍', '家乡', '住址', '语言', '联系方式', '身高',
                      '体重', '手', '脚', '颜色', '兴趣', '爸爸', '妈妈', '哥哥',
                      '弟弟', '姐姐', '妹妹', '功能', '其他']
        prediction = None
        result = {}
        if not sent:
            return 0
        examples = [InputExample(0, sent, label='0')]
        features = convert_examples_to_features(examples, self.muti_label_list, self.max_seq_len, self.tokenizer)
        input_ids = [feature.input_ids for feature in features]
        input_mask = [feature.input_mask for feature in features]
        segment_ids = [feature.segment_ids for feature in features]
        label_ids = [feature.label_id for feature in features]

        with self.sess2.as_default():
            with self.sess2.graph.as_default():
                feed_dict = {self.muti_input_ids_ph: input_ids, self.muti_input_mask_ph: input_mask,
                             self.muti_segment_ids_ph: segment_ids, self.muti_labels_ph: label_ids}
                prediction = self.sess2.run([self.muti_prediction], feed_dict)
        if not prediction or type(prediction) != list:
            return None

        prediction_list = prediction[0].tolist()[0]
        predict_label = prediction_list.index(max(prediction_list))
        result['attr'] = attr_label[predict_label]

        return result

    def get_three_tuple(self, sent):
        bi_result = self.bi_predict(sent)
        if bi_result['attr'] == 'no':
            return None
        else:
            muti_result = serving.muti_predict(sent)
            if muti_result['attr'] == '其他':
                return None
            else:
                three_tuple = ('机器人', muti_result['attr'], '?')
                return three_tuple


if __name__ == '__main__':
    serving = BotImageModel()

    while True:
        sentence = input("sentence：")
        three_tuple = serving.get_three_tuple(sentence)
        print(three_tuple)

    # true_corpus_file_path = '2020-01-10.csv'
    # true_corpus_result_path = '2020-01-10-result.tsv'
    # true_corpus_result = open(true_corpus_result_path, 'w', encoding='utf-8')
    #
    # df = pd.read_csv(true_corpus_file_path, header=None, names=['sent'])
    # corpus_num = len(df)
    #
    # for i in range(corpus_num):
    #
    #     three_tuple = serving.get_three_tuple(df.iloc[i]['sent'])
    #     true_corpus_result.write(df.iloc[i]['sent'] + '\t' + str(three_tuple) + '\n')
    #
    # true_corpus_result.close()

    # import time
    # from tqdm import tqdm
    # sum_time = 0
    # for i in tqdm(range(2003)):
    #     sent = '你是个男生还是女生'
    #     start = time.time()
    #     three_tuple = serving.get_three_tuple(sent)
    #     end = time.time()
    #     if i < 3:
    #         print(three_tuple)
    #         continue
    #     else:
    #         sum_time += end-start
    #
    # print(sum_time/2000)



