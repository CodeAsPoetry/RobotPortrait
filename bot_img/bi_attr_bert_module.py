import os
import time
import bot_img.bi_modeling as modeling
import collections
import bot_img.tokenization as tokenization
import tensorflow as tf

from bot_img.run_classifier_bi_attr import InputExample, InputFeatures, convert_single_example, create_model, \
    convert_examples_to_features
import bot_img.bi_configurations as cf_bi

from tqdm import tqdm

# from wasabi import Printer
# printer = Printer()
printer = print

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class BiBertTest(object):
    def __init__(self):
        self.configuration = cf_bi.get_config()
        self.bert_config_file = self.configuration['bert_config']
        self.do_lower_case = self.configuration['do_lower_case']
        self.init_checkpoint = self.configuration['init_checkpoint']
        self.max_seq_length = self.configuration['max_seq_length']
        self.vocab_file = self.configuration['vocab_file']
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.label_list = ['0', '1']

        Args = collections.namedtuple("Args", ["init_checkpoint", "max_seq_length", "bert_config", "num_labels"])
        self.args = Args(init_checkpoint=self.init_checkpoint, max_seq_length=self.max_seq_length,
                         bert_config=self.bert_config, num_labels=len(self.label_list))

        self.is_training = False
        self.use_one_hot_embeddings = False
        self.session = None
        self.graph = None

        self.input_ids_ph = None
        self.input_mask_ph = None
        self.segment_ids_ph = None

        self.init_checkpoint = self.args.init_checkpoint
        self.max_seq_length = self.args.max_seq_length
        self.bert_config = self.args.bert_config
        self.num_labels = self.args.num_labels

        self._init_session()

        self._build_graph()

    def _init_session(self):

        # printer.info("*** Init Session ***")

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        log_device_placement = True
        self.session = tf.Session(config=sess_config)

    def _build_graph(self):
        if not os.path.exists(self.init_checkpoint):
            raise Exception(
                "*** NO SUCH FILE: {} ***".format(self.init_checkpoint))

        # printer.info("*** Build Graph ***")

        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.input_ids_ph = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="input_ids")
            self.input_mask_ph = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="input_mask")
            self.segment_ids_ph = tf.placeholder(
                tf.int32, [None, self.max_seq_length], name="segment_ids")
            self.labels_ph = tf.placeholder(
                tf.int32, [None, ], name="labels_ph")

            _, _, self.logits, self.prediction = create_model(
                self.bert_config, self.is_training, self.input_ids_ph, self.input_mask_ph, self.segment_ids_ph,
                self.labels_ph, self.num_labels, self.use_one_hot_embeddings)

            saver = tf.train.Saver()
            saver.restore(self.session, tf.train.latest_checkpoint(self.init_checkpoint))

    def predict(self, sent):
        prediction = None
        result = {}
        if not sent:
            return 0
        examples = [InputExample(0, sent, label='0')]
        features = convert_examples_to_features(examples, self.label_list, self.max_seq_length, self.tokenizer)
        input_ids = [feature.input_ids for feature in features]
        input_mask = [feature.input_mask for feature in features]
        segment_ids = [feature.segment_ids for feature in features]
        label_ids = [feature.label_id for feature in features]

        with self.graph.as_default():
            feed_dict = {self.input_ids_ph: input_ids, self.input_mask_ph: input_mask,
                         self.segment_ids_ph: segment_ids, self.labels_ph: label_ids}
            prediction = self.session.run([self.prediction], feed_dict)
        if not prediction or type(prediction) != list:
            return None

        if prediction[0][0][0] < prediction[0][0][1]:
            result['attr'] = 'yes'
        else:
            result['attr'] = 'no'

        return result

    def get_session(self):
        return self.session

    def get_some_config(self):
        return self.label_list, self.max_seq_length, self.tokenizer

    def get_some_tensor(self):
        return self.input_ids_ph, self.input_mask_ph, self.segment_ids_ph, self.prediction


if __name__ == "__main__":
    serving = BiBertTest()
    sentence = input("sentence：")
    while sentence:
        bi_attr_result = serving.predict(sentence)
        print(bi_attr_result)
        sentence = input("sentence：")

    # sum_time = 0
    # for i in tqdm(range(203)):
    #     t_start = time.time()
    #     bi_attr_result = serving.predict(sentence)
    #     t_end = time.time()
    #     # print(bi_attr_result)
    #     if i < 3:
    #         continue
    #     else:
    #         sum_time += t_end - t_start
    #
    # print(sum_time/200)
