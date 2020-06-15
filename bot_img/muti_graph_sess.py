import tensorflow as tf

from bot_img.run_classifier_bi_attr import create_model as bi_create_model

from bot_img.run_classifier_muti_attr import create_model as muti_create_model

import bot_img.tokenization as tokenization
import bot_img.bi_modeling as bi_modeling
import bot_img.muti_modeling as muti_modeling

# 建立两个 graph
g1 = tf.Graph()
g2 = tf.Graph()

# 为每个 graph 建创建一个 session
sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)

with sess1.as_default():
    with sess1.graph.as_default():

        # modelpath = r'bi_model/'
        # saver = tf.train.import_meta_graph(modelpath + 'bi_attr_model.meta')
        # saver.restore(sess1, tf.train.latest_checkpoint(modelpath))
        graph = tf.get_default_graph()

        bi_input_ids_ph = tf.placeholder(
            tf.int32, [None, 128], name="input_ids")
        bi_input_mask_ph = tf.placeholder(
            tf.int32, [None, 128], name="input_mask")
        bi_segment_ids_ph = tf.placeholder(
            tf.int32, [None, 128], name="segment_ids")
        bi_labels_ph = tf.placeholder(
            tf.int32, [None, ], name="labels_ph")

        bi_bert_config_file = '../bert_3/chinese_L-3_H-576_A-3/bert_config.json'
        bi_bert_config = bi_modeling.BertConfig.from_json_file(bi_bert_config_file)

        _, _, bi_logits, bi_prediction = bi_create_model(
            bi_bert_config, False, bi_input_ids_ph, bi_input_mask_ph,
            bi_segment_ids_ph, bi_labels_ph, 2, False)

        saver1 = tf.train.Saver()
        saver1.restore(sess1, tf.train.latest_checkpoint("bi_model_bert3_0108/"))

        print('Successfully load the model_1!')

with sess2.as_default():
    with sess2.graph.as_default():

        # modelpath = r'muti_model/'
        # saver = tf.train.import_meta_graph(modelpath + 'muti_attr_model.meta')
        # saver.restore(sess2, tf.train.latest_checkpoint(modelpath))
        graph = tf.get_default_graph()

        muti_input_ids_ph = tf.placeholder(
            tf.int32, [None, 128], name="input_ids")
        muti_input_mask_ph = tf.placeholder(
            tf.int32, [None, 128], name="input_mask")
        muti_segment_ids_ph = tf.placeholder(
            tf.int32, [None, 128], name="segment_ids")
        muti_labels_ph = tf.placeholder(
            tf.int32, [None, ], name="labels_ph")

        muti_bert_config_file = '../bert_3/chinese_L-3_H-576_A-3/bert_config.json'
        muti_bert_config = muti_modeling.BertConfig.from_json_file(muti_bert_config_file)

        _, _, muti_logits, muti_prediction = muti_create_model(
            muti_bert_config, False, muti_input_ids_ph, muti_input_mask_ph,
            muti_segment_ids_ph, muti_labels_ph, 40, False)

        saver2 = tf.train.Saver()
        saver2.restore(sess2, tf.train.latest_checkpoint("muti_model_bert3_4_0107/"))

        print('Successfully load the model_2!')

from bot_img.run_classifier_bi_attr import InputExample, convert_examples_to_features

bi_label_list = ['0', '1']
bi_num_labels = len(bi_label_list)

muti_label_list = [str(i) for i in range(40)]
muti_num_labels = len(muti_label_list)

tokenizer = tokenization.FullTokenizer(
            vocab_file="../bert_3/chinese_L-3_H-576_A-3/vocab_4380.txt", do_lower_case=True)


def bi_predict(sent):
    prediction = None
    result = {}
    if not sent:
        return 0
    examples = [InputExample(0, sent, label='0')]
    features = convert_examples_to_features(examples, bi_label_list, 128, tokenizer)
    input_ids = [feature.input_ids for feature in features]
    input_mask = [feature.input_mask for feature in features]
    segment_ids = [feature.segment_ids for feature in features]
    label_ids = [feature.label_id for feature in features]

    with sess1.as_default():
        with sess1.graph.as_default():
            feed_dict = {bi_input_ids_ph: input_ids, bi_input_mask_ph: input_mask,
                         bi_segment_ids_ph: segment_ids, bi_labels_ph: label_ids}
            prediction = sess1.run([bi_prediction], feed_dict)
    if not prediction or type(prediction) != list:
        return None

    if prediction[0][0][0] < prediction[0][0][1]:
        result['attr'] = 'yes'
    else:
        result['attr'] = 'no'

    return result


def muti_predict(sent):
    attr_label = ['中文名', '英文名', '昵称', '出生日期', '生肖', '年龄', '星座', '血型',
                  '性别', '优点', '缺点', '性格', '学历', '职业', '工作时长', '充电时长',
                  '行走速度', '越障能力', '国籍', '家乡', '住址', '语言', '联系方式', '身高',
                  '体重', '手', '脚', '颜色', '兴趣', '爸爸', '妈妈', '哥哥',
                  '弟弟', '姐姐', '妹妹', '行走', '唱歌', '跳舞', '导航', '其他']
    prediction = None
    result = {}
    if not sent:
        return 0
    examples = [InputExample(0, sent, label='0')]
    features = convert_examples_to_features(examples, muti_label_list, 128, tokenizer)
    input_ids = [feature.input_ids for feature in features]
    input_mask = [feature.input_mask for feature in features]
    segment_ids = [feature.segment_ids for feature in features]
    label_ids = [feature.label_id for feature in features]

    with sess2.as_default():
        with sess2.graph.as_default():
            feed_dict = {muti_input_ids_ph: input_ids, muti_input_mask_ph: input_mask,
                         muti_segment_ids_ph: segment_ids, muti_labels_ph: label_ids}
            prediction = sess2.run([muti_prediction], feed_dict)
    if not prediction or type(prediction) != list:
        return None

    prediction_list = prediction[0].tolist()[0]
    predict_label = prediction_list.index(max(prediction_list))
    result['attr'] = attr_label[predict_label]

    return result


while True:
    sent = input('请输入句子：')
    bi_result = bi_predict(sent)
    if bi_result['attr'] == 'no':
        print('bi:不是机器人画像属性')
    else:
        muti_result = muti_predict(sent)
        print(muti_result['attr'])
        if muti_result['attr'] == '其他':
            print('muti:不是机器人画像属性')
