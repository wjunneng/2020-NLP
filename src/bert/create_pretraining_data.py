# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
# 这个文件的作用就是将原始输入语料转换成模型预训练所需要的数据格式TFRecoed。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
from src.bert import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

# 输出tf-record文件
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# 是否转小写
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# 是否使用整个单词掩码而不是每个WordPiece掩码
flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

# max_predictions_per_seq： 一个句子里最多有多少个[MASK]标记
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

# dupe_factor： 重复参数，即对于同一个句子，我们可以设置不同位置的【MASK】次数。
# 比如对于句子Hello world, this is bert.，为了充分利用数据，第一次可以mask成Hello [MASK], this is bert.，
# 第二次可以变成Hello world, this is [MASK[.
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

# masked_lm_prob： 多少比例的Token被MASK掉
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

# short_seq_prob： 长度小于“max_seq_length”的样本比例。
# 因为在fine-tune过程里面输入的target_seq_length是可变的（小于等于max_seq_length），
# 那么为了防止过拟合也需要在pre-train的过程当中构造一些短的样本。
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


# 训练样本的类
class TrainingInstance(object):
    """
    A single training instance (sentence pair).

    __repr__和__str__这两个方法都是用于显示的，__str__是面向用户的，而__repr__面向程序员。
    __repr__:不管直接输出对象还是使用print打印对象，输出的格式都是按照__repr的来。
    __str__:只用使用print打印对象的是否才使用__str__定义的格式。

    假设原始两个句子为：”it is a good day”和”I want to go out”，那么处理后的TrainingInstance可能为：

    1. tokens = ["[CLS], "it", "is" "a", "[MASK]", "day", "[SEP]", "I", "apple", "to", "go", "out", "[SEP]"]
    2. segment_ids=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    3. is_random_next=False

    注意：tokens已经处理过了，good被替换成[MASK]，而want被替换成apple，而to还是被替换成它自己， 原因前面的理论部分已经介绍过了。
    因此根据masked_lm_positions、masked_lm_labels和tokens是可以恢复出原始(分词后的)句子的。

    4. masked_lm_positions=[4, 8, 9]
       表示Mask后为["[CLS], "it", "is" "a", "[MASK]", "day", "[SEP]", "I", "[MASK]", "to", "go", "out", "[SEP]"]
    5. masked_lm_labels=["good", "want", "to"]

    is_random_next表示这两句话是有关联的，预测句子关系的分类器应该把这个输入判断为1。
    masked_lm_positions记录哪些位置被Mask了，而masked_lm_labels记录被Mask之前的词。
    """

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


# 创建训练实例
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    # all_documents是list的list，第一层list表示document，
    # 第二层list表示document里的多个句子。
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    # 在源码包中Google提供了一个实例训练样本输入（「sample_text.txt」），输入文件格式为：
    # 输入文件格式：
    # (1) 每行一个句子。这应该是实际的句子，不应该是整个段落或者段落的随机片段(span)，因为我们需
    # 要使用句子边界来做下一个句子的预测。
    # (2) 文档之间有一个空行。我们会认为同一个文档的相邻句子是有关系的。

    # 下面的代码读取所有文件，然后根据空行切分Document
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                # 空行表示文档分割
                # 空行表示旧文档的结束和新文档的开始。
                if not line:
                    # 添加新文档
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    # 删除空文档
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    # dupe_factor 重复设置mask的次数 重复dupe_factor次
    for _ in range(dupe_factor):
        # 遍历所有文档
        for document_index in range(len(all_documents)):
            # 从一个文档(下标为document_index)里抽取多个TrainingInstance
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


# 实现从一个文档中抽取多个训练实例
def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """
    Creates `TrainingInstance`s for a single document.

    上面的函数会调用create_instances_from_document来从一个文档里抽取多个训练数据(TrainingInstance)。
    普通的语言模型只要求连续的字符串就行，通常是把所有的文本(比如维基百科的内容)拼接成一个很大很大的文本文件，
    然后训练的时候随机的从里面抽取固定长度的字符串作为一个”句子”。但是BERT要求我们的输入是一个一个的Document，
    每个Document有很多句子，这些句子是连贯的真实的句子，需要正确的分句，而不能随机的(比如按照固定长度)切分句子

    w11,w12,.....,
    w21,w22,....
    wn1,wn2,....

    那么算法首先找到一个chunk，它会不断往chunk加入一个句子的所有Token，使得chunk里的token数量大于等于target_seq_length。
    通常我们期望target_seq_length为max_num_tokens(128-3)，这样padding的尽量少，训练的效率高。
    但是有时候我们也需要生成一些短的序列，否则会出现训练与实际使用不匹配的问题。

    找到一个chunk之后，比如这个chunk有5个句子，那么我们随机的选择一个切分点，比如3。
    把前3个句子当成句子A，后两个句子当成句子B。这是两个句子A和B有关系的样本(is_random_next=False)。
    为了生成无关系的样本，我们还以50%的概率把B用随机从其它文档抽取的句子替换掉，
    这样就得到无关系的样本(is_random_next=True)。如果是这种情况，后面两个句子需要放回去，以便在下一层循环中能够被再次利用。

    """
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    # 为[CLS], [SEP], [SEP]预留三个空位
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.

    # 我们通常希望Token序列长度为最大的max_seq_length，否则padding后的计算是无意义的，浪费计
    # 算资源。但是有的时候我们有希望生成一些短的句子，因为在实际应用中会有短句，如果都是
    # 长句子，那么就很容易出现Mismatch，所有我们以short_seq_prob == 0.1 == 10%的概率生成
    # 短句子。

    target_seq_length = max_num_tokens
    # short_seq_prob： 长度小于“max_seq_length”的样本比例
    # 以short_seq_prob的概率随机生成（2~max_num_tokens）的长度
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    # 我们不只是将文档中的所有标记连接成一个长序列，并选择一个任意的分割点，因为这会使下一个句子的预测任务变得很容易。
    # 相反，我们根据用户输入提供的实际“句子”将输入分为“ A”和“ B”两段。

    # 我们不能把一个文档的所有句子的Token拼接起来，然后随机地选择两个片段。
    # 因为这样很可能这两个片段是同一个句子(至少很可能第二个片段的开头和第一个片段的结尾是同一个
    # 句子)，这样预测是否相关句子的任务太简单，学习不到深层的语义关系。
    # 这里我们使用"真实"的句子边界。

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        # segment: 当前句子
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        # 将句子加入到current_chunk直到加完，或者达到限制长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                # a_end 是第一个句子A结束的下标
                a_end = 1
                # 随机选取切分边界
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False

                # 0.5概率A - B两个句子连续，0.5
                # 概率随机选择B A使用A embedding，B使用B embedding
                # A和B总长度最大为512 tokens
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    # 随机的挑选另外一篇文档的随机开始的句子
                    # 但是理论上有可能随机到的文档就是当前文档，因此需要一个while循环
                    # 这里只while循环10次，理论上还是有重复的可能性，但是我们忽略
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break
                    # 随机挑选的文档
                    random_document = all_documents[random_document_index]
                    # 随机选择开始句子
                    random_start = rng.randint(0, len(random_document) - 1)
                    # 把Token加到tokens_b里，如果Token数量够了(target_b_length)就break。
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.

                    # 之前我们虽然挑选了len(current_chunk)个句子，但是a_end之后的句子替换成随机的其它
                    # 文档的句子，因此我们并没有使用a_end之后的句子，因此我们修改下标i，使得下一次循环
                    # 可以再次使用这些句子(把它们加到新的chunk里)，避免浪费。
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                # 真实的下一句
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                # 如果太多了，随机去掉一些。
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


# create_masked_lm_predictions来随机的选择某些Token，把它变成[MASK]
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    # 首先找到可以被替换的下标，[CLS]和[SEP]是不能用于MASK的。
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        # [CLS] 和 [SEP] 不能用于MASK
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    # 需要被模型预测的Token个数：min(max_predictions_per_seq(20)，实际Token数*15%)
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    # 随机的挑选num_to_predict个需要预测的Token
    # 因为cand_indexes打散过，因此顺序的取就行
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


# 这个函数很简单，如果两个序列的长度小于max_length，那么不用truncate，
# 否则在tokens_a和tokens_b中选择长的那个序列来pop掉最后面的那个Token，
# 这样的结果是使得两个Token序列一样长(或者最多a比b多一个Token)。
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    """
    1、构造tokenizer对输入语料进行分词处理（Tokenizer部分会在后续说明）
    2、经过create_training_instances函数构造训练instance
    3、调用write_instance_to_example_files函数以TFRecord格式保存数据
    下面我们一一解析这些函数。
    :param _:
    :return:
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        # tf.gfile.Glob: 查找匹配pattern的文件并以列表的形式返回，
        # filename可以是一个具体的文件名，也可以是包含通配符的正则表达式。
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    # 以tf-record格式写到output_files
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
