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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf

"""
Masked LM: 
    1、在训练过程中作者随机mask 15%的token, 而不是把像cbow一样把每个词都预测一遍, 最终的损失函数只计算被mask掉的那个token.
    2、因为序列长度太大(512)会影响训练速度，所以90%的steps都是用seq_len=128来训练, 余下的10%步数训练512长度的输入
    3、Masked LM预训练阶段模型是不知道真正被mask的是哪个词的，所以模型对每个词都要关注
    
Next Sentence Prediction:
    1、作者特意说了语料的选取很重要, 要选用document-level的而不是sentence-level的，这样可以具备连续序列特征的能力
    
embedding --> N *【multi-head attention --> Add(Residual: 剩余的) &Norm--> 
                    Feed-Forward --> Add(ResidualResidual: 剩余的) &Norm】    
1. 配置类（BertConfig）
2. 获取词向量（Embedding_lookup）
3. 词向量的后续处理（embedding_postprocessor）
4. 构造attention_mask
5. 注意力层（attention layer）
6. Transformer
7. 函数入口（init）

"""


# 1、配置类 主要定义了BERT模型的一些默认参数，另外包括了一些文件处理函数。
class BertConfig(object):
    """
    Configuration for `BertModel`.
    配置类
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        type_vocab_size: next sentence prediction 任务中的Segment A 和 Segment B

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            词表大小

          hidden_size: Size of the encoder layers and the pooler layer.
            隐藏层神经元数

          num_hidden_layers: Number of hidden layers in the Transformer encoder.
            Transformer encoder中的隐藏层数

          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
            multi-head attention 的head数

          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
            encoder的“中间”隐层神经元数（例如feed-forward layer）

          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
            隐藏层激活函数

          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
            隐层dropout率

          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
            注意力部分的dropout

          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
            最大位置编码

          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
            token_type_ids的词典大小

          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
          initializer_range: truncated_normal_initializer初始化方法的stdev
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# 7、函数入口(init)
class BertModel(object):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    # 假设输入已经分词并且变成WordPiece的id了   # 输入是[2, 3]，表示batch=2，max_seq_length=3
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])

     # 第一个例子实际长度为3，第二个例子长度为2
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])

    # 第一个例子的3个Token中前两个属于句子1，第三个属于句子2
    # 而第二个例子的第一个Token属于句子1，第二个属于句子2(第三个是padding)
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    # 创建一个BertConfig，词典大小是32000，Transformer的隐单元个数是512
    # 8个Transformer block，每个block有8个Attention Head，全连接层的隐单元是1024
    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    # 创建BertModel
    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    # label_embeddings用于把512的隐单元变换成logits
    label_embeddings = tf.get_variable(...)

    # 得到[CLS]最后一层输出，把它看成句子的Embedding(Encoding)
    pooled_output = model.get_pooled_output()

    # 计算logits
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
             `BertConfig` 对象

          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
            bool 表示训练还是eval，是会影响dropout

          input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_ids: int32 Tensor  shape是[batch_size, seq_length]

          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            input_mask: (可选) int32 Tensor shape是[batch_size, seq_length]

          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (可选) int32 Tensor shape是[batch_size, seq_length]

          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
            (可选) bool
            如果True，使用矩阵乘法实现提取词的Embedding；否则用tf.embedding_lookup()
            对于TPU，使用前者更快，对于GPU和CPU，后者更快。

          scope: (optional) variable scope. Defaults to "bert".
            scope: (可选) 变量的scope。默认是"bert"

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
            ValueError: 如果config或者输入tensor的shape有问题就会抛出这个异常
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # 如果输入的input_mask为None，那么构造一个shape合适值全为1的input_mask，
        # 这表示输入都是”真实”的输入，没有padding的内容。
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        # 如果token_type_ids为None，那么构造一个shape合适并且值全为0的tensor，表示所有Token都属于第一个句子。
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                # 词的Embedding lookup
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                # 增加位置embeddings和token type的embeddings，然后是layer normalize和dropout。
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                # 把shape为[batch_size, seq_length]的2D mask变成
                # shape为[batch_size, seq_length, seq_length]的3D mask
                # 以便后向的attention计算，读者可以对比之前的Transformer的代码。
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # 接着用transformer_model函数构造多个Transformer
                # SubLayer然后stack在一起。得到的all_encoder_layers是一个list，
                # 长度为num_hidden_layers（默认12），每一层对应一个值。
                # 每一个值都是一个shape为[batch_size, seq_length, hidden_size]的tensor。

                # Run the stacked transformer.
                # 多个Transformer模型stack起来。
                # all_encoder_layers是一个list，长度为num_hidden_layers（默认12），每一层对应一个值。
                # 每一个值都是一个shape为[batch_size, seq_length, hidden_size]的tensor。
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            # 最后对self.sequence_output再加一个线性变换，得到的tensor仍然是[batch_size, hidden_size]。
            # `sequence_output`是最后一层的输出 shape = [batch_size, seq_length, hidden_size].
            self.sequence_output = self.all_encoder_layers[-1]

            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.

            # ‘pooler’部分将encoder输出【batch_size, seq_length, hidden_size】
            # 转成【batch_size, hidden_size】
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                # 取最后一层的第一个时刻[CLS]对应的tensor， 对于分类任务很重要
                # 从[batch_size, seq_length, hidden_size]变成[batch_size, hidden_size]
                # sequence_output[:, 0:1, :]得到的是[batch_size, 1, hidden_size]
                # 我们需要用squeeze把第二维去掉。
                # first_token_tensor是第一个Token([CLS])
                # 最后一层的输出，shape是[batch_size, hidden_size]。
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                # 然后再加一个全连接层，输出仍然是[batch_size, hidden_size]
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


# gelu激活函数
def gelu(x):
    """
    Gaussian Error Linear Unit.
       高斯误差线性单元

       与ReLU的不同：GELU为其按照输入的magnitude（等级）为inputs加权值的；
       ReLUs是根据inputs的sign（正负）来gate（加门限）的。

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.

      对公式中的正态分布的累积分布函数进行了tanh三阶多项式近似，取得了相较于swish用sigmoid近似更好的效果。
      累积分布函数是指随机变量X小于或等于x的概率
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


# 激活函数
def get_activation(activation_string):
    """
    Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


#  随机失活
def dropout(input_tensor, dropout_prob):
    """
    Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)

    return output


# 图层归一化。
def layer_norm(input_tensor, name=None):
    """
        Run layer normalization on the last dimension of the tensor.
        在张量的最后一个维度上运行图层归一化。
    """
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


# 网络标准化和失活
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """
        Runs layer normalization followed by dropout.
        运行图层规范化，然后随机失活。
    """
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)

    return output_tensor


# 创建截断正态分布初始化对象
def create_initializer(initializer_range=0.02):
    """
        Creates a `truncated_normal_initializer` with the given range.
        从截断的正态分布中输出随机值, 生成的值服从指定的均值和方差的正态分布，如果生成的值大于均值相差两个标准差的值则丢弃，重新选择
    """
    return tf.truncated_normal_initializer(stddev=initializer_range)


# 2、获取词向量 对于输入word_ids，返回embedding table。可以选用one-hot或者tf.gather()
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """
    Looks up words embeddings for id tensor.
    查找id张量的词嵌入。

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
        即word id 【batch_size, seq_length】

      vocab_size: int. Size of the embedding vocabulary.
        embedding词表

      embedding_size: int. Width of the word embeddings.
        embedding维度

      initializer_range: float. Embedding initialization range.
        embedding初始化范围 [标准差值]

      word_embedding_name: string. Name of the embedding table.
        embeddding table命名

      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.
        是否使用one-hot embedding

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
      返回:【batch_size, seq_length, embedding_size】
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    # 该函数默认输入的形状为【batch_size, seq_length, input_num】
    # 如果输入为2D的【batch_size, seq_length】，则扩展到【batch_size, seq_length, 1】

    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    # [batch_size*seq_length*input_num]
    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        # [batch_size * sequence_size, vocab_size]
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        # [batch_size*sequence_size, vocab_size] * [vocab_size, embedding_size]
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        # 按照索引取值
        output = tf.gather(embedding_table, flat_input_ids)

    # [batch_size, sequence_size, num_inputs]
    input_shape = get_shape_list(input_ids)

    #  转成:[batch_size, seq_length, num_inputs*embedding_size]
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])

    return (output, embedding_table)


# 3、词向量的后续处理
def embedding_postprocessor(input_tensor,  # [batch_size, seq_length, embedding_size]
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,  # 一般是2
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,  # 最大位置编码，必须大于等于max_seq_len
                            dropout_prob=0.1):
    """
    Performs various post-processing on a word embedding tensor.
    对词嵌入张量执行各种后处理。

    在Transformer论文中的position embedding是由sin/cos函数生成的固定的值，
    而在这里代码实现中是跟普通word embedding一样随机生成的，可以训练的。
    作者这里这样选择的原因可能是BERT训练的数据比Transformer那篇大很多，完全可以让模型自己去学习。

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
        输入tensor:[batch_size, seq_length, embedding_size].

      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        是否为`token_type_ids`添加嵌入。

      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
        （可选）int32张量，形状为[batch_size，seq_length]。 如果`use_token_type`为True，则必须指定。

      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
        “ token_type_ids”的词汇量。

      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
        token类型ID的嵌入表变量的名称。

      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
        是否为序列中每个token的位置添加位置嵌入。

      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
        位置嵌入的嵌入表变量的名称。

      initializer_range: float. Range of the weight initialization.
        权重初始化的范围。

      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
        最大序列长度可能是与该模型一起使用。 该长度可以比input_tensor的序列长度长，但不能短。

      dropout_prob: float. Dropout probability applied to the final output tensor.
        dropout率应用于最终输出张量。

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    # num_inputs * embedding_size
    width = input_shape[2]

    output = input_tensor

    # Segment position信息
    # Segment Embeddings 用来区分两种句子，因为预训练不光做LM还要用来预测两个句子为输入的分类任务
    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    # Position embedding信息
    if use_position_embeddings:
        # 这种情况下,如果对于每对(可能广播)元素 x[i],y[i], 我们有 x[i] <= y[i].如果 x 和 y 都是空的,该条件很容易满足.
        # 确保seq_length小于等于max_position_embeddings
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        # 指定上下文中构建的所有操作的控制依赖关系的上下文管理器.
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.

            # 这里position embedding是可学习的参数，[max_position_embeddings, width]
            # 但是通常实际输入序列没有达到max_position_embeddings
            # 所以为了提高训练速度，使用tf.slice取出句子长度的embedding
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            # word embedding之后的tensor是[batch_size, seq_length, width]
            # 因为位置编码是与输入内容无关，它的shape总是[seq_length, width]
            # 我们无法把位置Embedding加到word embedding上
            # 因此我们需要扩展位置编码为[1, seq_length, width]
            # 然后就能通过broadcasting加上去了。
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)

    return output


# 4、构造attention_mask 构造Mask矩阵
def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """
    Create 3D attention mask from a 2D tensor mask.

    比如调用它时的两个参数是:
        input_ids=[     # 两个样本
            [1,2,3,0,0],
            [1,3,5,6,1]
        ]
        input_mask=[
            [1,1,1,0,0],
            [1,1,1,1,1]
        ]

    表示这个batch有两个样本, 第一个样本长度为3(padding了2个0), 第二个样本长度为5.
    在计算self-Attention的时候每一个样本都需要一个Attention Mask矩阵,表示每一个时刻可以attend to 的范围,
    1表示可以attend to, 0表示是padding的（或者在机器翻译的Decoder中不能attend to未来的词）.
    对于上面的输入, 这个函数返回一个shape是[2, 5, 5]->(计算过程: [2, 5, 1] * [2, 1, 5] = [2, 5, 5])的tensor，
    分别代表来两个Attention Mask矩阵。
        [
            [1, 1, 1, 0, 0], #它表示第1个词可以attend to 3个词
            [1, 1, 1, 0, 0], #它表示第2个词可以attend to 3个词
            [1, 1, 1, 0, 0], #它表示第3个词可以attend to 3个词
            [1, 1, 1, 0, 0], #无意义，因为输入第4个词是padding的0
            [1, 1, 1, 0, 0]  #无意义，因为输入第5个词是padding的0
        ]

        [
            [1, 1, 1, 1, 1], # 它表示第1个词可以attend to 5个词
            [1, 1, 1, 1, 1], # 它表示第2个词可以attend to 5个词
            [1, 1, 1, 1, 1], # 它表示第3个词可以attend to 5个词
            [1, 1, 1, 1, 1], # 它表示第4个词可以attend to 5个词
            [1, 1, 1, 1, 1]	 # 它表示第5个词可以attend to 5个词
        ]

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    # [batch_size, from_seq_length, 1] * [batch_size, 1, to_seq_length] = [batch_size, from_seq_length, to_seq_length]

    mask = broadcast_ones * to_mask

    return mask


# 5、注意力层
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,

                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """
    Performs multi-headed attention from `from_tensor` to `to_tensor`.
    multi-head attention的实现

    ‘今天很美好‘
    当我们希望计算'今天'这个单词和其他单词的Attention-score的时候，则取出“今天”对应的
    Query, 与所有单词对应的key做点乘，结果经softmax归一化之后，与各个单词对应的value
    相乘，即可产生不同权重的结果，最终结果为output的加和。

    用`from_tensor`(作为Query)去attend to `to_tensor`(提供Key和Value)
    也就是用from的Query去乘以所有to的Key，softmax后得到weight，然后把所有to的Value加权求和起来。

    考虑key-query-value形式的attention，输入的from_tensor当做是query，
    to_tensor当做是key和value，当两者相同的时候即为self-attention。

    这个函数首先把`from_tensor`变换成一个"query" tensor，
    然后把`to_tensor`变成"key"和"value" tensors。
    总共有`num_attention_heads`组Query、Key和Value，
    每一个Query，Key和Value的shape都是[batch_size(8), seq_length(128), size_per_head(512/8=64)].

    然后计算query和key的内积并且除以size_per_head的平方根(8)。
    然后softmax变成概率，最后用概率加权value得到输出。
    因为有多个Head，每个Head都输出[batch_size, seq_length, size_per_head]，
    最后把8个Head的结果concat起来，就最终得到[batch_size(8), seq_length(128), size_per_head*8=512]

    实际上我们是把这8个Head的Query，Key和Value都放在一个Tensor里面的，
    因此实际通过transpose和reshape就达到了上面的效果。

    attention layer的主要流程：
    1、对输入的tensor进行形状校验，提取batch_size、from_seq_length 、to_seq_length
    2、输入如果是3d张量则转化成2d矩阵
    3、from_tensor作为query， to_tensor作为key和value，经过一层全连接层后得到query_layer、key_layer 、value_layer
    4、将上述张量通过transpose_for_scores转化成multi-head
    5、根据论文公式计算attention_score以及attention_probs（注意attention_mask的trick）：

    将得到的attention_probs与value相乘，返回2D或3D张量

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    “ from_tensor”中的每个时间步都遵循“ to_tensor”中的相应序列，并返回固定向量。

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
    该函数首先将from_tensor投影到“查询”张量中，将to_tensor投影到“键”和“值”张量中。
    这些（有效地）是长度为num_attention_heads`的张量的列表，
    其中每个张量的形状为[batch_size，seq_length，size_per_head]。

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
    然后，对查询张量 和 键张量 进行点乘和缩放。 这些被最大化以获得关注概率。
    然后，通过这些概率对值张量进行插值，然后将其连接回单个张量并返回。

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    实际上，多头注意力是通过转置和重塑而不是实际的单独张量来完成的。

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
        【batch_size, from_seq_length, from_width】

      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        【batch_size, to_seq_length, to_width】

      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
        【batch_size,from_seq_length, to_seq_length】
        值可以是0或者1，在计算attention score的时候，
        我们会把0变成负无穷(实际是一个绝对值很大的负数)，而1不变，
        这样softmax的时候进行exp的计算，前者就趋近于零，从而间接实现Mask的功能。


      num_attention_heads: int. Number of attention heads.
        attention head numbers

      size_per_head: int. Size of each attention head.
        每个head的大小

      query_act: (optional) Activation function for the query transform.
        query变换的激活函数

      key_act: (optional) Activation function for the key transform.
        key变换的激活函数

      value_act: (optional) Activation function for the value transform.
        value变换的激活函数

      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
        attention层的dropout

      initializer_range: float. Range of the weight initializer.
        初始化取值范围

      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
        是否返回2d张量。如果True，输出形状【batch_size*from_seq_length, num_attention_heads*size_per_head】
        如果False，输出形状【batch_size, from_seq_length, num_attention_heads*size_per_head】

      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
        如果输入是3D的那么batch就是第一维，但是可能3D的压缩成了2D的，所以需要告诉函数batch_size

      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
        如果输入是3D的那么batch就是第一维，但是可能3D的压缩成了2D的，所以需要告诉函数batch_size

      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.
        如果输入是3D的那么batch就是第一维，但是可能3D的压缩成了2D的，所以需要告诉函数batch_size
    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    # reshape并且转置维度
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])

        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if batch_size is None or from_seq_length is None or to_seq_length is None:
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here [为了方便备注shape，采用以下简写]:
    #   B = batch size (number of sequences) 默认配置是8
    #   F = `from_tensor` sequence length  默认配置是128
    #   T = `to_tensor` sequence length 默认配置是128
    #   N = `num_attention_heads` 默认配置是12
    #   H = `size_per_head` 默认配置是64

    # 把from_tensor和to_tensor压缩成2D张量
    from_tensor_2d = reshape_to_matrix(from_tensor)  # 【B*F, hidden_size】
    to_tensor_2d = reshape_to_matrix(to_tensor)  # 【B*T, hidden_size】

    #  将from_tensor输入全连接层得到query_layer
    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        inputs=from_tensor_2d,
        units=num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        inputs=to_tensor_2d,
        units=num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        inputs=to_tensor_2d,
        units=num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]  query_layer转成多头：[B*F, N*H]==>[B, F, N, H]==>[B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]  key_layer转成多头：[B*T, N*H] ==> [B, T, N, H] ==> [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # 将query与key做点积，然后做一个scale
    # attention scores.
    # `attention_scores` = [B, N, F, T]   [B, N, F, H] * [B, N, T, H].T = [B, N, F, T]
    # 最后两维[F(128), T(128)]表示from的128个时刻attend to到to的128个score。
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    # 两个矩阵中对应元素各自相乘
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.

        # 如果attention_mask里的元素为1，则通过下面运算有（1-1）*-10000，adder就是0
        # 如果attention_mask里的元素为0，则通过下面运算有（1-0）*-10000，adder就是-10000
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # 我们最终得到的attention_score一般不会很大，
        # 所以上述操作对mask为0的地方得到的score可以认为是负无穷
        attention_scores += adder

    # 负无穷经过softmax之后为0，就相当于mask为0的位置不计算attention_score
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # 对attention_probs进行dropout，这虽然有点奇怪，但是Transforme原始论文就是这么做的
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    # 加权计算
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


# 6、Transformer
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """
    Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        float Tensor，shape为[batch_size, seq_length, hidden_size]

      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
        (可选) int32 Tensor，shape [batch_size, seq_length, seq_length], 1表示可以attend to，0表示不能。

      hidden_size: int. Hidden size of the Transformer.
        Transformer隐单元个数

      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        有多少个SubLayer

      num_attention_heads: int. Number of attention heads in the Transformer.
        Transformer Attention Head个数。

      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
        全连接层的隐单元个数

      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
        函数. 全连接层的激活函数。

      hidden_dropout_prob: float. Dropout probability for the hidden layers.
        float. Self-Attention层残差之前的Dropout概率

      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
        float. attention的Dropout概率

      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
        初始化范围(truncated normal的标准差)

      do_return_all_layers: Whether to also return all layers or just the final
        layer.
        返回所有层的输出还是最后一层的输出。

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.
      如果do_return_all_layers True，返回最后一层的输出，是一个Tensor，
      shape为[batch_size, seq_length, hidden_size]；

      否则返回所有层的输出，是一个长度为num_hidden_layers的list，
      list的每一个元素都是[batch_size, seq_length, hidden_size]。

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    # 这里注意，因为最终要输出hidden_size， 我们有num_attention_head个区域，
    # 每个head区域有size_per_head多的隐层
    # 所以有 hidden_size = num_attention_head * size_per_head
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    # 因为encoder中有残差操作，所以需要shape相同
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    # reshape操作在CPU/GPU上很快，但是在TPU上很不友好
    # 所以为了避免2D和3D之间的频繁reshape，我们把所有的3D张量用2D矩阵表示
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    # 如果有多个head，将他们拼接起来
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                # 对attention的输出进行线性映射, 目的是将shape变成与input一致
                # 然后dropout+residual+norm
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            # 对feed-forward层的输出使用线性变换变回‘hidden_size’
            # 然后dropout + residual + norm
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


# 获取张量形状的列表
def get_shape_list(tensor, expected_rank=None, name=None):
    """
    Returns a list of the shape of tensor, preferring static dimensions.
    返回张量形状的列表，首选静态尺寸。

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    # TensorShape([Dimension(2), Dimension(3), Dimension(4)]) -> [2, 3, 4]
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)  # tensor 类型
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


# 将input_tensor压缩成2D张量
def reshape_to_matrix(input_tensor):
    """
        Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).
        将> =等级2张量重整为等级2张量（即矩阵）。
    """
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])

    return output_tensor


# 将等级2张量重整为原始等级> = 2张量。
def reshape_from_matrix(output_tensor, orig_shape_list):
    """
        Reshapes a rank 2 tensor back to its original rank >= 2 tensor.
        将等级2张量重整为原始等级> = 2张量。
    """
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


# 判断tensor的秩是否等于expected_rank
def assert_rank(tensor, expected_rank, name=None):
    """
    Raises an exception if the tensor rank is not of the expected rank.
    如果张量等级不属于期望等级，则引发异常。

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):  # (int, long)
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
