from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import abc

import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
# import tf.contrib.layers.fully_connected as fully_connect_layer
from . import model
import tensorflow.contrib.distributions as distributions
import tensorflow as tf
from . import modules
import os, codecs
from tqdm import tqdm
from . import classifier



class AttentionAll(classifier.Classifier):


  def _build_encoder(self, hparams):
    iterator = self.iterator
    source = iterator.source

    #make sure the dimension of source
    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype

      ##change the size of embedding
      embedding_transformer = tf.get_variable('embedding_transformer', shape=[300,hparams.num_units])
      self.embedding_encoder = tf.matmul(self.embedding_encoder, embedding_transformer)

      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
        self.embedding_encoder, source)


      #position embedding
      self.enc = encoder_emb_inp + modules.positional_encoding(source,
                                            num_units=hparams.num_units,
                                            zero_pad=False,
                                            scale=False,
                                            scope="enc_pe")

      ## Dropout
      self.enc = tf.layers.dropout(self.enc,
                                   rate=hparams.dropout,
                                   training=tf.convert_to_tensor(self.mode ==tf.contrib.learn.ModeKeys.TRAIN))

      for i in range(hparams.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ### Multihead Attention
          self.enc = modules.multihead_attention(queries=self.enc,
                                         keys=self.enc,
                                         num_units=hparams.num_units,
                                         num_heads=hparams.num_heads,
                                         dropout_rate=hparams.dropout,
                                         is_training=self.mode==tf.contrib.learn.ModeKeys.TRAIN,
                                         causality=False)

          ### Feed Forward
          self.enc = modules.feedforward(self.enc, num_units=[4*hparams.num_units, hparams.num_units])


  def build_graph(self, hparams, scope = None):
    """Subclass must implement this method.

      Creates classifier. The feature comes from sentence representation
      Args:
        hparams: Hyperparameter configurations.
        scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

      Returns:
        A tuple of the form (logits, loss),
    where:
      logits: float32 Tensor [batch_size x num_classes].
      loss: the total loss / batch_size.

    Raises
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32
    num_gpus = hparams.num_gpus

    with tf.variable_scope(scope or "classifier", dtype=dtype):
      #Encoder

      self._build_encoder(hparams)
      self.enc = tf.transpose(self.enc, [1,0,2])
      encoder_state = self.attention(self.enc)
      ## Note tha when mode == infer, the following function return scalar logits,
      ## samplie_id, final_context_state

      ## Loss



      logits = self.fully_connect_layer(encoder_state, hparams.num_class)
      # logits = tf.Print(logits,[logits,"shape of logits",tf.shape(logits)]) #128 times 15
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(0, num_gpus)):
          labels = self.iterator.target_input
          loss = self._compute_loss(logits, labels, hparams.num_class, hparams.label_smoothing)
          vars = [v for v in tf.trainable_variables() if not 'bias' in v.name]
          if hparams.regularization == 'l2':
            loss_ = tf.add_n([ tf.nn.l2_loss(v) for v in vars]) * hparams.regu_weight
          elif hparams.regularization == 'l1':
            l1_regularizer = tf.contrib.layers.l1_regularizer(
              scale=hparams.regu_weight, scope=None )
            loss_= tf.contrib.layers.apply_regularization(l1_regularizer, vars)
          if hparams.ptb_we:
            loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

          loss += loss_
      else:
        loss = None
        labels = None
      #return None is for compatability with other model
      sample_id = [tf.argmax(input=logits, axis=1)]
      sample_id = tf.transpose(sample_id)
      return logits, loss, labels, sample_id, sample_id
