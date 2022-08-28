# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:19:16 2022

@author: haris.medjahed
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow_text as tf_text
from official.nlp.transformer.utils import tokenizer as tok
from tensorflow.keras.layers.experimental import preprocessing
from official.nlp.transformer import metrics, embedding_layer
from official.nlp.transformer import transformer_main, model_params
from official.utils.misc import distribution_utils
from official.nlp.transformer import transformer
from official.nlp.transformer import optimizer
import pickle


path_data = "data/"
path_model = "../../../tensorboard_data/"

COL_TYPES =  [0]+ [0] + ['a']+['a']
DEFAULTS = [tf.int64, tf.int64, tf.int64 ,tf.int64]
COL_NAMES = ["DocID","LineID","RawText","target"]




def tf_lower_and_split_punct(text):
  # Split accecented characters.
  #import unidecode
  #text = unidecode.unidecode(text)
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '^ a-z.!?,¿', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '.?!,¿', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END] '], separator=' ')
  return text


def processor(train_file):
    input_text_processor = preprocessing.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=None)

    target_processor = preprocessing.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=None)
    
    data = pd.read_csv(train_file, encoding='utf-8', sep=";")
    input_text_processor.adapt(data.RawText)
    target_processor.adapt(data.target)

    pickle.dump({'config': input_text_processor.get_config(),
             'weights': input_text_processor.get_weights()}
            , open("input_text_processor.pkl", "wb"))
    pickle.dump({'config': target_processor.get_config(),
             'weights': target_processor.get_weights()}
            , open("target_processor.pkl", "wb"))
    
    return



def preprocess_train(features):

    from_disk = pickle.load(open("input_text_processor.pkl", "rb"))
    input_text_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
    input_text_processor.set_weights(from_disk['weights'])
    
    from_disk = pickle.load(open("target_processor.pkl", "rb"))
    target_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
    target_processor.set_weights(from_disk['weights'])
    
    
    txt = input_text_processor(features.pop('RawText'))
    
    labels = target_processor(features.pop('target'))
    
    return (txt,labels), labels


def preprocess_test(features):


    txt = input_text_processor(features.pop('RawText'))
    
    labels = target_processor(features.pop('target'))
    
    return txt, labels


def input_fn(dataset_name, batch_size, preprocess):

    data = tf.data.experimental.make_csv_dataset(
        path_data+'%s.csv' % dataset_name,
        batch_size=batch_size,
        column_names=COL_NAMES,
        column_defaults=COL_TYPES,
        header=True,
        shuffle=True,
        shuffle_buffer_size=20000,
        sloppy=True,
        prefetch_buffer_size=1,
        use_quote_delim=True,
        field_delim=';'

    )

    data = data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.prefetch(buffer_size=1)

    return data


def input_fn_test(dataset_name, batch_size, preprocess):

    data = tf.data.experimental.make_csv_dataset(
        path_data+'%s.csv' % dataset_name,
        batch_size=batch_size,
        column_names=COL_NAMES,
        column_defaults=COL_TYPES,
        header=True,
        shuffle=False,
        shuffle_buffer_size=20000,
        sloppy=True,
        prefetch_buffer_size=1,
        use_quote_delim=True,
        field_delim=';'

    )

    data = data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.prefetch(buffer_size=1)

    return data

