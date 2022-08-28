# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:37:21 2022

@author: haris.medjahed
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from official.nlp.transformer import model_params
import pickle
import training


path_data = "data/"
path_model = "/model_train"

COL_TYPES =  [0]+ [0] + ['a']+['a']
DEFAULTS = [tf.int64, tf.int64, tf.int64 ,tf.int64]
COL_NAMES = ["DocID","LineID","RawText","target"]


## Fonctions

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
        max_tokens= None)

    target_processor = preprocessing.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens= None)
    
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

    
    txt = input_text_processor(features.pop('RawText'))
    
    labels = input_text_processor(features.pop('target'))
    
    return (txt,labels), labels




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






## Code



#processor("data/Extrait500000_corpus_train_code_target.csv") # Création des tokenizers et adaptation sur le texte d'entrainement

from_disk = pickle.load(open("pickle/full_adapt_processor.pkl", "rb"))
input_text_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
input_text_processor.set_weights(from_disk['weights'])

from_disk = pickle.load(open("pickle/target_processor.pkl", "rb"))
target_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
target_processor.set_weights(from_disk['weights'])

## Dataset

train_ds = input_fn(dataset_name = "Full_corpus_train_code_target",batch_size =32, preprocess  = preprocess_train)
eval_ds = input_fn(dataset_name = "Full_corpus_val_code_target",batch_size =32, preprocess  = preprocess_train)

## Paramètres du modèle

params= model_params.TINY_PARAMS
params["batch_size"] = params["default_batch_size"] = 16
params["use_synthetic_data"] = True
params["hidden_size"] = 12
params["num_hidden_layers"] = 1
params["filter_size"] = 14
params["num_heads"] = 3
params["extra_decode_length"] = 2
params["beam_size"] = 3
params["dtype"] = tf.float32
params['input_vocab_size'] = len(input_text_processor.get_vocabulary())
params['vocab_size'] = len(input_text_processor.get_vocabulary())
params['default_batch_size'] = 80*3

drop_out_val = 0.3
params['layer_postprocess_dropout'] = drop_out_val
params['attention_dropout'] = drop_out_val
params['relu_dropout'] = drop_out_val
model_dir = path_model

training.train_model(train_ds,eval_ds,params, model_dir)