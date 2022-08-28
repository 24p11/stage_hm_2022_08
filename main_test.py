# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:50:07 2022

@author: haris.medjahed
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from official.nlp.transformer import model_params
from official.nlp.transformer import transformer

import pickle


path_data = "data/"
path_model = "/model_train"

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



def preprocess_test(features):


    txt = input_text_processor(features.pop('RawText'))
    
    labels = input_text_processor(features.pop('target'))
    
    return txt, labels



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


def isin(padded_outputs,padded_labels):
    """Percentage of time logits contains labels on non-0s."""
    with tf.name_scope("isin"):
        tile_multiple = tf.shape(padded_labels)[-1]
        tiled_outputs = tf.tile(tf.expand_dims(padded_outputs, axis=-1), multiples=[1, 1, tile_multiple])
        tiled_labels = tf.reshape(tf.tile(padded_labels, multiples=[1, tile_multiple]), [-1, tile_multiple, tile_multiple])
        equal = tf.equal(tiled_outputs, tiled_labels)
        any = tf.reduce_any(equal, axis=-1)
        return any


def true_false_positives(logits, labels):
    labels = labels.numpy()
    size = max(np.shape(labels)[1],np.shape(logits)[1])
    
    if np.shape(logits)[1]<size: #On ajoute des zeros si la prediction n'est pas assez longue pour correspondre aux labels
             result = np.zeros((logits.shape[0],size))
             result[:logits.shape[0],:logits.shape[1]] = logits
             logits = result
    if np.shape(labels)[1]<size: #On ajoute des zeros si la prediction n'est pas assez longue pour correspondre aux labels
             result = np.zeros((labels.shape[0],size))
             result[:labels.shape[0],:labels.shape[1]] = labels
             labels = result         
             
             
    logits = np.where(logits==1,0,logits)         
    logits = np.where(logits==2,0,logits)
    logits = np.where(logits==3,0,logits)
    labels = np.where(labels==1,0,labels)
    labels = np.where(labels==2,0,labels)
    labels = np.where(labels==3,0,labels)
    padded_outputs = tf.cast(logits, tf.int32)
    padded_labels = tf.cast(labels, tf.int32)

    weights_outputs = tf.logical_and(tf.not_equal(padded_outputs, 0), tf.not_equal(padded_outputs, 1))
    weights_labels = tf.logical_and(tf.not_equal(padded_labels, 0), tf.not_equal(padded_labels, 1))
    
    true_p = tf.logical_and(isin(padded_labels, padded_outputs), weights_labels, weights_outputs)
    out_in_lab = tf.logical_or(isin(padded_labels, padded_outputs), tf.logical_not(weights_outputs))
    lab_in_out = tf.logical_or(isin(padded_labels, padded_outputs), tf.logical_not(weights_labels))
    
    true_positives = tf.cast(true_p, tf.float32)
    false_positives = tf.cast(tf.logical_not(out_in_lab), tf.float32)
    false_negatives = tf.cast(tf.logical_not(lab_in_out), tf.float32)

    return tf.reduce_sum(true_positives), tf.reduce_sum(false_positives), tf.reduce_sum(false_negatives)


def my_metrics(logits, labels):

    true_positives, false_positives, false_negatives = true_false_positives(logits, labels)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_measure = 2 * precision * recall / (precision + recall)
    jaccard = true_positives / (true_positives + false_positives + false_negatives)


    return precision, recall, f_measure, jaccard


from_disk = pickle.load(open("pickle/full_ADAPT_processor.pkl", "rb"))
input_text_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
input_text_processor.set_weights(from_disk['weights'])

from_disk = pickle.load(open("pickle/target_processor.pkl", "rb"))
target_processor = preprocessing.TextVectorization.from_config(from_disk['config'])
target_processor.set_weights(from_disk['weights'])




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








dataset = input_fn_test(dataset_name = "Full_corpus_test_code_target",batch_size =32, preprocess  = preprocess_test
                  )
iterator = iter(dataset)


data = pd.read_csv("data/Full_corpus_test_code_target.csv", encoding='utf-8', sep=";")
Nb_ligne = len(data)
Repet = int(Nb_ligne/32)+1

## Importation du modèle

Modele_test = transformer.create_model(params, is_train=False)
Modele_test.load_weights("Model_poids/FULL_20000token/Wed_24_Aug_2022_00_14_57/")

Precision = 0
Recall = 0
F_measure = 0
Tp = 0
Fp =0
Fn = 0

for i in range(Repet):
        batch = iterator.get_next()
        batch_predict = Modele_test.predict(batch[0])
        Metrics = my_metrics(batch_predict[0],batch[1])
        TFP = true_false_positives(batch_predict[0], batch[1])
        Tp += float(TFP[0])
        Fp += float(TFP[1])
        Fn += float(TFP[2])
        Precision += float(Metrics[0])
        Recall += float(Metrics[1])
        F_measure += float(Metrics[2])
        
Precision = Precision/Repet
Recall = Recall/Repet
F_measure = F_measure/Repet

Prec = Tp /(Tp+Fp)
Rec = Tp / (Tp+Fn)
F_mea = 2 * Prec * Rec/(Prec+Rec)


print("La précision du modèle est de : ", Precision, " ou ", Prec)
print("Le recall du modèle est de : ", Recall, " ou ", Rec)
print("La f_measure du modèle est de : ", F_measure, " ou ", F_mea)



# from sklearn.metrics import accuracy_score

# def precision(prediction, labels):
#     labels = labels.numpy()
#     prediction = np.where(prediction==2,0,prediction)
#     prediction = np.where(prediction==3,0,prediction)
#     labels = np.where(labels==2,0,labels)
#     labels = np.where(labels==3,0,labels)
#     size = np.shape(labels)[1]
#     if np.shape(prediction)[1]<size: #On ajoute des zeros si la prediction n'est pas assez longue pour correspondre aux labels
#         result = np.zeros(labels.shape)
#         result[:prediction.shape[0],:prediction.shape[1]] = prediction
#         prediction = result
#     y_pred = prediction[:,:size] #On réduit la prédiction à la taille des labels sinon, et on exclut le [START]
#     precision = 0
#     for i in range(labels.shape[0]):
#         k = 1
#         for j in range(size):
#             if labels[i,k] != 0 :
#                 k+=1
#         print("La précision de la phrase", i+1,"est de :", accuracy_score(labels[i,1:k], y_pred[i,1:k]))
#         precision += accuracy_score(labels[i,1:k], y_pred[i,1:k])
#     precision_moyenne = precision/labels.shape[0]
#     print("La précision moyenne du modèle sur ce batch est de : ", precision_moyenne)
#     return


# precision(batch_predict[0],batch[1])


for i in range(len(batch_predict[0])):
    vocab = input_text_processor.get_vocabulary()
    print(i)
    print("")
    print("RawText : " + " ".join([vocab[each] for each in tf.squeeze(batch[0][i])]))
    print("")
    print("Predict : " + " ".join([vocab[each] for each in tf.squeeze(batch_predict[0][i])]))
    print("")
    print("Target : " + " ".join([vocab[each] for each in tf.squeeze(batch[1][i])]))
    print("")
    print("")

    
batch = iterator.get_next()
batch_predict = Modele_test.predict(batch[0])