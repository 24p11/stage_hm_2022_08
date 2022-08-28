# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:36:01 2022

@author: haris.medjahed
"""


import tensorflow as tf
from official.nlp.transformer import transformer
from official.nlp.transformer import optimizer
import time




def create_optimizer(params):
    lr_schedule = optimizer.LearningRateSchedule(
        params['learning_rate'], params['hidden_size'],
        params['learning_rate_warmup_steps']
    )

    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        params['optimizer_adam_beta1'],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"]
    )

    return opt

def train_model(train_ds,eval_ds,params, model_dir):
       
    model = transformer.create_model(params, is_train=True)
    opt = create_optimizer(params)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
    board = tf.keras.callbacks.TensorBoard(
        log_dir=model_dir, write_graph=False, update_freq=500
    )
    model.compile(opt)

    model.fit(
        x = train_ds,
        epochs=1,
        steps_per_epoch=250000,
        verbose=1,
        validation_data=eval_ds,
        validation_steps=100,
        callbacks=board)
    model.save_weights("Model_poids/FULL_20000token/"+time.strftime("%a_%d_%b_%Y_%H_%M_%S/", time.localtime()))
    for i in range(5):
        model.fit(
            x = train_ds,
            epochs=1,
            steps_per_epoch=250000,
            verbose=1,
            validation_data=eval_ds,
            validation_steps=100,
            callbacks=board)
        model.save_weights("Model_poids/FULL_20000token/"+time.strftime("%a_%d_%b_%Y_%H_%M_%S/", time.localtime()))
    
