#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from datetime import datetime


import pdb

try:
    import ujson as json
except:
    import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import yaml

from src.utils import get_class_weights,  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import GAT_model,  RNN_model, TransFormer


handle_flags()



def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)
    #tf.config.run_functions_eagerly(True)#TODO remove

    # Load data
    # cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader) #TODO
    if FLAGS.dataset=='AF':
        data_prefix = '{}/AF_PTM_'.format(
            FLAGS.data_path) 
    else:
        data_prefix = '{}/PTM_'.format(
                FLAGS.data_path) 


    train_dat_aug = PTMDataGenerator(data_prefix+'train.json', FLAGS, shuffle=True,ind=None, eval=False)
    unique_labels = train_dat_aug.unique_labels
    val_dat_aug = PTMDataGenerator(data_prefix+'val.json', FLAGS, shuffle=True,ind=None, eval=True)
    test_dat_aug = PTMDataGenerator(data_prefix+'test.json', FLAGS, shuffle=True,ind=None, eval=True)
    val_dat_aug.update_unique_labels(unique_labels)
    test_dat_aug.update_unique_labels(unique_labels)

    # setting up

    # class_weights = get_class_weights(train_data, val_data, test_data, unique_labels) if FLAGS.class_weights else None

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # metrics = [CategoricalTruePositives(13,batch_size=FLAGS.batch_size)]#,tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()
    metrics = []
    # Build model
    if FLAGS.model=='proteinbert':
        next
        # if FLAGS.binary and FLAGS.multilabel:
        #     # for first trained on multilabel then binary
        #     model = ProteinBert(optimizer, loss_object, unique_labels, FLAGS.learning_rate,False, FLAGS.multilabel)
        #     model.create_model( train_data,  FLAGS.seq_len, \
        #         freeze_pretrained_layers=False, binary=FLAGS.binary, graph=FLAGS.graph, n_gcn=FLAGS.n_gcn)
        # else:
        #     model = ProteinBert(optimizer, loss_object, unique_labels, FLAGS.learning_rate,FLAGS.binary, FLAGS.multilabel)
        #     model.create_model( train_data,  FLAGS.seq_len, \
        #         freeze_pretrained_layers=False, binary=FLAGS.binary, graph=FLAGS.graph, n_gcn=FLAGS.n_gcn)            
    elif FLAGS.model=='RNN':
        if FLAGS.binary and FLAGS.multilabel:
            model = RNN_model(optimizer, loss_object, FLAGS.learning_rate)
            model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6,metrics, False, \
                FLAGS.multilabel, FLAGS.graph, n_lstm=FLAGS.n_lstm, n_gcn=FLAGS.n_gcn)
        else:
            model = RNN_model(optimizer, loss_object, FLAGS.learning_rate)
            model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6,metrics, FLAGS.binary, \
                FLAGS.multilabel, FLAGS.graph, n_lstm=FLAGS.n_lstm, n_gcn=FLAGS.n_gcn)
    elif FLAGS.model=='GAT':
        if FLAGS.binary and FLAGS.multilabel:
            model = GAT_model(optimizer, loss_object, FLAGS.learning_rate)
            model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6, False, n_gcn=FLAGS.n_gcn)
        else:
            model = GAT_model(optimizer, loss_object, FLAGS.learning_rate)
            model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6, FLAGS.binary, n_gcn=FLAGS.n_gcn)
    elif FLAGS.model=='Transformer':
        model = TransFormer(FLAGS.model,optimizer, loss_object, FLAGS.learning_rate, d_model=128, num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512,\
            rate=0.1,binary=False, unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
    if FLAGS.multilabel:# multi-label      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        checkpoint_path = './saved_model/'+current_time
        os.system('mkdir '+ checkpoint_path)
 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1)
        training_callbacks = [
            #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
            keras.callbacks.EarlyStopping(monitor='loss',patience = 2, restore_best_weights = True),
            cp_callback
        ] 
        model.train( train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels, lr = FLAGS.learning_rate, callbacks=training_callbacks,graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
        logging.info('------------------evaluate---------------------' )
        model.model = tf.keras.models.load_model(checkpoint_path)
        AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, FLAGS.graph, num_cont=FLAGS.fill_cont)
        for u in unique_labels:
            print('%.3f'%PR_AUC[u])
        for u in unique_labels:
            print(u)
            print(confusion_matrixs[u])
        if not FLAGS.binary:
            if FLAGS.graph:
                model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model+'_'+'_multi_graph_'+str(FLAGS.seq_len) +\
                    '_'+str(FLAGS.fill_cont) + 'no_pdb' + str(FLAGS.no_pdb)
            else:
                model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model+'_'+'_multi_'+str(FLAGS.seq_len)
            model.model.save(model_name)
        os.system('rm -r '+checkpoint_path)

    if FLAGS.binary:# multi-label
        # initiate binary with weight from multilabel
        if FLAGS.model=='proteinbert':
            next
            # b_model = ProteinBert(optimizer, loss_object, unique_labels, FLAGS.learning_rate,FLAGS.binary, False)
            # b_model.create_model( train_data,  FLAGS.seq_len, \
            #     freeze_pretrained_layers=False, binary=FLAGS.binary, graph=FLAGS.graph, n_gcn=FLAGS.n_gcn) 
        elif FLAGS.model=='RNN':
            b_model = RNN_model(optimizer, loss_object, FLAGS.learning_rate)
            b_model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6,metrics, FLAGS.binary, \
                False, FLAGS.graph, n_lstm=FLAGS.n_lstm, n_gcn=FLAGS.n_gcn)
        elif FLAGS.model=='GAT':
            b_model = GAT_model(optimizer, loss_object, FLAGS.learning_rate)
            b_model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6, FLAGS.binary, n_gcn=FLAGS.n_gcn)
        elif FLAGS.model=='Transformer':
            b_model = TransFormer(FLAGS.model, optimizer, loss_object, FLAGS.learning_rate, d_model=128, num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512,\
            rate=0.1,binary=FLAGS.binary, unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
            b_model.create_model(FLAGS.seq_len, graph=FLAGS.graph) 
        #for layer in model.model.layers:
        for li in range(len(model.model.layers)):
            layer = model.model.layers[li]  
            if layer.name not in ['my_last_dense', 'reshape',]:
                b_model.model.get_layer(index=li).set_weights(layer.get_weights())


        # temp_y = train_data.Y.reshape((train_data.Y.shape[0], FLAGS.seq_len, -1))
        # sort_ind = np.argsort(-np.array([np.sum(temp_y[:,:,i]) for i in range(len(unique_labels))]))
        AUCs, PR_AUCs, confu_mats = {}, {}, {}
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        checkpoint_path = './saved_model/'+current_time
        os.system('mkdir '+ checkpoint_path)
        # now binary only coupled with multiclass

        for i in range(len(unique_labels)):
            train_dat_aug = PTMDataGenerator(data_prefix+'train.json', FLAGS, shuffle=True,ind=i, eval=False, binary=True)
            unique_labels = train_dat_aug.unique_labels
            val_dat_aug = PTMDataGenerator(data_prefix+'val.json', FLAGS, shuffle=True,ind=i, eval=True, binary=True)
            test_dat_aug = PTMDataGenerator(data_prefix+'test.json', FLAGS, shuffle=True,ind=i, eval=True, binary=True)
            val_dat_aug.update_unique_labels(unique_labels)
            test_dat_aug.update_unique_labels(unique_labels)
            print('training on '+ unique_labels[i])
            # make checkpoint for each predictor
            os.system('mkdir '+ checkpoint_path+'/'+unique_labels[i])
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+'/'+unique_labels[i], save_best_only=True, verbose=1)
            training_callbacks = [
                #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
                keras.callbacks.EarlyStopping(monitor='loss',patience = 2, restore_best_weights = True),
                cp_callback 
            ] 
            model=b_model
            # if unique_labels[i]=='Hydro_K':
            #     pdb.set_trace()
            model.train(train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels,lr = FLAGS.learning_rate, \
                callbacks=training_callbacks, binary=FLAGS.binary, ind=i, graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
            # model.model = tf.keras.models.load_model(checkpoint_path+'/'+unique_labels[i])
            AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, graph=FLAGS.graph, binary=FLAGS.binary, ind=i, num_cont=FLAGS.fill_cont)
            AUCs[unique_labels[i]], PR_AUCs[unique_labels[i]], confu_mats[unique_labels[i]] = AUC, PR_AUC, confusion_matrixs
            if FLAGS.graph:
                model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model+'_'+FLAGS.dataset+'_multi_binary_'+\
                    unique_labels[i]+'_graph_'+str(FLAGS.seq_len)+'_'+str(FLAGS.fill_cont) + 'no_pdb' + str(FLAGS.no_pdb)
            else:
                model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model+'_'+FLAGS.dataset+'_multi_binary_'+\
                    unique_labels[i]+'_'+str(FLAGS.seq_len)
            model.model.save(model_name)
        for u in unique_labels:
            # print(u)
            print('%.3f'%PR_AUCs[u])
        for u in unique_labels:
            print(u)
            print(confu_mats[u])
        

        os.system('rm -r '+checkpoint_path)

    # os.system('rm -r '+ checkpoint_path)

    # if FLAGS.binary:
    #     # train on large samples first
    #     sort_ind = np.argsort(-np.array([train_data.Y[i].shape[0] for i in range(len(train_data.Y))]))
    #     AUCs, PR_AUCs, confu_mats = {}, {}, {}
    #     for i in sort_ind:
    #         print('training on '+ unique_labels[i])
    #         model.train(train_data, val_data, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, lr = FLAGS.learning_rate, callbacks=training_callbacks, binary=FLAGS.binary, ind=i)
    #         AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_data, FLAGS.batch_size, unique_labels, binary=FLAGS.binary, ind=i)
    #         AUCs[unique_labels[i]], PR_AUCs[unique_labels[i]], confu_mats[unique_labels[i]] = AUC, PR_AUC, confusion_matrixs

    #     for i in sort_ind:
    #         u = unique_labels[i]
    #         print(u)
    #         #print('%.3f'%AUCs[u])
    #         print('%.3f'%PR_AUCs[u])
    #     for i in sort_ind:
    #         u = unique_labels[i]
    #         print(u)    
    #         print(confu_mats[u])
            

    

    # # Training and Evaluating.
    # best_f1 = 0.0
    # for epoch in range(FLAGS.num_epochs):
    #     # Reset metrics.
    #     train_loss.reset_states()
    #     # Training.
    #     num_batches = (len(train_data.records) + FLAGS.batch_size - 1)
    #     num_batches = num_batches // FLAGS.batch_size
    #     preds, lbls = [], []
    #     for data in tqdm(train_data.batch_iter(), desc='Training',
    #             total=num_batches):
    #         preds.extend(list(train_step(data)))
    #         lbls.extend(list(data['label']))
    #     train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe = \
    #             eval(lbls, preds)

    #     tmpl = 'Epoch {} (CV={}, K={}, L={})\n' +\
    #             'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    #     print(tmpl.format(
    #         epoch + 1, FLAGS.cv, FLAGS.K, FLAGS.L,
    #         train_loss.result(),
    #         train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe),
    #         file=sys.stderr)

    # # Testing and Evaluating.
    # # Reset metrics.
    # test_loss.reset_states()
    # # Training.
    # num_batches = (len(test_data.records) + FLAGS.batch_size - 1)
    # num_batches = num_batches // FLAGS.batch_size
    # preds, lbls = [], []
    # for data in tqdm(test_data.batch_iter(is_random=False),
    #         desc='Testing', total=num_batches):
    #     preds.extend(list(valid_step(data, test_loss)))
    #     lbls.extend(list(data['label']))

    # lbls = [int(x) for x in lbls]
    # preds = [float(x) for x in preds]
    # test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe = \
    #         eval(lbls, preds)

    # tmpl = 'Testing (CV={}, K={}, L={})\n' +\
    #         'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    # print(tmpl.format(FLAGS.cv, FLAGS.K, FLAGS.L,
    #     test_loss.result(),
    #     test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe),
    #     file=sys.stderr)

    # logging.info('Saving testing predictions to to {}.'.format(path_pred))
    # with open(path_pred, 'w') as wp:
    #     json.dump(list(zip(preds, lbls)), wp)




if __name__ == '__main__':
    app.run(main)

