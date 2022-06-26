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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score

from tqdm import tqdm
import yaml
import time
import gc


from src.utils import get_class_weights,  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import GAT_model,  RNN_model, TransFormer, TransFormerGAT, LSTMTransFormer, CNN_model, TransFormer_CNN, CustomTransFormer

t0 = time.time()
handle_flags()

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def build_model(FLAGS, optimizer , unique_labels, binary=False):
    if FLAGS.model=='RNN':
        model = RNN_model()
        model.create_model(FLAGS.model, optimizer, FLAGS.seq_len, 128, unique_labels, 0.6, is_binary=binary, \
            graph=FLAGS.graph, n_lstm=FLAGS.n_lstm, n_gcn=FLAGS.n_gcn)
    elif FLAGS.model=='CNN':
        model = CNN_model()  
        model.create_model(FLAGS.model, optimizer, FLAGS.seq_len,is_binary=binary, unique_labels=unique_labels)
    elif FLAGS.model=='GAT':
        model = GAT_model()
        model.create_model(FLAGS.model, optimizer, FLAGS.n_lstm, FLAGS.seq_len, is_binary=binary, \
            unique_labels=unique_labels,d_hidden_seq=128, dropout=0.6)
    elif FLAGS.model=='Transformer':
        model = TransFormer(FLAGS, FLAGS.model,optimizer,  \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=binary,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
        if FLAGS.pretrain:
            if FLAGS.pretrain_name=='PTM':
                pretrain_name = 'saved_model/MLM_Transformer/Transformer_multi_514_all_PTM_1'
            pretrain_model = tf.keras.models.load_model(pretrain_name)
            for li in range(len(pretrain_model.layers)):
                layer = pretrain_model.layers[li]
                if layer.name not in ['my_last_dense', 'reshape',]:
                    model.model.get_layer(index=li).set_weights(layer.get_weights())
    elif FLAGS.model=='CustomTransformer':
        model = CustomTransFormer(FLAGS.model,optimizer,  d_model=128, \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=binary,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
    elif FLAGS.model=='Transformer_CNN':
        model = TransFormer_CNN(FLAGS.model,optimizer,  d_model=128, \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=binary,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)                        
    elif FLAGS.model=='TransformerGAT':
        model = TransFormerGAT(FLAGS.model,optimizer,  d_model=128, \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=binary,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
        if FLAGS.pretrain:
            if FLAGS.pretrain_name=='PTM':
                pretrain_name = 'saved_model/MLM_Transformer/Transformer_multi_514_all_PTM'
            pretrain_model = tf.keras.models.load_model(pretrain_name)
            for li in range(len(pretrain_model.layers)):
                layer = pretrain_model.layers[li]
                if layer.name not in ['my_last_dense', 'reshape',]:
                    model.model.get_layer(index=li).set_weights(layer.get_weights())   
    elif FLAGS.model=='LSTMTransformer':
        model = LSTMTransFormer(FLAGS,FLAGS.model,optimizer,  \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=binary,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
        if FLAGS.pretrain:
            if FLAGS.pretrain_name=='PTM':
                pretrain_name = 'saved_model/MLM_Transformer/Transformer_multi_514_all_PTM'
            pretrain_model = tf.keras.models.load_model(pretrain_name)
            for li in range(len(pretrain_model.layers)):
                layer = pretrain_model.layers[li]
                if layer.name not in ['my_last_dense', 'reshape',]:
                    model.model.get_layer(index=li).set_weights(layer.get_weights())   
    return model   

def save_model(model, FLAGS, fold=None):
    model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model + '_'+str(FLAGS.seq_len)
    if FLAGS.neg_sam:
        model_name+='_negsam'
    if FLAGS.pretrain:
        model_name+='_pretrain_'+ FLAGS.pretrain_name
    if FLAGS.multilabel:
        model_name += '_multi'
    # if FLAGS.binary:
    #     model_name += '_binary'
    if FLAGS.graph:
        model_name += '_graph_'
        model_name += str(FLAGS.fill_cont)
        if FLAGS.no_pdb:
            model_name+='_no_pdb_' + str(FLAGS.no_pdb)
    model_name+='n_layer_'+str(FLAGS.n_lstm)
    model_name+=FLAGS.suffix
    if FLAGS.ensemble:
        model_name+='_fold_'+str(fold)
    model.model.save(model_name)   
    return model_name

def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[i][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    # limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)
    #tf.config.run_functions_eagerly(True)#TODO remove

    data_prefix = '{}/PTM_'.format(
            FLAGS.data_path) 
    
    # class_weights = get_class_weights(train_data, val_data, test_data, unique_labels) if FLAGS.class_weights else None
    if not FLAGS.neg_sam:
        # with open('/workspace/PTM/PTM-pattern-finder/analysis/res/class_weights.json','r') as f:
        with open('/workspace/PTM/Data/OPTM/combined/class_weigth.json','r') as f:
            class_weights = json.load(f)
            lower = 1
            class_weights = {k:[class_weights[k][0]*lower,class_weights[k][1]] for k in class_weights}
    if FLAGS.ensemble:
        train_dat_aug = PTMDataGenerator(data_prefix+'train.json', FLAGS, shuffle=True,ind=None, eval=False, class_weights=class_weights)
        unique_labels = train_dat_aug.unique_labels
        train_dat_aug.train_val_split()
        val_dat_aug = train_dat_aug.init_fold(0)
        train_dat_aug.on_epoch_end()
    else:# Load data
        train_dat_aug = PTMDataGenerator(data_prefix+'train.json', FLAGS, shuffle=True,ind=None, eval=False, class_weights=class_weights)
        unique_labels = train_dat_aug.unique_labels
        val_dat_aug = PTMDataGenerator(data_prefix+'val.json', FLAGS, shuffle=True,ind=None, eval=True)
        val_dat_aug.update_unique_labels(unique_labels)
    test_dat_aug = PTMDataGenerator(data_prefix+'test.json', FLAGS, shuffle=True,ind=None, eval=True)
    test_dat_aug.update_unique_labels(unique_labels)


    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    # metrics = [CategoricalTruePositives(13,batch_size=FLAGS.batch_size)]#,tf.keras.metrics.FalsePositives(),
    # tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()
     
    if FLAGS.multilabel:# multi-label      
        training_callbacks = [
            #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
            keras.callbacks.EarlyStopping(monitor='val_loss',patience = 25, restore_best_weights = True),
            #keras.metrics.Accuracy(),
        ] 
        if FLAGS.ensemble:
            PR_AUCs = {}
            optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=True)
            model = build_model(FLAGS, optimizer, unique_labels, binary=False) 
            model.train( train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels,\
                    lr = FLAGS.learning_rate, callbacks=training_callbacks,graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
            logging.info('------------------evaluate 0 fold---------------------' )
            AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,val_dat_aug, FLAGS.batch_size, unique_labels, \
                FLAGS.graph, num_cont=FLAGS.fill_cont) 
            for u in unique_labels:
                print('%.3f'%PR_AUC[u])
            for u in unique_labels:
                print(u)
                print(confusion_matrixs[u])
            model_name = save_model(model, FLAGS, fold=0)
            PR_AUCs[0] = PR_AUC
            
            for i in range(1, FLAGS.n_fold):
                val_dat_aug = train_dat_aug.init_fold(i)
                train_dat_aug.on_epoch_end()
                optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=True)
                model = build_model(FLAGS, optimizer, unique_labels, binary=False) 
                model.train( train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels,\
                    lr = FLAGS.learning_rate, callbacks=training_callbacks,graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
                logging.info('------------------evaluate %d fold---------------------'%(i))
                AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,val_dat_aug, FLAGS.batch_size, unique_labels, \
                    FLAGS.graph, num_cont=FLAGS.fill_cont) 
                for u in unique_labels:
                    print('%.3f'%PR_AUC[u])
                for u in unique_labels:
                    print(u)
                    print(confusion_matrixs[u])       
                model_name = save_model(model, FLAGS, fold=i)
                PR_AUCs[i] = PR_AUC
                # save PR_AUC 
            if FLAGS.n_fold >10:
                model_name = model_name[0:len(model_name)-1]
            PR_AUC_name=model_name[:-7]+'_PRAU.json'
            with open(PR_AUC_name,'w') as fw:
                json.dump(PR_AUCs, fw)
            print('----------------------Ensemble evaluation -----------------------')
            y_preds = []
            for i in range(FLAGS.n_fold):
                if i>10:
                    model_name = model_name[0:(len(model_name)-1)]# 
                model_name = model_name[0:(len(model_name)-1)] + str(i) # change fold
                model = build_model(FLAGS, optimizer, unique_labels) 
                model.model = tf.keras.models.load_model(model_name)
                y_true, y_pred = model.predict(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, binary=False)
                y_preds.append(y_pred)
            weights = ensemble_get_weights(PR_AUCs, unique_labels)

            for u in unique_labels:
                y_pred = np.stack([np.array(y_preds[i][u]*weights[u][i]) for i in range(len(y_preds))])
                y_pred = np.sum(y_pred, axis=0)
                pr_auc = average_precision_score(y_true[u], y_pred)
                print('%.3f'%pr_auc)
            
        else:
            model = build_model(FLAGS, optimizer, unique_labels) 
            model.train( train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels,\
                lr = FLAGS.learning_rate, callbacks=training_callbacks,graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
            logging.info('------------------evaluate---------------------' )
            AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, \
                FLAGS.graph, num_cont=FLAGS.fill_cont)

            optms = ["Arg-OH_R","Asn-OH_N","Asp-OH_D","Cys4HNE_C","CysSO2H_C",\
            "CysSO3H_C","Lys-OH_K","Lys2AAA_K","MetO_M","MetO2_M","Phe-OH_F",\
            "ProCH_P","Trp-OH_W","Tyr-OH_Y","Val-OH_V"]
            for u in unique_labels:
                print('%.3f'%PR_AUC[u])
            for u in unique_labels:
                print(u)
                print(confusion_matrixs[u])
            if FLAGS.save_model:
                save_model(model, FLAGS)

    if FLAGS.binary:# multi-label
        b_model = build_model(FLAGS, optimizer, unique_labels, binary=True)      
        #for layer in model.model.layers:
        if FLAGS.multilabel:
            for li in range(len(model.model.layers)):
                layer = model.model.layers[li]  
                if layer.name not in ['my_last_dense', 'reshape','my_last_dense_1', 'my_last_dense_2']:
                    b_model.model.get_layer(index=li).set_weights(layer.get_weights())


        # temp_y = train_data.Y.reshape((train_data.Y.shape[0], FLAGS.seq_len, -1))
        # sort_ind = np.argsort(-np.array([np.sum(temp_y[:,:,i]) for i in range(len(unique_labels))]))
        AUCs, PR_AUCs, confu_mats = {}, {}, {}
        # now binary only coupled with multiclass

        for i in range(len(unique_labels)):
            train_dat_aug = PTMDataGenerator(data_prefix+'train.json', FLAGS, shuffle=True,ind=i, eval=False, binary=True, class_weights=class_weights)
            unique_labels = train_dat_aug.unique_labels
            val_dat_aug = PTMDataGenerator(data_prefix+'val.json', FLAGS, shuffle=True,ind=i, eval=True, binary=True)
            test_dat_aug = PTMDataGenerator(data_prefix+'test.json', FLAGS, shuffle=True,ind=i, eval=True, binary=True)
            val_dat_aug.update_unique_labels(unique_labels)
            test_dat_aug.update_unique_labels(unique_labels)
            print('training on '+ unique_labels[i])

            training_callbacks = [
                #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
                keras.callbacks.EarlyStopping(monitor='val_loss',patience = 50, restore_best_weights = True),
            ] 
            model=b_model
            # if unique_labels[i]=='Hydro_K':
            #     pdb.set_trace()
            model.train(train_dat_aug, val_dat_aug, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels,lr = FLAGS.learning_rate, \
                callbacks=training_callbacks, binary=FLAGS.binary, ind=i, graph=FLAGS.graph, num_cont=FLAGS.fill_cont)
            AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, \
                graph=FLAGS.graph, binary=FLAGS.binary, ind=i, num_cont=FLAGS.fill_cont)
            AUCs[unique_labels[i]], PR_AUCs[unique_labels[i]], confu_mats[unique_labels[i]] = AUC, PR_AUC, confusion_matrixs
            
            model_name = './saved_model/'+FLAGS.model+'/'+FLAGS.model + '_'+str(FLAGS.seq_len) + '_negsam_'+str(FLAGS.neg_sam)

            if FLAGS.pretrain:
                model_name+='_pretrain_'+ FLAGS.pretrain_name
            if FLAGS.multilabel:
                model_name += '_multi'
            if FLAGS.binary:
                model_name += '_binary'
            if FLAGS.graph:
                model_name += '_graph_'
                model_name += str(FLAGS.fill_cont)
                if FLAGS.no_pdb:
                    model_name+='_no_pdb_' + str(FLAGS.no_pdb)
            if FLAGS.save_model:
                model.model.save(model_name)

        for u in unique_labels:
            # print(u)
            if u in optms:
                print('%.3f'%PR_AUCs[u])
        for u in unique_labels:
            if u in optms:
                print(u)
                print(confu_mats[u])
        
    t1 = time.time()
    total_time = t1-t0
    print('the total time used is:')
    print(total_time)





if __name__ == '__main__':
    app.run(main)

