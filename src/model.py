import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.eager import backprop

from datetime import datetime
from packaging import version


import pandas as pd
from spektral.layers import GCNConv, GlobalSumPool, GATConv


from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from .tokenization import  n_tokens
from .MyModel import MyModel

import pprint
import pdb
import sys
import importlib.util

from src.utils import PTMDataGenerator, get_graph

sys.path.append("/workspace/PTM/protein_bert/")
sys.path.append('/workspace/PTM/transformers/src/')
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from spektral.layers import GCNConv, GlobalSumPool

from tensorflow.python.eager import monitoring
from tensorflow.python.keras.engine import compile_utils

class Raw_model():
    def __init__(self, optimizer , loss_object, lr ):
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.lr = lr
        
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, unique_labels, lr = None, callbacks=[], binary=None,ind=None, graph=False, num_cont=None):
        # train_X, train_Y, train_sample_weights = (encoded_train_set.X[ind], encoded_train_set.Y[ind], encoded_train_set.sample_weights[ind]) if binary \
        #     else (encoded_train_set.X, encoded_train_set.Y, encoded_train_set.sample_weights)

        # val_set = encoded_valid_set.X, encoded_valid_set.Y, encoded_valid_set.sample_weights
        # # get graph for val
        # if graph:
        #     adjs = get_graph(encoded_valid_set.uid, seq_len)
        #     val_set[0].append(adjs)

        aug = PTMDataGenerator( encoded_train_set, seq_len, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=False, num_cont=num_cont)
        val_aug = PTMDataGenerator( encoded_valid_set, seq_len, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=True, num_cont=num_cont)
        # logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # callbacks.append(keras.callbacks.TensorBoard(log_dir=logdir))
        #self.model.fit_generator(aug, validation_data = val_set,  epochs=n_epochs, callbacks = callbacks)
        # for i in range(n_epochs):
        #     if i%1==10 and i!=0:# 
        #         self.eval(seq_len, encoded_train_set, batch_size, unique_labels, graph=graph,binary=binary, train=True)
        #     print('================%d epoch==============='%i)
            # self.model.fit(aug,  epochs = 1, steps_per_epoch=aug.__len__(), validation_data = val_set, callbacks = callbacks)
        if batch_size > len(aug.list_id):
            # in case batch size greater than sample size
            aug = PTMDataGenerator( encoded_train_set, seq_len, batch_size=len(aug.list_id), unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=False, num_cont=num_cont)
        if batch_size > len(val_aug.list_id) and len(val_aug.list_id)!=0:
            val_aug = PTMDataGenerator( encoded_valid_set, seq_len, batch_size=len(val_aug.list_id), unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=True, num_cont=num_cont)
        self.model.fit(aug,  epochs = n_epochs, steps_per_epoch=aug.__len__(), validation_data = val_aug, callbacks = callbacks)#, validation_steps=val_aug.__len__())


    def eval(self, seq_len,test_data, batch_size, unique_labels,graph=False, binary=False, ind=None, num_cont=None):
        aug = PTMDataGenerator( test_data, seq_len, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont)#test_data.Y.shape[0]
        aug = PTMDataGenerator( test_data, seq_len, batch_size=len(aug.list_id), unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont)#test_data.Y.shape[0]
        test_X, test_Y, test_sample_weights = aug.__getitem__(0)
        y_pred = self.model.predict(test_X, batch_size=batch_size)
        
        if not binary:
            y_masks = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
            y_trues = test_Y.reshape(-1, seq_len, len(unique_labels))
            y_preds = y_pred.reshape(-1, seq_len, len(unique_labels))
        else:
            y_mask = test_sample_weights
            y_true = test_Y
                    

        ptm_type = {i:p for i, p in enumerate(unique_labels)}
        AUC = {}
        PR_AUC = {}
        confusion_matrixs = {}
        if binary:
            y_trues = y_true[y_mask==1]
            y_preds = y_pred[y_mask==1]
            AUC = roc_auc_score(y_trues, y_preds)
            PR_AUC = average_precision_score(y_trues, y_preds) 
            confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues, y_preds>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
        else:
            for i in range(len(unique_labels)):
                y_true = y_trues[:,:,i]
                y_pred = y_preds[:,:,i]
                y_mask = y_masks[:,:,i]
                if np.sum(y_true)==0:
                    continue
                yp = np.sum(y_true, axis=1).astype(int)
                # only retain case with positive labels
                y_true = y_true[yp>0]
                y_pred = y_pred[yp>0]
                y_mask = y_mask[yp>0]    

                y_true = y_true[y_mask==1]
                y_pred = y_pred[y_mask==1]


                AUC[ptm_type[i]] = roc_auc_score(y_true, y_pred) 
                PR_AUC[ptm_type[i]] = average_precision_score(y_true, y_pred) 
                confusion_matrixs[ptm_type[i]]=pd.DataFrame(confusion_matrix(y_true, y_pred>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
                # print(ptm_type[i]+' confusion matrix')
                # print(confusion_matrixs)
                #confusion_matrixs=None

        
        return AUC, PR_AUC, confusion_matrixs


class RNN_model(Raw_model):
    def __init__(self, optimizer_class, loss_object, lr):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
    def create_model(self, seq_len, d_hidden_seq, unique_labels,dropout, metrics=[],is_binary=None,is_multi=None, graph=False, n_lstm=3, n_gcn=None):
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
        lstm = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True), name='lstm-1')(encoding)
        assert n_lstm > 2
        for i in range(n_lstm-3):
            lstm = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True),  name='lstm-'+str(i+2))(lstm)
        lstm2 = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True), name='lstm-'+str(n_lstm-1))(lstm)
        last_hidden_layer = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True), name='lstm-'+str(n_lstm))(lstm2)  
        if graph:
            adj_input = keras.layers.Input(shape=(seq_len, seq_len), name = 'input-adj')
            last_hidden_layer = GATConv(channels=128, name='gcn-1' )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-1')(last_hidden_layer)
            assert n_gcn > 0
            for i in range(n_gcn-1):
                last_hidden_layer = GATConv(channels=128, name = 'gcn-'+str(i+2) )([last_hidden_layer, adj_input])
                last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-'+str(i+2))(last_hidden_layer)
        out = layers.Dense(1, activation = 'sigmoid', name = 'dense')(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels), activation = 'sigmoid', name='dense')(last_hidden_layer)

        if not is_binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq, adj_input] if graph else [input_seq]
        model = keras.models.Model(inputs = model_input , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )



class GAT_model(Raw_model):
    def __init__(self, optimizer_class, loss_object, lr):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
    def create_model(self, seq_len, d_hidden_seq, unique_labels,dropout, is_binary=None, n_gcn=None):
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        adj_input = keras.layers.Input(shape=(seq_len, seq_len), name = 'input-adj')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)

        last_hidden_layer = GATConv(channels=128, name='gcn-1' )([encoding, adj_input])
        last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-1')(last_hidden_layer)
        assert n_gcn > 0
        for i in range(n_gcn-1):
            last_hidden_layer = GATConv(channels=128, name = 'gcn-'+str(i+2) )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-'+str(i+2))(last_hidden_layer)

        for i in range(3):#TODO
            last_hidden_layer = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True),  name='lstm-'+str(i+1))(last_hidden_layer)
        out = layers.Dense(1, activation = 'sigmoid', name = 'dense')(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels), activation = 'sigmoid', name='dense')(last_hidden_layer)

        if not is_binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq, adj_input]
        model = keras.models.Model(inputs = model_input , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )


class ProteinBert(Raw_model):
    def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, multilabel=None):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
        pretraining_model_generator, _ = load_pretrained_model(local_model_dump_dir='/workspace/PTM/protein_bert/proteinbert',local_model_dump_file_name='epoch_92400_sample_23500000.pkl',\
            lr = self.lr)
        # if short:
        #     output_spec = OutputSpec(OutputType(False, 'binary'), [0,1]) if binary else OutputSpec(OutputType(False, 'multilabel'), unique_labels)
        if multilabel:
            output_spec = OutputSpec(OutputType(True, 'multilabel'), unique_labels)
        if binary:
            output_spec = OutputSpec(OutputType(True, 'binary'), [0,1])
        uni_l = [0,1] if binary else unique_labels
        self.model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = \
                get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5, unique_labels = uni_l)
    def create_model(self, train_data,  seq_len,freeze_pretrained_layers=None,binary=None, graph=False, n_gcn=None):   
        train_X, train_Y, _ = (train_data.X[0], train_data.Y[0], train_data.sample_weights[0]) if binary \
            else (train_data.X, train_data.Y, train_data.sample_weights)
        self.model_generator.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
        self.model = self.model_generator.create_model(seq_len,  freeze_pretrained_layers = freeze_pretrained_layers, graph = graph, n_gcn=n_gcn)


class ProtTrans(Raw_model):
    def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, short=None):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )



def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)

def _slice_arrays(arrays, slicing):
    if isinstance(arrays, list) or isinstance(arrays, tuple):
        return [array[slicing] for array in arrays]
    else:
        return arrays[slicing]

def precision_recall_AUC(y_true, y_pred_classes):
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred_classes.flatten())
    # pre_order = np.argsort(precision)
    # return auc(precision[pre_order], recall[pre_order])
    return auc(recall,precision)