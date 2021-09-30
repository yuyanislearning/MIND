import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from spektral.layers import GCNConv, GlobalSumPool, GATConv


from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix
from .tokenization import  n_tokens

import pprint
import pdb
import sys
import importlib.util


sys.path.append("/workspace/PTM/protein_bert/")
sys.path.append('/workspace/PTM/transformers/src/')
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, log
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from spektral.layers import GCNConv, GlobalSumPool



class Raw_model():
    def __init__(self, optimizer , loss_object, lr ):
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.lr = lr
        
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, lr = None, callbacks=[], binary=None,ind=None, graph=False):
        train_X, train_Y, train_sample_weights = (encoded_train_set.X[ind], encoded_train_set.Y[ind], encoded_train_set.sample_weights[ind]) if binary \
            else (encoded_train_set.X, encoded_train_set.Y, encoded_train_set.sample_weights)

        val_set = (encoded_valid_set.X[ind], encoded_valid_set.Y[ind], encoded_valid_set.sample_weights[ind]) if binary \
            else (encoded_valid_set.X, encoded_valid_set.Y, encoded_valid_set.sample_weights)
        #pdb.set_trace()
        self.model.fit(train_X, train_Y, sample_weight = train_sample_weights, batch_size = batch_size, epochs = n_epochs,\
             validation_data = val_set, callbacks = callbacks)
    def eval(self, seq_len,test_data, batch_size, unique_labels,binary, ind=None):
        test_X, test_Y, test_sample_weights = (test_data.X[ind], test_data.Y[ind], test_data.sample_weights[ind]) if binary \
            else (test_data.X, test_data.Y, test_data.sample_weights)
        y_pred = self.model.predict(test_X, batch_size)
        
        if not binary:
            y_mask = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
            y_true = test_Y.reshape(-1, seq_len, len(unique_labels))
            y_pred = y_pred.reshape(-1, seq_len, len(unique_labels))
        else:
            y_mask = test_sample_weights
            y_true = test_Y
            

        y_trues = y_true[y_mask==1] if binary \
            else [y_true[:,:,i][y_mask[:,:,i]==1] for i in range(len(unique_labels))]
        y_preds = y_pred[y_mask==1] if binary \
            else [y_pred[:,:,i][y_mask[:,:,i]==1] for i in range(len(unique_labels))]

        ptm_type = {i:p for i, p in enumerate(unique_labels)}
        AUC = {}
        PR_AUC = {}
        if binary:
            AUC = roc_auc_score(y_trues, y_preds)
            PR_AUC = precision_recall_AUC(y_trues, y_preds) 
            confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues, y_preds>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])

        else:
            for i in range(len(unique_labels)):
                AUC[ptm_type[i]] = roc_auc_score(y_trues[i], y_preds[i]) 
                PR_AUC[ptm_type[i]] = precision_recall_AUC(y_trues[i], y_preds[i]) 
                confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues[i], y_preds[i]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
                # print(ptm_type[i]+' confusion matrix')
                # print(confusion_matrixs)
                confusion_matrixs=None

        
        return AUC, PR_AUC, confusion_matrixs

class RNN_model(Raw_model):
    def __init__(self, optimizer_class, loss_object):
        Raw_model.__init__(self, optimizer_class, loss_object )
    def create_model(self, seq_len, d_hidden_seq, unique_labels,dropout, is_binary=None, graph=False):
        # model = keras.Sequential()
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
        lstm = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(encoding)
        lstm2 = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(lstm)
        last_hidden_layer = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(lstm2)
        # model.add(layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')) 
        # model.add(
        #     layers.Bidirectional(layers.LSTM(128, input_dim=seq_len, return_sequences=True)))
        # model.add(
        #     layers.Bidirectional(layers.LSTM(128,  return_sequences=True)))
        # model.add(
        #     layers.Bidirectional(layers.LSTM(128,  return_sequences=True)))   
        if graph:
            adj_input = keras.layers.Input(shape=(seq_len, seq_len), name = 'input-adj')
            # last_hidden_layer_chop = tf.keras.layers.Lambda(lambda x: x[:,1:(seq_len-1),:])
            # TODO the adj matrix should be padded in head and tail to be consistent
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout)(last_hidden_layer)
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout)(last_hidden_layer)
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input]) 
        out = layers.Dense(1)(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels))

        # model.add(layers.Dense(1) if is_binary else layers.Dense(len(unique_labels)))
        if not is_binary:
            out = layers.Reshape(-1,)(out)
            # model.add(layers.Reshape((-1,)))
        model_input = [input_seq, adj_input] if graph else [input_seq]
        model = keras.models.Model(inputs = model_input , outputs = out)

        self.model = model
        self.model.compile(
            loss = self.loss_object,
            optimizer = self.optimizer,
        )


class ProteinBert(Raw_model):
    def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, short=None):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
        pretraining_model_generator, _ = load_pretrained_model(local_model_dump_dir='/workspace/PTM/protein_bert/proteinbert',local_model_dump_file_name='epoch_92400_sample_23500000.pkl',\
            lr = self.lr)
        if short:
            output_spec = OutputSpec(OutputType(False, 'binary'), [0,1]) if binary else OutputSpec(OutputType(False, 'multilabel'), unique_labels)
        else:
            output_spec = OutputSpec(OutputType(True, 'binary'), [0,1]) if binary else OutputSpec(OutputType(True, 'multilabel'), unique_labels)

        uni_l = [0,1] if binary else unique_labels
        self.model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = \
                get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5, unique_labels = uni_l)
    def create_model(self, train_data,  seq_len,   binary=None, freeze_pretrained_layers=None, graph=False):   
        train_X, train_Y, _ = (train_data.X[0], train_data.Y[0], train_data.sample_weights[0]) if binary \
            else (train_data.X, train_data.Y, train_data.sample_weights)
        self.model_generator.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
        self.model = self.model_generator.create_model(seq_len,  freeze_pretrained_layers = freeze_pretrained_layers, graph = graph)

# class ProteinBert(ProteinBert):
#     def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, short=None):
#         Raw_model.__init__(self, optimizer_class, loss_object, lr )
#         pretraining_model_generator, _ = load_pretrained_model(local_model_dump_dir='/workspace/PTM/protein_bert/proteinbert',local_model_dump_file_name='epoch_92400_sample_23500000.pkl',\
#             lr = self.lr)
#         if short:
#             output_spec = OutputSpec(OutputType(False, 'binary'), [0,1]) if binary else OutputSpec(OutputType(False, 'multilabel'), unique_labels)
#         else:
#             output_spec = OutputSpec(OutputType(True, 'binary'), [0,1]) if binary else OutputSpec(OutputType(True, 'multilabel'), unique_labels)

#         uni_l = [0,1] if binary else unique_labels
#         self.model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = \
#                 get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5, unique_labels = uni_l)
#     def create_model(self, train_data,  seq_len,   binary=None, freeze_pretrained_layers=None):   
#         train_X, train_Y, _ = (train_data.X[0], train_data.Y[0], train_data.sample_weights[0]) if binary \
#             else (train_data.X, train_data.Y, train_data.sample_weights)
#         self.model_generator.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
#         self.model = self.model_generator.create_model(seq_len,  freeze_pretrained_layers = freeze_pretrained_layers)



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
    pre_order = np.argsort(precision)
    return auc(precision[pre_order], recall[pre_order])