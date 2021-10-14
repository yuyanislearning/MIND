import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout

from datetime import datetime
from packaging import version


import pandas as pd
from spektral.layers import GCNConv, GlobalSumPool, GATConv


from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix
from .tokenization import  n_tokens

import pprint
import pdb
import sys
import importlib.util

from src.utils import PTMDataGenerator, get_graph

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
        
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, unique_labels, lr = None, callbacks=[], binary=None,ind=None, graph=False, ):
        train_X, train_Y, train_sample_weights = (encoded_train_set.X[ind], encoded_train_set.Y[ind], encoded_train_set.sample_weights[ind]) if binary \
            else (encoded_train_set.X, encoded_train_set.Y, encoded_train_set.sample_weights)

        val_set = (encoded_valid_set.X[ind], encoded_valid_set.Y[ind], encoded_valid_set.sample_weights[ind]) if binary \
            else (encoded_valid_set.X, encoded_valid_set.Y, encoded_valid_set.sample_weights)
        # get graph for val
        if graph:
            adjs = get_graph(encoded_valid_set.uid, seq_len)
            val_set[0].append(adjs)
            

        aug = PTMDataGenerator( encoded_train_set, seq_len, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=True)

        # logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # callbacks.append(keras.callbacks.TensorBoard(log_dir=logdir))
        #self.model.fit_generator(aug, validation_data = val_set,  epochs=n_epochs, callbacks = callbacks)
        for i in range(n_epochs):
            if i%1==0 and i!=0:# 
                self.eval(seq_len, encoded_train_set, batch_size, unique_labels, graph=graph,binary=binary, train=True)
            print('================%d epoch==============='%i)
            pdb.set_trace()
            self.model.fit(aug,  epochs = 1, steps_per_epoch=aug.__len__(), validation_data = val_set, callbacks = callbacks)
        # self.model.fit(aug,  epochs = n_epochs, steps_per_epoch=aug.__len__(), validation_data = val_set, callbacks = callbacks)


    def eval(self, seq_len,test_data, batch_size, unique_labels,graph=False, binary=False, ind=None, train=False):
        test_X, test_Y, test_sample_weights = (test_data.X[ind], test_data.Y[ind], test_data.sample_weights[ind]) if binary \
            else (test_data.X, test_data.Y, test_data.sample_weights)
        if graph and not train:
            adjs = get_graph(test_data.uid, seq_len)
            test_X.append(adjs)
        if train:#TODO remove
            aug = PTMDataGenerator( test_data, seq_len, batch_size=2000, unique_labels=unique_labels, graph = graph,shuffle=False)#test_data.Y.shape[0]
            test_X, test_Y, test_sample_weights = aug.__getitem__(index=0)
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

        # print('true label')
        # print(y_trues[2])
        # print('prediction')
        # print(y_preds[2])
        ptm_type = {i:p for i, p in enumerate(unique_labels)}
        AUC = {}
        PR_AUC = {}
        confusion_matrixs = {}
        if binary:
            AUC = roc_auc_score(y_trues, y_preds)
            PR_AUC = precision_recall_AUC(y_trues, y_preds) 
            confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues, y_preds>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
        else:
            for i in range(len(unique_labels)):
                if np.sum(y_trues[i])==0:
                    continue
                AUC[ptm_type[i]] = roc_auc_score(y_trues[i], y_preds[i]) 
                PR_AUC[ptm_type[i]] = precision_recall_AUC(y_trues[i], y_preds[i]) 
                confusion_matrixs[ptm_type[i]]=pd.DataFrame(confusion_matrix(y_trues[i], y_preds[i]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
                # print(ptm_type[i]+' confusion matrix')
                # print(confusion_matrixs)
                #confusion_matrixs=None
        for i in range(len(unique_labels)):#TODO remove
            try:
                print(confusion_matrixs[ptm_type[i]])
            except:
                next
        
        return AUC, PR_AUC, confusion_matrixs


class MyModel(keras.models.Model):
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
        data: A nested structure of `Tensor`s.
        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned. Example:
        `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
        loss = self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


class RNN_model(Raw_model):
    def __init__(self, optimizer_class, loss_object, lr):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
    def create_model(self, seq_len, d_hidden_seq, unique_labels,dropout, metrics=[],is_binary=None,is_multi=None, graph=False):
        # model = keras.Sequential()
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
        lstm = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(encoding)
        # lstm = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(lstm)
        # lstm = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(lstm)
        # lstm = layers.Bidirectional(layers.LSTM(128,  return_sequences=True))(lstm)
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
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout)(last_hidden_layer)
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout)(last_hidden_layer)
            last_hidden_layer = GATConv(channels=128 )([last_hidden_layer, adj_input]) 
        out = layers.Dense(1, activation = 'sigmoid')(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels), activation = 'sigmoid')(last_hidden_layer)# TODO change back 

        # model.add(layers.Dense(1) if is_binary else layers.Dense(len(unique_labels)))
        if not is_binary:
            out = layers.Reshape((-1,))(out)
            # model.add(layers.Reshape((-1,)))
        model_input = [input_seq, adj_input] if graph else [input_seq]
        model = keras.models.Model(inputs = model_input , outputs = out)#keras.models.Model
        loss = keras.losses.BinaryCrossentropy(from_logits=False)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,#self.loss_object,
            optimizer = self.optimizer,
            #weighted_metrics = metrics,
            #run_eagerly=True,
        )


class ProteinBert(Raw_model):
    def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )
        pretraining_model_generator, _ = load_pretrained_model(local_model_dump_dir='/workspace/PTM/protein_bert/proteinbert',local_model_dump_file_name='epoch_92400_sample_23500000.pkl',\
            lr = self.lr)
        # if short:
        #     output_spec = OutputSpec(OutputType(False, 'binary'), [0,1]) if binary else OutputSpec(OutputType(False, 'multilabel'), unique_labels)
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