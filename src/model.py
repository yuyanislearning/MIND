import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils import losses_utils

import pandas as pd
from spektral.layers import GATConv
from sklearn.metrics import  precision_recall_curve, auc, roc_auc_score,  confusion_matrix, average_precision_score
from .tokenization import  n_tokens

import sys
from src.transformer import  EncoderLayer, create_padding_mask, graph_seq_attn

class Raw_model():
    def __init__(self):
        next
    def train(self,  aug, val_aug, n_epochs, callbacks=[]):
        if val_aug is not None:
            self.model.fit(aug,  epochs = n_epochs, steps_per_epoch=aug.__len__(), validation_data = val_aug, callbacks = callbacks)#, validation_steps=val_aug.__len__())
        else:
            self.model.fit(aug,  epochs = n_epochs, steps_per_epoch=aug.__len__(),  callbacks = callbacks)#, validation_steps=val_aug.__len__())

    def eval(self, seq_len,aug, batch_size, unique_labels, binary=False):
        ptm_type = {i:p for i, p in enumerate(unique_labels)}

        if binary:
            y_trues = []
            y_preds = []
        else:
            y_trues = {ptm_type[i]:[] for i in ptm_type}
            y_preds = {ptm_type[i]:[] for i in ptm_type}
        count=1
        for test_X,test_Y,test_sample_weights in aug:
            count+=1
            y_pred = self.model.predict(test_X, batch_size=batch_size)
            if not binary:
                y_mask_all = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
                y_true_all = test_Y.reshape(-1, seq_len, len(unique_labels))
                y_pred_all = y_pred.reshape(-1, seq_len, len(unique_labels))
            else:
                y_mask = test_sample_weights
                y_true = test_Y
                    

            AUC = {}
            PR_AUC = {}
            confusion_matrixs = {}
            if binary:
                y_true = y_true[y_mask==1]
                y_pred = y_pred[y_mask==1]
                y_trues.append(y_true)
                y_preds.append(y_pred)
            else:
                for i in range(len(unique_labels)):
                    
                    y_true = y_true_all[:,:,i]
                    y_pred = y_pred_all[:,:,i]
                    y_mask = y_mask_all[:,:,i]

                    y_true = y_true[y_mask==1]
                    y_pred = y_pred[y_mask==1]
                    y_trues[ptm_type[i]].append(y_true)
                    y_preds[ptm_type[i]].append(y_pred)

        if binary:
            y_trues = np.concatenate(y_trues, axis=0)
            y_preds = np.concatenate(y_preds, axis=0)
            AUC = roc_auc_score(y_trues, y_preds)
            PR_AUC = average_precision_score(y_trues, y_preds) 
            confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues, y_preds>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
        else:
            y_trues = {ptm:np.concatenate(y_trues[ptm],axis=0) for ptm in y_trues}
            y_preds = {ptm:np.concatenate(y_preds[ptm],axis=0) for ptm in y_preds}
            
            for i in range(len(unique_labels)):
                # print(y_trues[ptm_type[i]])
                AUC[ptm_type[i]] = roc_auc_score(y_trues[ptm_type[i]], y_preds[ptm_type[i]])
                PR_AUC[ptm_type[i]] = average_precision_score(y_trues[ptm_type[i]], y_preds[ptm_type[i]]) 
                confusion_matrixs[ptm_type[i]]=pd.DataFrame(confusion_matrix(y_trues[ptm_type[i]], y_preds[ptm_type[i]]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
        
        return AUC, PR_AUC, confusion_matrixs
    def predict(self, seq_len,aug, batch_size, unique_labels, binary=False):
        # predict cases
        ptm_type = {i:p for i, p in enumerate(unique_labels)}

        if binary:# TODO add or remove binary
            y_trues = []
            y_preds = []
        else:
            y_trues = {ptm_type[i]:[] for i in ptm_type}#{ptm_type:np.array:(n_sample,1)}
            y_preds = {ptm_type[i]:[] for i in ptm_type}

        for test_X,test_Y,test_sample_weights in aug:
            y_pred = self.model.predict(test_X, batch_size=batch_size)
            if not binary:
                y_mask = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
                y_true = test_Y.reshape(-1, seq_len, len(unique_labels))
                y_pred = y_pred.reshape(-1, seq_len, len(unique_labels))
                for i in range(len(unique_labels)):
                    y_true_i = y_true[:,:,i]
                    y_pred_i = y_pred[:,:,i]
                    y_mask_i = y_mask[:,:,i]

                    y_true_i = y_true_i[y_mask_i==1]
                    y_pred_i = y_pred_i[y_mask_i==1]
                    y_trues[ptm_type[i]].append(y_true_i)
                    y_preds[ptm_type[i]].append(y_pred_i)
            else:
                y_mask = test_sample_weights
                y_true = test_Y
        y_trues = {ptm:np.concatenate(y_trues[ptm],axis=0) for ptm in y_trues}
        y_preds = {ptm:np.concatenate(y_preds[ptm],axis=0) for ptm in y_preds}
                    
        return y_trues, y_preds



class LSTMTransFormer(Raw_model):
    def __init__(self, FLAGS,model_name,  optimizer,  num_layers,  num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = FLAGS.d_model
        self.model_name = model_name
        self.num_layers = num_layers
        self.FLAGS=FLAGS
        self.embedding = tf.keras.layers.Embedding(n_tokens, self.d_model, name='embedding')

        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate, split_head, global_heads, fill_cont, name='encoder_layer_'+str(n_l))
                        for n_l in range(num_layers)]
        self.graph_seq_attn = graph_seq_attn(self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels
        self.reg = FLAGS.l2_reg

    def create_model(self, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph and not self.FLAGS.gt:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj', dtype=np.int32)
        if self.FLAGS.gt:
            adj_input = keras.layers.Input(shape = (None, 20), name  ='graph_encoding')
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        if self.FLAGS.inter:
            my_emb = keras.layers.Input(shape = (None,128),  name = 'given_embed')
            x = my_emb
        x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True, \
            ), name='lstm')(x)
        
        if graph:
            graph_x = GATConv(channels=self.d_model//8, attn_heads=8,dropout_rate=0.5, activation='relu', name='gcn-1' )([x, adj_input])

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask)#self.enc_layers[i](x, mask, adj_input) if graph else 
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        if graph:
            x = tf.concat((x, graph_x), axis=-1)

        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), \
             activation = 'sigmoid', name='my_last_dense')(x)#kernel_regularizer=tf.keras.regularizers.L2(self.reg),
        
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq]
        if graph:
            model_input.append(adj_input)

        if self.FLAGS.inter:
            model_input.append(my_emb)
        model = keras.models.Model(inputs = model_input , outputs = out) 
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE) #+ reg_loss
        self.model = model
        
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
        )



class TransFormerFixEmbed():
    def __init__(self, d_model,  num_layers, num_heads, dff, rate, split_head, global_heads, fill_cont, lstm=False):
        self.d_model = d_model
        self.num_layers = num_layers
        self.lstm = lstm
        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.linear = keras.layers.Dense(self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

    def create_model(self, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        my_emb = keras.layers.Input(shape = (None,128),  name = 'given_embed')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        attention_weights = {}

        x = my_emb
        mask = create_padding_mask(input_seq)
        
        x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True), name='lstm')(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]
        
        # x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        self.attention_weights = attention_weights
        # x = self.fnn2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.01)
        out = layers.Dense(13, activation = 'sigmoid', name='my_last_dense')(x)

        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding, my_emb] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding, my_emb] , outputs = out)

        return model


def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)

def precision_recall_AUC(y_true, y_pred_classes):
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred_classes.flatten())
    # pre_order = np.argsort(precision)
    # return auc(precision[pre_order], recall[pre_order])
    return auc(recall,precision)
