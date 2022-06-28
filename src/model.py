import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
# import neural_structured_learning as nsl

# from tensorflow.python.keras.engine import data_adapter
# from tensorflow.python.keras.engine.training import _minimize
# from tensorflow.python.util import nest
# from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.eager import backprop

from datetime import datetime
from packaging import version


import pandas as pd
from spektral.layers import GCNConv, GlobalSumPool, GATConv


from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from .tokenization import  n_tokens
# from .MyModel import MyModel

# import pprint
import pdb
import sys
# import importlib.util

# from src.utils import PTMDataGenerator, get_graph
from src.transformer import positional_encoding, EncoderLayer, create_padding_mask

sys.path.append("/workspace/PTM/protein_bert/")
sys.path.append('/workspace/PTM/transformers/src/')
from spektral.layers import GCNConv, GlobalSumPool


class Raw_model():
    def __init__(self):
        next
    def train(self,  aug, val_aug, seq_len, batch_size, n_epochs, unique_labels, lr = None, callbacks=[], binary=None,ind=None, graph=False, num_cont=None):
        # aug = PTMDataGenerator( encoded_train_set, seq_len,model=self.model_name, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=False, num_cont=num_cont, d_model=self.d_model)


        # val_aug = PTMDataGenerator( encoded_valid_set, seq_len,model=self.model_name, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=True, num_cont=num_cont, d_model=self.d_model)
        # logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # callbacks.append(keras.callbacks.TensorBoard(log_dir=logdir))
        #self.model.fit_generator(aug, validation_data = val_set,  epochs=n_epochs, callbacks = callbacks)
        # for i in range(n_epochs):
        #     if i%1==10 and i!=0:# 
        #         self.eval(seq_len, encoded_train_set, batch_size, unique_labels, graph=graph,binary=binary, train=True)
        #     print('================%d epoch==============='%i)
            # self.model.fit(aug,  epochs = 1, steps_per_epoch=aug.__len__(), validation_data = val_set, callbacks = callbacks)
        # if batch_size > len(aug.list_id):
        #     # in case batch size greater than sample size
        #     aug = PTMDataGenerator( encoded_train_set, seq_len,model=self.model_name, batch_size=len(aug.list_id), unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=False, num_cont=num_cont, d_model=self.d_model)
        # if batch_size > len(val_aug.list_id) and len(val_aug.list_id)!=0:
        #     val_aug = PTMDataGenerator( encoded_valid_set, seq_len,model=self.model_name, batch_size=len(val_aug.list_id), unique_labels=unique_labels, graph = graph,shuffle=True, binary=binary, ind=ind, eval=True, num_cont=num_cont, d_model=self.d_model)
        self.model.fit(aug,  epochs = n_epochs, steps_per_epoch=aug.__len__(), validation_data = val_aug, callbacks = callbacks)#, validation_steps=val_aug.__len__())


    def eval(self, seq_len,aug, batch_size, unique_labels,graph=False, binary=False, ind=None, num_cont=None):
        # aug = PTMDataGenerator( test_data, seq_len,model=self.model_name, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont, d_model=self.d_model, eval=True)#test_data.Y.shape[0]
        # aug = PTMDataGenerator( test_data, seq_len, model=self.model_name,batch_size=len(aug.list_id)//5+1, unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont, d_model=self.d_model, eval=True)#test_data.Y.shape[0]
        # test_X, test_Y, test_sample_weights = aug.__getitem__(0)
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
            # seq_len = test_X[0].shape[1]
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
            # seq_len = test_X[0].shape[1]
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


class RNN_model(Raw_model):
    def __init__(self):
        Raw_model.__init__(self)
    def create_model(self, model, optimizer, seq_len, d_hidden_seq, unique_labels,dropout,is_binary=None, graph=False, n_lstm=3, n_gcn=None):
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)
        lstm = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True), name='lstm-1')(encoding)
        lstm = tf.nn.leaky_relu(lstm, alpha=0.01)
        lstm = tf.keras.layers.Dropout(0.6)(lstm)

        lstm = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True),  name='lstm-2')(lstm)
        lstm = tf.nn.leaky_relu(lstm, alpha=0.01)
        lstm = tf.keras.layers.Dropout(0.6)(lstm)

        last_hidden_layer = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True), name='lstm-3')(lstm)
        lstm = tf.nn.leaky_relu(lstm, alpha=0.01)
        lstm = tf.keras.layers.Dropout(0.6)(lstm)
        
        if graph:
            adj_input = keras.layers.Input(shape=(seq_len, seq_len), name = 'input-adj')
            last_hidden_layer = GATConv(channels=128, name='gcn-1' )([last_hidden_layer, adj_input])
            last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-1')(last_hidden_layer)
            assert n_gcn > 0
            for i in range(n_gcn-1):
                last_hidden_layer = GATConv(channels=128, name = 'gcn-'+str(i+2) )([last_hidden_layer, adj_input])
                last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-'+str(i+2))(last_hidden_layer)
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels), activation = 'sigmoid', name='my_last_dense')(last_hidden_layer)

        if not is_binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq, adj_input] if graph else [input_seq]
        model = keras.models.Model(inputs = model_input , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,
            optimizer = optimizer,
            #run_eagerly=True,
        )

class GAT_model(Raw_model):
    def __init__(self):
        Raw_model.__init__(self)
    def create_model(self,model, optimizer,num_layers, seq_len,is_binary, unique_labels,d_hidden_seq, dropout):
        input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        adj_input = keras.layers.Input(shape=(seq_len, seq_len), name = 'input-adj')
        encoding = layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')(input_seq)

        last_hidden_layer = GATConv(channels=128, attn_heads=8,dropout_rate=dropout, activation=None, name='gcn-1' )([encoding, adj_input])
        last_hidden_layer = tf.nn.leaky_relu(last_hidden_layer, alpha=0.01)
        # last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-1')(last_hidden_layer)
        assert num_layers > 0
        for i in range(num_layers-1):
            last_hidden_layer = GATConv(channels=128, attn_heads=8,dropout_rate=dropout, activation=None,name = 'gcn-'+str(i+2) )([last_hidden_layer, adj_input])
            last_hidden_layer = tf.nn.leaky_relu(last_hidden_layer, alpha=0.01)
            # last_hidden_layer = keras.layers.Dropout(dropout, name='dropout-'+str(i+2))(last_hidden_layer)

        # for i in range(3):#TODO
        #     last_hidden_layer = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True),  name='lstm-'+str(i+1))(last_hidden_layer)
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(last_hidden_layer) if is_binary else layers.Dense(len(unique_labels), activation = 'sigmoid', name='my_last_dense')(last_hidden_layer)

        if not is_binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq, adj_input]
        model = keras.models.Model(inputs = model_input , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,
            optimizer = optimizer,
            #run_eagerly=True,
        )



class CNN_model(Raw_model):
    def __init__(self):
        Raw_model.__init__(self)
        self.CNN1 = layers.Conv1D(256, 3,padding='same', activation=None )
        self.pooling1 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN2 = layers.Conv1D(256, 6,padding='same', activation=None )
        self.pooling2 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN3 = layers.Conv1D(256, 9,padding='same', activation=None )
        self.pooling3 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN4 = layers.Conv1D(256, 12,padding='same', activation=None )
        self.dropout = tf.keras.layers.Dropout(0.6)
        self.batchnorm = layers.BatchNormalization()
        
    def create_model(self,model, optimizer, seq_len,is_binary, unique_labels ):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        # encoding = keras.layers.CategoryEncoding(n_tokens, output_mode='one_hot', name = 'embedding-seq-input')(input_seq)
        x = tf.one_hot(input_seq, n_tokens, 1.0, 0.0, axis=-1)

        x = self.CNN1(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        # res_x = x
        # x = self.padding1(x)
        # x = self.dropout(x)
        x = self.CNN2(x)
        x = tf.nn.leaky_relu(x, alpha=0.01) #+ res_x
        # res_x = x
        # x = self.padding2(x)
        # x = self.dropout(x) 

        x = self.CNN3(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        # x = self.dropout(x) 
        # x = x+ res_x
        # x = self.padding3(x)
        x = self.CNN4(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        # x = self.dropout(x) 

        # for i in range(3):#TODO
        #     last_hidden_layer = layers.Bidirectional(layers.LSTM(d_hidden_seq,  return_sequences=True),  name='lstm-'+str(i+1))(last_hidden_layer)
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if is_binary else \
            layers.Dense(len(unique_labels), activation = 'sigmoid', name='my_last_dense')(x)

        if not is_binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)
        model_input = [input_seq]
        model = keras.models.Model(inputs = model_input , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        self.model = model
        print(model.summary())
        self.model.compile(
            loss = loss,
            optimizer = optimizer,
            #run_eagerly=True,
        )


# class ProteinBert(Raw_model):
#     def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, multilabel=None):
#         Raw_model.__init__(self, optimizer_class, loss_object, lr )
#         pretraining_model_generator, _ = load_pretrained_model(local_model_dump_dir='/workspace/PTM/protein_bert/proteinbert',local_model_dump_file_name='epoch_92400_sample_23500000.pkl',\
#             lr = self.lr)
#         # if short:
#         #     output_spec = OutputSpec(OutputType(False, 'binary'), [0,1]) if binary else OutputSpec(OutputType(False, 'multilabel'), unique_labels)
#         if multilabel:
#             output_spec = OutputSpec(OutputType(True, 'multilabel'), unique_labels)
#         if binary:
#             output_spec = OutputSpec(OutputType(True, 'binary'), [0,1])
#         uni_l = [0,1] if binary else unique_labels
#         self.model_generator = FinetuningModelGenerator(pretraining_model_generator, output_spec, pretraining_model_manipulation_function = \
#                 get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5, unique_labels = uni_l)
#     def create_model(self, train_data,  seq_len,freeze_pretrained_layers=None,binary=None, graph=False, n_gcn=None):   
#         train_X, train_Y, _ = (train_data.X[0], train_data.Y[0], train_data.sample_weights[0]) if binary \
#             else (train_data.X, train_data.Y, train_data.sample_weights)
#         self.model_generator.dummy_epoch = (_slice_arrays(train_X, slice(0, 1)), _slice_arrays(train_Y, slice(0, 1)))
#         self.model = self.model_generator.create_model(seq_len,  freeze_pretrained_layers = freeze_pretrained_layers, graph = graph, n_gcn=n_gcn)


class TransFormer(Raw_model):
    def __init__(self,FLAGS, model_name,  optimizer, num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = FLAGS.d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.Embedding = tf.keras.layers.Embedding(n_tokens, self.d_model, name='embedding')

        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.linear = keras.layers.Dense(self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels
        self.embedding = FLAGS.embedding


    def create_model(self, seq_len, graph=False):
        if self.embedding:
            input_seq = keras.layers.Input(shape = (None,1024),  name = 'input-seq')
        else:
            input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        if self.embedding:
            msk = keras.layers.Input(shape=(None, ), name = 'mask')
            mask = msk[:, tf.newaxis, tf.newaxis, :]
            x = self.linear(input_seq)
        else:
            mask = create_padding_mask(input_seq)
            x = self.Embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        # adding embedding and position encoding.  
        # x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True), name='lstm')(x)
  
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]
        
        # x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        
        # x = self.fnn2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.01)
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), activation = 'sigmoid', name='my_last_dense')(x)
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input

        if self.embedding:
            model = keras.models.Model(inputs = [input_seq,msk, pos_encoding] , outputs = out)
        else:
            model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph \
                else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)
        print(model.summary())
        # if self.MLM:
        #     loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        #model.build(input_seq)

        # adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
        # adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)
        # self.model = adv_model
        self.model = model
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        # adversial training



        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )



class CustomTransFormer(Raw_model):
    def __init__(self, model_name,  optimizer, d_model, num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.embedding = tf.keras.layers.Embedding(n_tokens, d_model, name='embedding')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels

        self.CNN1 = layers.Conv1D(256, 3,padding='same', activation=None )
        self.pooling1 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN2 = layers.Conv1D(256, 6,padding='same', activation=None )
        self.pooling2 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN3 = layers.Conv1D(256, 9,padding='same', activation=None )
        self.pooling3 = layers.MaxPooling1D(pool_size=3,strides=1, padding='same')
        self.CNN4 = layers.Conv1D(256, 12,padding='same', activation=None )
        self.dropout = tf.keras.layers.Dropout(0.6)
        self.batchnorm = layers.BatchNormalization()

    def create_model(self, seq_len, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        
        conv_x = tf.one_hot(input_seq, n_tokens, 1.0, 0.0, axis=-1)

        conv_x = self.CNN1(conv_x)
        conv_x = tf.nn.leaky_relu(conv_x, alpha=0.01)
        # res_x = x
        # x = self.padding1(x)
        conv_x = self.dropout(conv_x)
        conv_x = self.CNN2(conv_x)
        conv_x = tf.nn.leaky_relu(conv_x, alpha=0.01) #+ res_x
        # res_x = x
        # x = self.padding2(x)
        conv_x = self.dropout(conv_x) 

        conv_x = self.CNN3(conv_x)
        conv_x = tf.nn.leaky_relu(conv_x, alpha=0.01)
        conv_x = self.dropout(conv_x) 
        
        
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        # adding embedding and position encoding.
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        x = tf.concat([x,conv_x], 2, name = 'concat')
        # x = self.fnn2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.01)
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), activation = 'sigmoid', name='my_last_dense')(x)
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input


        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)
        print(model.summary())
        # if self.MLM:
        #     loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        #model.build(input_seq)

        # adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
        # adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)
        # self.model = adv_model
        self.model = model
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        # adversial training



        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )


class TransFormer_CNN(Raw_model):
    def __init__(self, model_name,  optimizer, d_model, num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.embedding = tf.keras.layers.Embedding(n_tokens, d_model, name='embedding')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels

        self.CNN1 = layers.Conv1D(256, 1,padding='same', activation=None )
        self.CNN2 = layers.Conv1D(128, 9,padding='same', activation=None )
        self.CNN3 = layers.Conv1D(256, 10,padding='same', activation=None )

    def create_model(self, seq_len, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        # adding embedding and position encoding.
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        conv_x = self.CNN2(x)
        x = tf.nn.leaky_relu(conv_x, alpha=0.01) + x
        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), activation = 'sigmoid', name='my_last_dense')(x)
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input
        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)
        # if self.MLM:
        #     loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        #model.build(input_seq)
        self.model = model
        print(model.summary())
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )


class TransFormerGAT(Raw_model):
    def __init__(self, model_name,  optimizer, d_model, num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.embedding = tf.keras.layers.Embedding(n_tokens, d_model, name='embedding')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels

    def create_model(self, seq_len, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        # adding embedding and position encoding.
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        x = GATConv(channels=128, attn_heads=8,dropout_rate=0.5, activation='relu', name='gcn-1' )([x, adj_input])
        # x = self.dropout(last_hidden_layer)

        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), activation = 'sigmoid', name='my_last_dense')(x)
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input
        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)
        # if self.MLM:
        #     loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        #model.build(input_seq)
        self.model = model
        print(model.summary())
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )



class LSTMTransFormer(Raw_model):
    def __init__(self, FLAGS,model_name,  optimizer,  num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        Raw_model.__init__(self)
        self.optimizer = optimizer
        self.d_model = FLAGS.d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.embedding = tf.keras.layers.Embedding(n_tokens, self.d_model, name='embedding')

        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels
        self.reg = FLAGS.l2_reg

    def create_model(self, seq_len, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        # adding embedding and position encoding.
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        # x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True), name='lstm-2')(x)
        # x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True), name='lstm-3')(x)

        x = layers.Bidirectional(layers.LSTM(self.d_model//2,  return_sequences=True, \
            ), name='lstm')(x)#kernel_regularizer=tf.keras.regularizers.L2(self.reg)
            
        x = GATConv(channels=self.d_model//8, attn_heads=8,dropout_rate=0.5, activation='relu', name='gcn-1' )([x, adj_input])
        # x = GATConv(channels=self.d_model//8, attn_heads=8,dropout_rate=0.5, activation='relu', name='gcn-2' )([x, adj_input])


        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        # x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask)#self.enc_layers[i](x, mask, adj_input) if graph else 
            attention_weights[f'encoder_layer{i+1}_block1'] = block1


        out = layers.Dense(1, activation = 'sigmoid', name = 'my_last_dense')(x) if self.binary else layers.Dense(len(self.unique_labels), \
             activation = 'sigmoid', name='my_last_dense')(x)#kernel_regularizer=tf.keras.regularizers.L2(self.reg),
        #reg_loss = out.losses
        if not self.binary: 
            out = layers.Reshape((-1,1), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input
        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        loss = keras.losses.BinaryCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE) #+ reg_loss
        # if self.MLM:
        #     loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        #model.build(input_seq)
        self.model = model
        print(model.summary())
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            #run_eagerly=True,
        )


class MLMTransFormer():
    def __init__(self, model_name, optimizer_class,  d_model, num_layers, seq_len, num_heads, dff, rate, binary, unique_labels, split_head, global_heads, fill_cont):
        self.optimizer = optimizer_class
        self.d_model = d_model
        self.model_name = model_name
        self.num_layers = num_layers
        # max_seq_len = 35220
        self.embedding = tf.keras.layers.Embedding(n_tokens, d_model, name='embedding')

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, split_head, global_heads, fill_cont)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.binary = binary
        self.unique_labels = unique_labels

    def create_model(self, seq_len, graph=False):
        input_seq = keras.layers.Input(shape = (None,), dtype = np.int32, name = 'input-seq')
        if graph:
            adj_input = keras.layers.Input(shape=(None, None), name = 'input-adj')
        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        mask = create_padding_mask(input_seq)
        # adding embedding and position encoding.
        x = self.embedding(input_seq)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = keras.layers.Input(shape=(None, self.d_model), name='input_pos_encoding')
        x += pos_encoding #self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x)#, training=training

        for i in range(self.num_layers):
            x, block1 = self.enc_layers[i](x, mask, adj_input) if graph else self.enc_layers[i](x, mask)#, training
            attention_weights[f'encoder_layer{i+1}_block1'] = block1

        out = layers.Dense(len(self.unique_labels), activation = 'softmax', name='my_last_dense')(x)
        # out = layers.Reshape((-1,len(self.unique_labels)), name ='reshape')(out)

        # return out, attention_weights  # (batch_size, input_seq_len, d_model)
        # if not is_binary: 
        #     out = layers.Reshape((-1,1), name ='reshape')(out)
        # model_input = model.input
        model = keras.models.Model(inputs = [input_seq, adj_input, pos_encoding] , outputs = out) if graph else keras.models.Model(inputs = [input_seq, pos_encoding] , outputs = out)
        # loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses_utils.ReductionV2.NONE)

        #model.build(input_seq)
        self.model = model
        print(model.summary())
        # learning_rate = CustomSchedule(d_model) TODO
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
        #                              epsilon=1e-9)
        self.model.compile(
            loss = loss,
            optimizer = self.optimizer,
            metrics = keras.metrics.CategoricalAccuracy()
            #run_eagerly=True,
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
        if self.lstm:
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

class ProtTrans(Raw_model):
    def __init__(self, optimizer_class, loss_object, unique_labels,lr,  binary=None, short=None):
        Raw_model.__init__(self, optimizer_class, loss_object, lr )



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
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