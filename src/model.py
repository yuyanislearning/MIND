import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix
from .tokenization import  n_tokens

import pprint



class Raw_model():
    def __init__(self, optimizer , loss_object):
        self.optimizer = optimizer
        self.loss_object = loss_object
    def train(self, encoded_train_set, encoded_valid_set, seq_len, batch_size, n_epochs, lr = None, callbacks=[]):
        
        train_X, train_Y, train_sample_weights = encoded_train_set.X, encoded_train_set.Y, encoded_train_set.sample_weights

        val_set = (encoded_valid_set.X, encoded_valid_set.Y, encoded_valid_set.sample_weights)

        self.model.fit(train_X, train_Y, sample_weight = train_sample_weights, batch_size = batch_size, epochs = n_epochs, validation_data = val_set, \
                callbacks = callbacks)
    def eval(self, seq_len,test_data, batch_size, unique_labels):
        test_X, test_Y, test_sample_weights = test_data.X, test_data.Y, test_data.sample_weights
        y_pred = self.model.predict(test_X, batch_size)
        
        y_mask = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
        y_true = test_Y.reshape(-1, seq_len, len(unique_labels))
        y_pred = y_pred.reshape(-1, seq_len, len(unique_labels))

        y_trues = [y_true[:,:,i][y_mask[:,:,i]==1] for i in range(len(unique_labels))]
        y_preds = [y_pred[:,:,i][y_mask[:,:,i]==1] for i in range(len(unique_labels))]

        ptm_type = {i:p for i, p in enumerate(unique_labels)}
        AUC = {}
        PR_AUC = {}
        for i in range(len(unique_labels)):
            AUC[ptm_type[i]] = roc_auc_score(y_trues[i], y_preds[i]) 
            PR_AUC[ptm_type[i]] = precision_recall_AUC(y_trues[i], y_preds[i]) 
            confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues[i], y_preds[i]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
            print(ptm_type[i]+' confusion matrix')
            print(confusion_matrixs)
            confusion_matrixs=None
        
                
        print('PR_AUC')
        pprint.pprint(PR_AUC)
        
        print('AUC')
        pprint.pprint(AUC)

class RNN_model(Raw_model):
    def __init__(self, optimizer_class, loss_object):
        Raw_model.__init__(self, optimizer_class, loss_object )
    def create_model(self, seq_len, d_hidden_seq, unique_labels):
        model = keras.Sequential()
        # input_seq = keras.layers.Input(shape = (seq_len,), dtype = np.int32, name = 'input-seq')
        model.add(layers.Embedding(n_tokens, d_hidden_seq, name = 'embedding-seq-input')) 
        model.add(
            layers.Bidirectional(layers.LSTM(128, input_dim=seq_len, return_sequences=True)))
        model.add(layers.Dense(len(unique_labels), activation = 'sigmoid'))
        model.add(layers.Reshape((-1,)))

        self.model = model
        self.model.compile(
            loss = self.loss_object,
            optimizer = self.optimizer,
        )




def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)



def precision_recall_AUC(y_true, y_pred_classes):
    precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred_classes.flatten())
    pre_order = np.argsort(precision)
    return auc(precision[pre_order], recall[pre_order])