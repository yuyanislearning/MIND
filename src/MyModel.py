import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize, _keras_api_gauge
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context

from tensorflow.python.eager import backprop

from datetime import datetime
from packaging import version


import pandas as pd
from spektral.layers import GCNConv, GlobalSumPool, GATConv

from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix

import pprint
import pdb
import sys
import importlib.util

sys.path.append("/workspace/PTM/protein_bert/")
sys.path.append('/workspace/PTM/transformers/src/')
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, log
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from spektral.layers import GCNConv, GlobalSumPool

from tensorflow.python.eager import monitoring
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.keras.utils import losses_utils




def get_mask(y_p):
  """Returns Keras mask from tensor."""
  return getattr(y_p, '_keras_mask', None)


def apply_mask(y_p, sw, mask):
  """Applies any mask on predictions to sample weights."""
  if mask is not None:
    mask = math_ops.cast(mask, y_p.dtype)
    if sw is not None:
      mask, _, sw = (
          tf_losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sw))
      sw *= mask
    else:
      sw = mask
  return sw

def match_dtype_and_rank(y_t, y_p, sw):
    """Match dtype and rank of predictions."""
    if y_t.shape.rank == 1 and y_p.shape.rank == 2:
        y_t = array_ops.expand_dims_v2(y_t, axis=-1)
    if sw is not None:
        if sw.shape.rank == 1 and y_p.shape.rank == 2:
            sw = array_ops.expand_dims_v2(sw, axis=-1)

    # Dtype.
    # This is required mainly for custom loss functions which do not take care
    # casting dtypes.
    if ((y_t.dtype.is_floating and y_p.dtype.is_floating) or
        (y_t.dtype.is_integer and y_p.dtype.is_integer)):
        y_t = math_ops.cast(y_t, y_p.dtype)

    if sw is not None:
        sw = math_ops.cast(sw, y_p.dtype)
    return y_t, y_p, sw

class MyLossesContainer(compile_utils.LossesContainer):
    def __call__(self,
               y_true,
               y_pred,
               sample_weight=None,
               regularization_losses=None):
        # pdb.set_trace()
        y_true = self._conform_to_outputs(y_pred, y_true)
        sample_weight = self._conform_to_outputs(y_pred, sample_weight)

        if not self._built:
            self.build(y_pred)

        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true)
        sample_weight = nest.flatten(sample_weight)

        loss_values = []  # Used for gradient calculation.
        loss_metric_values = []  # Used for loss metric calculation.
        batch_dim = None
        zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                    self._per_output_metrics)
        for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
            if y_t is None or loss_obj is None:  # Ok to have no loss for an output.
                continue

            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            sw = apply_mask(y_p, sw, get_mask(y_p))
            #y_p = tf.expand_dims(y_p, -1)
            loss_value = loss_obj(y_t, y_p, sample_weight=sw)

            loss_metric_value = loss_value
            # Correct for the `Mean` loss metrics counting each replica as a batch.
            if loss_obj.reduction == losses_utils.ReductionV2.SUM:
                loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync

            if batch_dim is None:
                batch_dim = array_ops.shape(y_t)[0]
            if metric_obj is not None:
                metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)

            if loss_weight is not None:
                loss_value *= loss_weight
                loss_metric_value *= loss_weight

            if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
                loss_obj.reduction == losses_utils.ReductionV2.AUTO):
                loss_value = losses_utils.scale_loss_for_distribution(loss_value)

            loss_values.append(loss_value)
            loss_metric_values.append(loss_metric_value)

        if regularization_losses:
            regularization_losses = losses_utils.cast_losses_to_common_dtype(
                regularization_losses)
            reg_loss = math_ops.add_n(regularization_losses)
            loss_metric_values.append(reg_loss)
            loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

        if loss_values:
            loss_metric_values = losses_utils.cast_losses_to_common_dtype(
                loss_metric_values)
            total_loss_metric_value = math_ops.add_n(loss_metric_values)
            self._loss_metric.update_state(
                total_loss_metric_value, sample_weight=batch_dim)

            loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
            total_loss = math_ops.add_n(loss_values)
            return total_loss
        else:
        # Ok for a model to have no compiled loss.
            return array_ops.zeros(shape=())


class MyModel(keras.models.Model):
    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        #print("Successfully hacked in !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        #pdb.set_trace()
        return {m.name: m.result() for m in self.metrics}

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        _keras_api_gauge.get_cell('compile').set(True)
        with self.distribute_strategy.scope():
            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)
            self.compiled_loss = MyLossesContainer(
                loss, loss_weights, output_names=self.output_names)
            self.compiled_metrics = compile_utils.MetricsContainer(
                metrics, weighted_metrics, output_names=self.output_names)

            experimental_steps_per_execution = kwargs.pop(
                'experimental_steps_per_execution', 1)
            self._configure_steps_per_execution(experimental_steps_per_execution)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True

            self.loss = loss or {}  # Backwards compat.

