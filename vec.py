import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import tensorboard
from pprint import pprint
import os

# del os.environ['TF_GPU_THREAD_MODE']
#os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
#print(os.environ["TF_GPU_THREAD_MODE"])
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

class MyAverageLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    # @tf.function
    def call(self, inputs, *args, **kwargs):
        retval = tf.reduce_mean(inputs, axis=2)
        return retval

class MySqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    # @tf.function
    def call(self, inputs, *args, **kwargs):
        #print(inputs.shape)
        retval = tf.squeeze(inputs, axis=-1)
        return retval

class MyTextVectorization(tf.keras.layers.TextVectorization):
    def __init__(
        self, 
        max_tokens=None, 
        standardize="lower_and_strip_punctuation", 
        split="whitespace", 
        ngrams=None, 
        output_mode="int", 
        output_sequence_length=None, 
        pad_to_max_tokens=False, 
        vocabulary=None, 
        idf_weights=None, 
        sparse=False, 
        ragged=True,
         **kwargs):
        super().__init__(
            max_tokens, 
            standardize, 
            split, 
            ngrams, 
            output_mode, 
            output_sequence_length, 
            pad_to_max_tokens, 
            vocabulary, 
            idf_weights, 
            sparse, 
            ragged, 
            **kwargs)

    # @tf.function
    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        if len(inputs.shape) > 1:
            inputs = inputs.with_row_splits_dtype(tf.int32)
            # tf.print('inputS:', inputs.shape, inputs.dtype)
            # list_to_be_stacked=list()



            flat_vals = inputs.flat_values
            # sq_fl_vals = tf.squeeze(flat_vals, axis=0)
            vectorized_flat = super().call(flat_vals)
            # tf.print(vectorized_flat.shape)
            #tf.print('!!!!!!!!', inputs.nested_value_rowids())
            nested_value_rowids = list(inputs.nested_value_rowids())[0]
            # nested_value_rowids = inputs.nested_value_rowids()

            retval = tf.RaggedTensor.from_value_rowids(values=vectorized_flat.with_row_splits_dtype(tf.int32), value_rowids=nested_value_rowids)
            # tf.print(retval)
            




            # tf.print(1)
            ##for inp in inputs:
            #     # tf.print('__________inP:', inp.shape, inp.dtype)
            #     # tf.print(2)
            ##    list_to_be_stacked.append(super().call(inp)) ## works fine
            # tf.print(3)

            #tuple_to_be_stacked=list(super().call(inp) for inp in inputs)

            ##stacked_tensors = tf.stack(list_to_be_stacked, axis=0)
            ##stacked_tensors2 = tf.cast(stacked_tensors, dtype=tf.int32)
            #tf.print('stacke_list dims', stacked_tensors.shape)
            # tf.print(4)

        #return stacked_tensors2
        return retval

list_ds = tf.data.Dataset.list_files('*.pkl')
# df = pd.read_pickle('df.pkl')
ds = tf.data.Dataset.from_tensor_slices(
    dict(
            pd.concat(
                
                    [pd.read_pickle(
                        str(x.numpy()).strip("b'")
                    ) for x in list_ds], axis=0
                 
            )
        )
)
print('SUCCESS LOAD')
# for y in ds.take(1):
#     pprint(y)

texts_ds = ds.map(lambda x: x['service_id'])


# for x in texts_ds.take(1):
#     pprint(x)

qids = ds.map(lambda x: x['qid'])

qids_vocabulary = tf.keras.layers.IntegerLookup()
qids_vocabulary.adapt(qids.batch(256))
key_f = lambda key: qids_vocabulary(key['qid'])
reduce_f = lambda _, dataset: dataset.batch(100)
ds_ragged = ds.group_by_window(
    key_func=key_f,
    reduce_func=reduce_f,
    window_size=10,
)
ds_ragged_applied = ds_ragged.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32)).prefetch(tf.data.AUTOTUNE)

class MyAverageLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    # @tf.function
    def call(self, inputs, *args, **kwargs):
        retval = tf.reduce_mean(inputs, axis=2)
        return retval




class MyModel(tfrs.models.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task = tfrs.tasks.Ranking(
            loss=tfr.keras.losses.ListMLELoss(ragged=True),
            metrics=[tfr.keras.metrics.MeanAveragePrecisionMetric(name='MAP@5', ragged=True, topn=5)]
        )


        layer_0_mytextvectorization = MyTextVectorization(ragged=True) # pad_to_max_tokens = ?
        layer_0_mytextvectorization.adapt(texts_ds)
        layer_1_embedding = tf.keras.layers.Embedding(input_dim=len(layer_0_mytextvectorization.get_vocabulary())+2, output_dim=64)
        layer_2_pooling_alt = MyAverageLayer()
        layer_dense = tf.keras.layers.Dense(1)
        layer_squeeze = MySqueezeLayer()

        self.layer_0_mytextvectorization = layer_0_mytextvectorization
        self.layer_1_embedding = layer_1_embedding
        self.layer_2_pooling_alt = layer_2_pooling_alt
        self.layer_dense = layer_dense
        self.layer_squeeze = layer_squeeze

        # self.seq_model = tf.keras.models.Sequential([
        #     layer_0_mytextvectorization,
        #     layer_1_embedding,
        #     layer_2_pooling_alt,
        #     layer_dense,
        #     layer_squeeze,
        # ])

        # self.seq_model.build(input_shape=(8,None,))

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        serv_id = inputs.get('service_id')
        out_0 = self.layer_0_mytextvectorization(serv_id)
        #out_0 = tf.cast(out_0, dtype=tf.int32)
        out_1 = self.layer_1_embedding(out_0)
        out_2 = self.layer_2_pooling_alt(out_1)
        out_3 = self.layer_dense(out_2)
        out_4 = self.layer_squeeze(out_3)
        #return self.seq_model.call(serv_id, training, mask)
        return out_4

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        y_true = inputs.pop('label')
        y_pred = self.call(inputs, training=training)
        loss = self.task(labels=y_true, predictions=y_pred)
        # print(loss)
        return loss

mdl = MyModel()
optimizer = tf.keras.optimizers.Adagrad()
# mdl.build(input_shape=(None,))
mdl.compile(optimizer=optimizer, run_eagerly=False)

# assert mdl.built

# for x in ds_ragged_applied.take(1):
#     rv = mdl(x)
# print('Forward pass - ok', rv)

# assert mdl.built
from datetime import datetime
#logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M")
#tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq=0, profile_batch=100)

mdl.fit(
    ds_ragged_applied,
    epochs=5, 
    #callbacks=[tboard_callback]
    )
print('Backward pass - ok')
