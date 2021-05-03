import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ.pop('TF_CONFIG', None)
import tensorflow as tf
import numpy as np
#tf.random.set_seed(10)

def mnist_dataset(batch_size):
	(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
	# The `x` arrays are in uint8 and have values in the range [0, 255].
	# You need to convert them to float32 with values in the range [0, 1]
	x_train = x_train / np.float32(255)
	y_train = y_train.astype(np.int64)
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(x_train, y_train)).batch(batch_size)
		#(x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
	return train_dataset

def model():
	model = tf.keras.Sequential([
		tf.keras.Input(shape=(28, 28)),
		tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(10)
	])
	model.compile(
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
		metrics=['accuracy'])
	return model



batch_size = 1000
#single_worker_dataset = mnist_dataset(batch_size)
#single_worker_model = model()
#single_worker_model.fit(single_worker_dataset, epochs=10, steps_per_epoch=600)
#exit()

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456', 'localhost:34567']
    },
    'task': {'type': 'worker', 'index': int(sys.argv[1])}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)


tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

comm_opts = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=comm_opts)

global_batch_size = batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)
#dataset_no_auto_shard = multi_worker_dataset.with_options(options)


with strategy.scope():
	# Model building/compiling need to be within `strategy.scope()`.
	multi_worker_model = model()

multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=20)
#multi_worker_model.fit(dataset_no_auto_shard, epochs=3, steps_per_epoch=70)

