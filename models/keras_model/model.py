from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio

from models import features
from models.keras_model import constants

def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(features.LABEL_KEY)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    return {'outputs': outputs}

  return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=128):
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=features.transformed_name(features.LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema).repeat()


def build_keras_model(hidden_units, learning_rate):

  inputs = tf.keras.layers.Input(name = features.transformed_name("image") ,shape=(28,28,1))
  x = inputs
  for num_units in hidden_units:
    x = tf.keras.layers.Conv2D(num_units, kernel_size = (3,3), padding = 'same', activation='relu')(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x)

  model = tf.keras.Model(inputs, outputs)
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=['accuracy'])

  model.summary(print_fn=logging.info)

  return model


# TFX Trainer will call this function.
def run_fn(fn_args):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output, constants.TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output, constants.EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = build_keras_model(
        hidden_units=constants.HIDDEN_UNITS,
        learning_rate=constants.LEARNING_RATE)

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='epoch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)