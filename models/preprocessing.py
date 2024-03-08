import tensorflow as tf
from models import features

IMAGE_KEY = features.FEATURE_KEYS[0]
LABEL_KEY = features.LABEL_KEY
transformed_name = features.transformed_name


def preprocessing_fn(inputs):
  
  outputs = {}

  #We use tf.map_fn since we are processing batches of images
  image_features = tf.map_fn(
      lambda x: tf.io.decode_png(x[0], channels=1),
      inputs[IMAGE_KEY],
      dtype=tf.uint8)
  image_features = tf.cast(image_features, tf.float32)
  image_features = tf.image.resize(image_features, [28, 28])

  outputs[transformed_name(IMAGE_KEY)] = image_features
  outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

  return outputs