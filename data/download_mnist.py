import tensorflow_datasets as tfds

data_dir = "."
tfds.load("mnist", split= ["all"], data_dir= data_dir)