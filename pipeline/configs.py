from tfx.proto import example_gen_pb2

PIPELINE_NAME = 'mnist_pipeline'

INPUT_CONFIG = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='mnist-train*'),
        example_gen_pb2.Input.Split(name='eval', pattern='mnist-test*')]
    )

PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.keras_model.model.run_fn'

NUM_EPOCHS = 10
TRAIN_NUM_STEPS = int(60000/128) * NUM_EPOCHS
EVAL_NUM_STEPS = int(10000/128)

EVAL_ACCURACY_THRESHOLD = 0.9

