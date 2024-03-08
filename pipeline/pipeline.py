from typing import Any, Dict, List, Optional
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    input_config: example_gen_pb2.Input,
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:

    """Implements the pipeline with TFX Components."""

    components = []

    # ImportExampleGen
    example_gen = tfx.components.ImportExampleGen(input_base=data_path, input_config=input_config)
    components.append(example_gen)
    
    # StatisticsGen
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # SchemaGen
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'])
    components.append(schema_gen)

    # ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Tranform
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=preprocessing_fn)
    components.append(transform)

    # Trainer
    trainer_args = {
        'run_fn': run_fn,
        'examples': transform.outputs['transformed_examples'],
        'schema': schema_gen.outputs['schema'],
        'transform_graph': transform.outputs['transform_graph'],
        'train_args': train_args,
        'eval_args': eval_args,
        }
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    # Resolver
    model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing)).with_id('latest_blessed_model_resolver')
    components.append(model_resolver)

    # Evaluator
    eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key='label_xf',
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(class_name='ExampleCount'),
              tfma.MetricConfig(
                  class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': eval_accuracy_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ]
    )
    evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)
    components.append(evaluator)

    # Pusher
    pusher_args = {
        'model': trainer.outputs['model'],
        'model_blessing': evaluator.outputs['blessing'],
    }

    pusher_args['push_destination'] = tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir))
    pusher = tfx.components.Pusher(**pusher_args)
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
    )
