import copy
import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

import torch
from ignite.contrib.handlers import ProgressBar, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine, OptimizerParamsHandler
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import HebbianEvaluator, SupervisedEvaluator
from pytorch_hebbian.handlers.tensorboard_logger import WeightsImageHandler
from pytorch_hebbian.handlers.tqdm_logger import TqdmLogger
from pytorch_hebbian.learning_rules import KrotovsRule, RobustHebbsRule
from pytorch_hebbian.metrics import UnitConvergence
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import RobustHebbianTrainer, SupervisedTrainer

PATH = os.path.dirname(os.path.abspath(__file__))



def attach_handlers(run, model, optimizer, learning_rule, trainer, evaluator, train_loader, val_loader, params):
    # Metrics
    UnitConvergence(model[0], learning_rule.norm, device=trainer.device).attach(trainer.engine, 'unit_conv')

    # Progress Bar
    pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
    pbar.attach(trainer.engine, metric_names='all')

    # Evaluator
    evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED(every=100), train_loader, val_loader)

    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 1 - epoch / params['epochs']
    )
    lr_scheduler = LRScheduler(lr_scheduler)
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    # Model checkpointing
    mc_handler = ModelCheckpoint(
        dirname=config.MODELS_DIR,
        filename_prefix=run.replace('/', '-'),
        n_saved=1,
        create_dir=True,
        require_empty=False,
        global_step_transform=global_step_from_engine(trainer.engine)
    )
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    # TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=os.path.join(config.TENSORBOARD_DIR, run))
    images, labels = next(iter(train_loader))
    tb_logger.writer.add_graph(copy.deepcopy(model).cpu(), images.cpu())
    tb_logger.writer.add_hparams(params, {})

    # Validation metrics
    tb_logger.attach(
        evaluator.engine,
        log_handler=OutputHandler(
            tag="validation",
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer.engine)
        ),
        event_name=Events.COMPLETED
    )

    # Training metrics
    tb_logger.attach(
        trainer.engine,
        log_handler=OutputHandler(
            tag="train",
            metric_names=["unit_conv"]
        ),
        event_name=Events.EPOCH_COMPLETED
    )

    # Weights image handler (custom)
    input_shape = tuple(next(iter(train_loader))[0].shape[1:])
    tb_logger.attach(
        trainer.engine,
        log_handler=WeightsImageHandler(model, input_shape),
        event_name=Events.EPOCH_COMPLETED
    )

    # Optimizer parameters
    tb_logger.attach(
        trainer.engine,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.EPOCH_STARTED
    )

    return tb_logger

def main(args: Namespace, params: dict, dataset_name, run_postfix=""):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = '{}/rob-heb/{}'.format(dataset_name, identifier)
    if run_postfix:
        run += '-' + run_postfix
    print("Starting run '{}'".format(run))

    # Loading the model
    if dataset_name == "mnist" or dataset_name == "mnist-fashion":
        model = models.create_conv1_model(28, 1, num_kernels=400, n=1, batch_norm=True)
    elif dataset_name == "cifar-10":
        model = models.create_conv1_model(
            input_dim=32,             # Input image size
            input_channels=3,         # RGB images
            num_kernels=64,           # Number of convolutional filters
            kernel_size=5,            # Convolution kernel size
            pool_size=2,              # Pooling size
            n=1,                      # Parameter for RePU
            batch_norm=True,          # Include batch normalization
            dropout=0.5               # Include dropout
        )
    else:
        raise ValueError(f"Dataset name not recognized: {dataset_name}")
        
    if args.initial_weights is not None:
        raise ValueError("Initial weights should not be used for training from scratch.")
    model.to(utils.get_device(args.device))
    print("Device set to '{}'.".format(args.device))

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name, subset=10000, pca=True)

    # Define learning rule, evaluator, and trainer
    learning_rule = RobustHebbsRule()
    optimizer = Local(named_params=model.named_parameters(), lr=params['lr'])
    evaluator = HebbianEvaluator(
        model=model,
        score_name='accuracy',
        score_function=lambda engine: engine.state.metrics['accuracy'],
        epochs=500,
        supervised_from=-1
    )
    trainer = RobustHebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, device=args.device)

    # Attach TensorBoard logger and other handlers
    tb_logger = attach_handlers(run, model, optimizer, learning_rule, trainer, evaluator, train_loader, val_loader,
                                params)

    # Validation after every epoch
    @trainer.engine.on(Events.EPOCH_COMPLETED)
    def validate_after_epoch(engine):
        print(f"Running validation for epoch {engine.state.epoch}...")
        evaluator.run(train_loader=train_loader, val_loader=val_loader)  # Manually call evaluator

        # Access validation metrics
        val_metrics = evaluator.engine.state.metrics
        accuracy = val_metrics.get("accuracy", None)
        if accuracy is not None:
            print(f"Epoch {engine.state.epoch}: Validation Accuracy: {accuracy:.4f}")
            tb_logger.writer.add_scalar("validation/accuracy", accuracy, engine.state.epoch)
        else:
            print("Validation accuracy not available.")

    # Run the trainer
    trainer.run(train_loader=train_loader, epochs=params['epochs'])

    # Close the TensorBoard logger
    tb_logger.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, required=True,
                        help='use a preset json file to specify parameters')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug logging')
    parser.add_argument("--log_dir", type=str, default=config.TENSORBOARD_DIR,
                        help="log directory for Tensorboard log output")
    parser.add_argument('--initial_weights', type=str, default=None,
                        help='model weights to initialize training')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help="'cuda' (GPU) or 'cpu'")
    parser.add_argument('--dataset', type=str, required=True, 
                        help="'mnist' or 'mnist-fashion' or 'cifar-10'")

    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.DEBUG if args_.debug else logging.INFO)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    main(args_, params_, dataset_name=args_.dataset)
