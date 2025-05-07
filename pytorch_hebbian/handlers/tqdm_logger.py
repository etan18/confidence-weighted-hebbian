# import numbers
# import warnings

# import torch
# from ignite.contrib.handlers.base_logger import BaseOutputHandler, BaseLogger


# class OutputHandler(BaseOutputHandler):
#     """Helper handler to log engine's output and/or metrics.

#     Args:
#         tag (str): common title for all produced plots. For example, 'training'
#         metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
#             metrics.
#         output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
#             For example, `output_transform = lambda output: output`
#             This function can also return a dictionary, e.g `{'loss': loss1, 'another_loss': loss2}` to label the plot
#             with corresponding keys.
#         global_step_transform (callable, optional): global step transform function to output a desired global step.
#     """

#     def __init__(self, tag, metric_names="all", output_transform=None, global_step_transform=None):
#         super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)

#     def __call__(self, engine, logger, event_name):

#         if not isinstance(logger, TqdmLogger):
#             raise RuntimeError("Handler 'OutputHandler' works only with TqdmLogger")

#         metrics = self._setup_output_metrics(engine)

#         global_step = self.global_step_transform(engine, event_name)

#         if not isinstance(global_step, int):
#             raise TypeError(
#                 "global_step must be int, got {}."
#                 " Please check the output of global_step_transform.".format(type(global_step))
#             )

#         message = "{} epoch {}: ".format(self.tag.capitalize(), global_step)
#         metrics_str = []
#         for key, value in metrics.items():
#             if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
#                 if value > 1e4:
#                     metrics_str.append("{}={:.4e}".format(key, value))
#                 else:
#                     metrics_str.append("{}={:.4f}".format(key, value))
#             elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
#                 for i, v in enumerate(value):
#                     metrics_str.append("{}{}={}".format(key, i, v.item()))
#             else:
#                 warnings.warn(
#                     "TqdmLogger output_handler can not log " "metrics value type {}".format(type(value))
#                 )
#         logger.pbar.log_message(message + ", ".join(metrics_str))


# class TqdmLogger(BaseLogger):
#     """Tqdm logger to log messages using the progress bar."""

#     def __init__(self, pbar):
#         self.pbar = pbar

#     def close(self):
#         if self.pbar:
#             self.pbar.close()
#         self.pbar = None

#     def _create_output_handler(self, *args, **kwargs):
#         return OutputHandler(*args, **kwargs)

#     def _create_opt_params_handler(self, *args, **kwargs):
#         """Intentionally empty"""
#         pass

import numbers
import warnings
import torch
from ignite.contrib.handlers.base_logger import BaseOutputHandler, BaseLogger


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

    Args:
        tag (str): common title for all produced logs. For example, 'training'
        metric_names (list of str or "all", optional): list of metric names to log or "all" for all available metrics.
        output_transform (callable, optional): function to transform engine.state.output into a number or a dictionary.
            This can be used to log custom values in addition to or instead of metrics.
        global_step_transform (callable, optional): function to produce a global step from the engine and event_name.
    """

    def __init__(self, tag, metric_names="all", output_transform=None, global_step_transform=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TqdmLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TqdmLogger")

        # Gather metrics
        if self.metric_names == "all":
            metrics = dict(engine.state.metrics)
        elif isinstance(self.metric_names, list):
            metrics = {name: engine.state.metrics[name] for name in self.metric_names if name in engine.state.metrics}
        else:
            metrics = {}

        # Apply output_transform if provided
        if self.output_transform is not None:
            output = self.output_transform(engine.state.output)
            if isinstance(output, dict):
                metrics.update(output)
            else:
                metrics["output"] = output

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        message = f"{self.tag.capitalize()} epoch {global_step}: "
        metrics_str = []
        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or (isinstance(value, torch.Tensor) and value.ndimension() == 0):
                # Format large numbers in scientific notation
                if value > 1e4:
                    metrics_str.append(f"{key}={value:.4e}")
                else:
                    metrics_str.append(f"{key}={value:.4f}")
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    metrics_str.append(f"{key}{i}={v.item()}")
            else:
                warnings.warn(
                    f"TqdmLogger output_handler can not log metrics value type {type(value)}"
                )
        logger.pbar.log_message(message + ", ".join(metrics_str))


class TqdmLogger(BaseLogger):
    """Tqdm logger to log messages using the progress bar."""

    def __init__(self, pbar):
        self.pbar = pbar

    def close(self):
        if self.pbar:
            self.pbar.close()
        self.pbar = None

    def _create_output_handler(self, *args, **kwargs):
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args, **kwargs):
        """Intentionally empty"""
        pass

