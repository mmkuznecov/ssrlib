"""Probe for running processors on embeddings during training/evaluation.

Typical use:

    >>> from ssrlib.processing import CovarianceProcessor, NESumProcessor, ConditionNumberProcessor
    >>> from ssrlib.analysis import EmbeddingProbe
    >>> probe = EmbeddingProbe(
    ...     processors=[NESumProcessor(), ConditionNumberProcessor()],
    ...     every_n_epochs=5,
    ... )
    >>> for epoch in range(epochs):
    ...     train_one_epoch(...)
    ...     if probe.should_run(epoch=epoch):
    ...         emb = encode_validation_set(model)
    ...         metrics = probe(emb, epoch=epoch)
    ...         logger.info(metrics)

For zero-infrastructure use, the ``embedding_probe`` decorator wraps a
function that returns embeddings and runs the probe on every call.

Some processors (notably :class:`NeuralCollapseProcessor`) need extra context
beyond the embedding tensor itself — labels, classifier weights, etc. Pass
those as keyword arguments to the probe call; the probe inspects each
processor's signature and forwards only the kwargs it accepts:

    >>> probe(emb, labels=y, classifier_weights=W, classifier_bias=b, epoch=epoch)
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from ..processing.base import BaseProcessor

logger = logging.getLogger(__name__)


def _filter_context_for(
    processor: BaseProcessor, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Pick the subset of ``context`` that ``processor.process`` accepts.

    A processor whose signature includes ``**kwargs`` receives the entire
    context. Otherwise we keep only kwargs whose name matches a declared
    parameter of ``process``.
    """
    if not context:
        return {}
    try:
        sig = inspect.signature(processor.process)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(context)
    return {k: v for k, v in context.items() if k in params}


class EmbeddingProbe:
    """Run a list of processors on embeddings and return a flat metrics dict.

    Args:
        processors: Iterable of processors to run on each call.
        sink: optional callable invoked with the metrics dict (e.g. wandb.log).
        every_n_steps: if set, ``should_run(step=...)`` returns True only
            when ``step % every_n_steps == 0``.
        every_n_epochs: only used by ``should_run(epoch=...)``; defaults to 1.
        scalar_only: if True, vector / matrix outputs are reduced to summary
            scalars (Frobenius norm + shape) rather than expanded element-wise.
    """

    def __init__(
        self,
        processors: Iterable[BaseProcessor],
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        every_n_steps: Optional[int] = None,
        every_n_epochs: int = 1,
        scalar_only: bool = False,
    ):
        self.processors: List[BaseProcessor] = list(processors)
        if not self.processors:
            raise ValueError("EmbeddingProbe requires at least one processor")
        self.sink = sink
        self.every_n_steps = every_n_steps
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.scalar_only = bool(scalar_only)

    def should_run(
        self, step: Optional[int] = None, epoch: Optional[int] = None
    ) -> bool:
        if step is not None and self.every_n_steps:
            return step % self.every_n_steps == 0
        if epoch is not None:
            return epoch % self.every_n_epochs == 0
        return True

    def __call__(
        self,
        embeddings: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **context: Any,
    ) -> Dict[str, Any]:
        """Run all processors on ``embeddings`` and return a flat metrics dict.

        Args:
            embeddings: numpy array, torch tensor, or anything ``np.asarray``-able
                of shape (N, D).
            step, epoch: optional integer counters added to the returned dict
                and used by ``should_run``.
            **context: extra kwargs forwarded to processors that accept them.
                Common keys: ``labels``, ``classifier_weights``, ``classifier_bias``.
                A processor whose signature doesn't declare a context kwarg
                simply doesn't receive it.
        """
        emb = self._to_numpy(embeddings)
        metrics: Dict[str, Any] = {}
        for processor in self.processors:
            try:
                kwargs = _filter_context_for(processor, context)
                out = processor.process(emb, **kwargs)
                self._flatten_into(metrics, processor.name, out)
            except Exception as exc:
                logger.warning("Processor %s failed: %s", processor.name, exc)
                metrics[f"{processor.name}.error"] = str(exc)
        if step is not None:
            metrics["step"] = step
        if epoch is not None:
            metrics["epoch"] = epoch
        if self.sink:
            self.sink(metrics)
        return metrics

    @classmethod
    def from_pipeline(cls, pipeline, **kwargs) -> "EmbeddingProbe":
        """Build a probe reusing the processors already configured on a pipeline."""
        return cls(processors=list(pipeline.processors), **kwargs)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _to_numpy(emb: Any) -> np.ndarray:
        # Avoid an unconditional torch import — works fine even if torch absent.
        try:
            import torch  # type: ignore

            if isinstance(emb, torch.Tensor):
                return emb.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(emb)

    def _flatten_into(self, dest: Dict[str, Any], name: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        if arr.ndim == 0 or arr.shape == (1,):
            dest[name] = float(arr.flat[0])
            return
        if arr.ndim == 1:
            if self.scalar_only:
                dest[f"{name}.norm"] = float(np.linalg.norm(arr))
                dest[f"{name}.shape"] = list(arr.shape)
                return
            for i, v in enumerate(arr):
                dest[f"{name}.{i}"] = float(v)
            return
        # 2-D or higher: store summary stats only.
        dest[f"{name}.fro"] = float(np.linalg.norm(arr))
        dest[f"{name}.shape"] = list(arr.shape)


def embedding_probe(
    processors: Iterable[BaseProcessor],
    **probe_kwargs,
) -> Callable:
    """Decorator that runs a probe on the wrapped function's return value.

    The wrapped function should return embeddings (numpy array or torch tensor).
    The decorator returns a tuple ``(embeddings, metrics)``.

    Example:

        >>> @embedding_probe(processors=[NESumProcessor(), ConditionNumberProcessor()])
        ... def encode(model, batch):
        ...     return model(batch)
        >>>
        >>> emb, metrics = encode(model, batch)
    """
    probe = EmbeddingProbe(processors=processors, **probe_kwargs)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            emb = fn(*args, **kwargs)
            metrics = probe(emb)
            return emb, metrics

        return wrapper

    return decorator
