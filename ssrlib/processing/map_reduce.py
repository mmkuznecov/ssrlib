"""Optional mixin for processors that can be computed incrementally.

Streaming computation is a niche feature — useful only when embeddings don't
fit in RAM. For typical workloads (a few hundred thousand vectors at
embedding-dim ~1k = ~GB-scale) the whole-array path is fine.

Processors that implement this mixin must support BOTH paths:
    - ``process(X)``         : standard whole-array processing (BaseProcessor)
    - ``partial_fit(batch)`` : accumulate state from a batch
    - ``finalize()``         : produce the final result and reset

The pipeline auto-detects ``MapReduceMixin`` and dispatches accordingly when
``streaming=True`` is set.
"""

from __future__ import annotations

import numpy as np


class MapReduceMixin:
    """Marker mixin for processors that support incremental computation.

    Concrete subclasses must override ``partial_fit``, ``finalize``, and
    ``reset``. Calling these in any other order than
    ``reset() -> partial_fit()+ -> finalize()`` is undefined.
    """

    def partial_fit(self, batch: np.ndarray) -> None:
        """Update internal state with a batch of embeddings.

        Args:
            batch: (B, D) array of embeddings.
        """
        raise NotImplementedError

    def finalize(self) -> np.ndarray:
        """Compute and return the final result from accumulated state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear accumulator state so the processor can be reused."""
        raise NotImplementedError
