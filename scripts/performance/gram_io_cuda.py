"""Compatibility entry point for the now native, opt-in Gram IO implementation.

Historical frozen sources retain their own kernel. Current probes share the
native implementation so diagnostic and packaged execution cannot diverge.
"""

from gradpert.modeling.v2._weighted_gram_io_cuda import (
    _backward as _backward,
)
from gradpert.modeling.v2._weighted_gram_io_cuda import (
    backward as backward,
)
