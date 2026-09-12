from types import SimpleNamespace

import pytest

from scripts.performance.profile_native_a0 import ProfileGateError, _ordered_batch_identity


@pytest.mark.parametrize("rows,cached", [((), None), (("p",), None), (("p", "q"), "0" * 64)])
def test_invalid_actual_or_cached_identity_rejected(rows, cached):
    batch = SimpleNamespace(
        condition_ids=("A", "A"), perturbed_row_ids=rows, perturbed_row_ids_sha256=cached
    )
    with pytest.raises(ProfileGateError):
        _ordered_batch_identity(batch, 1)
