import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TORCH_AVAILABLE:
    from common_utils.ptutils.serialization import legacy_torch_load
else:
    def legacy_torch_load(): pass  # Dummy


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available or not installable in current environment")


# Test legacy_torch_load context manager
def test_legacy_torch_load_injects_weights_only():
    original_load = torch.load
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        captured['args'] = args
        return "loaded"

    torch.load = fake_load
    try:
        with legacy_torch_load():
            # torch.load should be patched inside the context
            assert torch.load is not fake_load
            result = torch.load("some_path", map_location="cpu")
            assert result == "loaded"
            # weights_only should be injected as False
            assert captured['weights_only'] is False
            # Other args/kwargs should be passed through
            assert captured['args'] == ("some_path",)
            assert captured['map_location'] == "cpu"
        # torch.load should be restored to our fake (the value at context entry)
        assert torch.load is fake_load
    finally:
        torch.load = original_load


def test_legacy_torch_load_respects_explicit_weights_only():
    original_load = torch.load
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return "loaded"

    torch.load = fake_load
    try:
        with legacy_torch_load():
            torch.load("some_path", weights_only=True)
            # Explicit weights_only should be preserved, not overridden
            assert captured['weights_only'] is True
    finally:
        torch.load = original_load


def test_legacy_torch_load_restores_on_exception():
    original_load = torch.load
    try:
        with pytest.raises(RuntimeError):
            with legacy_torch_load():
                # torch.load is patched here
                assert torch.load is not original_load
                raise RuntimeError("boom")
        # torch.load must be restored even when an exception propagates
        assert torch.load is original_load
    finally:
        torch.load = original_load


def test_legacy_torch_load_roundtrip(tmp_path):
    # End-to-end: saving an object with a pickled payload and loading it back
    # within the context should succeed even on PyTorch >= 2.6.
    payload = {"config": {"lr": 1e-3, "name": "exp"}, "tensor": torch.tensor([1.0, 2.0])}
    path = tmp_path / "ckpt.pth"
    torch.save(payload, path)

    with legacy_torch_load():
        loaded = torch.load(path, map_location="cpu")

    assert loaded["config"] == payload["config"]
    assert torch.equal(loaded["tensor"], payload["tensor"])
