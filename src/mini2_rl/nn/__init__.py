"""Neural network architectures for V2 / V3. Torch is optional at import-time."""
try:
    from .architectures import MLP, ConvQNet
    __all__ = ["MLP", "ConvQNet"]
except ImportError:  # pragma: no cover -- torch not installed
    __all__ = []
