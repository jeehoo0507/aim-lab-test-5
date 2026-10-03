"""imagecorruptions 1.1.2 compatibility with current scikit-image, plus determinism.

1. imagecorruptions calls skimage.filters.gaussian(..., multichannel=True) (glass_blur), an argument
   removed in scikit-image 0.19+. Translate it to channel_axis=-1 (documented equivalent).
2. impulse_noise uses skimage.util.random_noise, which draws from its own unseeded generator.
   Feed it a seed from the global numpy RNG so np.random.seed(...) makes every corruption reproducible.
3. plasma_fractal (fog) uses np.float_, removed in NumPy 2.0. imagecorruptions gets a numpy proxy
   where float_ = float64 (what np.float_ always was); global numpy is untouched.
"""
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="imagecorruptions")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import types  # noqa: E402

import numpy as _np  # noqa: E402
import imagecorruptions.corruptions as _ic  # noqa: E402


class _NumpyProxy(types.ModuleType):
    float_ = _np.float64

    def __getattr__(self, name):
        return getattr(_np, name)


_ic.np = _NumpyProxy("numpy")
from skimage.filters import gaussian as _gaussian  # noqa: E402


def _gaussian_compat(image, *args, multichannel=None, **kwargs):
    if multichannel is not None and "channel_axis" not in kwargs:
        kwargs["channel_axis"] = -1 if multichannel else None
    return _gaussian(image, *args, **kwargs)


_ic.gaussian = _gaussian_compat

_random_noise = _ic.sk.util.random_noise


def _random_noise_seeded(image, *args, rng=None, **kwargs):
    if rng is None:
        rng = int(_ic.np.random.randint(0, 2**31 - 1))
    return _random_noise(image, *args, rng=rng, **kwargs)


_ic.sk.util.random_noise = _random_noise_seeded

from imagecorruptions import corrupt, get_corruption_names  # noqa: E402,F401

CORRUPTIONS = tuple(get_corruption_names("common"))
assert len(CORRUPTIONS) == 15, CORRUPTIONS
