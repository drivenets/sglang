"""Build script for the fast decode C++ extension."""
import os
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))

_fast_decode_ext = load(
    name="_fast_decode_ext",
    sources=[os.path.join(_dir, "fast_decode_ext.cpp")],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)

# Re-export
cache_request = _fast_decode_ext.cache_request
uncache_request = _fast_decode_ext.uncache_request
clear_cache = _fast_decode_ext.clear_cache
set_finish_classes = _fast_decode_ext.set_finish_classes
fast_decode_step = _fast_decode_ext.fast_decode_step
