"""Stub — upstream module not cherry-picked.

Extends the per-bisect stub so a4af99d80's scheduler.py imports work.
"""
from contextlib import contextmanager


@contextmanager
def configure_subprocess(*args, **kwargs):
    yield


def get_numa_node_if_available(server_args, gpu_id):
    """Stub — returns None so scheduler skips numa-binding path."""
    return None


def numa_bind_to_node(node):
    """Shim — real impl lives in sglang.srt.utils.common.numa_bind_to_node."""
    from sglang.srt.utils.common import numa_bind_to_node as _bind
    return _bind(node)
