import threading
from collections import deque
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()

    def drain(self, max_items: int) -> list:
        """Block until at least one item is available, then drain up to
        ``max_items`` items without further blocking. Used by the coalescing
        transfer worker to amortize per-call overhead across multiple chunks
        when they queue up faster than the worker can drain them.
        """
        with self._cond:
            while not self._buf:
                self._cond.wait()
            items = []
            while self._buf and len(items) < max_items:
                items.append(self._buf.popleft())
            return items


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
