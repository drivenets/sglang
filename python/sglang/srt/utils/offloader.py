"""Stub — upstream module not cherry-picked."""
class _NullOffloader:
    def post_init(self): pass
    def update_param(self, old, new): return new
_INSTANCE = _NullOffloader()
def create_offloader_from_server_args(*a, **kw): return _INSTANCE
def get_offloader(): return _INSTANCE
def set_offloader(o): pass
def update_param(old, new): return new
