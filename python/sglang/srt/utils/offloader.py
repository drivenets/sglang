"""Stub — upstream module not cherry-picked.

Extended so a4af99d80's make_layers + weight loaders work against
the everything-warp32 image. All methods are identity pass-throughs.
"""


class _NullOffloader:
    def post_init(self):
        pass

    def update_param(self, old, new):
        return new

    def wrap_modules(self, module_iter, **kwargs):
        """Identity wrap — returns modules as a list."""
        return list(module_iter)


_INSTANCE = _NullOffloader()


def create_offloader_from_server_args(*args, **kwargs):
    return _INSTANCE


def get_offloader():
    return _INSTANCE


def set_offloader(o):
    pass


def update_param(old, new):
    return new
