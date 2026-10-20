"""Exclude the warmed process baseline from cyclic GC while serving video."""

import gc


class StartupGc:
    """Keep automatic GC for requests; release our baseline again on shutdown.

    Use only before admitting streams. Freezing a live process could retain
    session cycles. Reference-counted cache entries can still be evicted normally.
    An existing owner's frozen generation must not be unfrozen by this owner.
    """

    def __init__(self):
        self.owned = False

    def freeze(self):
        if self.owned or gc.get_freeze_count():
            return
        gc.collect()
        gc.freeze()
        self.owned = True

    def close(self):
        if self.owned:
            gc.unfreeze()
            self.owned = False
