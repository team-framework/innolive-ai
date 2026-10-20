"""Exclude the warmed process baseline from cyclic GC while serving video."""

import gc


class StartupGc:
    """Keep automatic GC for requests; release our baseline again on shutdown.

    Use only before admitting streams. Freezing a live process could retain
    session cycles. Reference-counted cache entries can still be evicted normally.
    Add warmed models even when Python has already frozen startup objects.
    An existing owner's frozen generation must not be unfrozen by this owner.
    """

    def __init__(self):
        self.owned = False
        self.applied = False

    def freeze(self):
        if self.applied:
            return
        self.owned = gc.get_freeze_count() == 0
        gc.collect()
        gc.freeze()
        self.applied = True

    def close(self):
        if self.owned:
            gc.unfreeze()
            self.owned = False
            self.applied = False
