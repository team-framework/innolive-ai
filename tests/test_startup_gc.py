import subprocess
import sys
import unittest


class StartupGcTests(unittest.TestCase):
    def run_process(self, source):
        result = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, timeout=15
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_session_cycles_collect_and_baseline_unfreezes_on_close(self):
        self.run_process("""
import gc, weakref
from service.startup_gc import StartupGc
# Python 3.12 can move immortal objects into the permanent generation during collect.
gc.unfreeze()
gc.collect()
interpreter_baseline = gc.get_freeze_count()
gc.unfreeze()
class Value: pass
baseline = Value()
baseline.cycle = baseline
baseline_ref = weakref.ref(baseline)
guard = StartupGc()
guard.freeze()
assert gc.isenabled()
assert gc.get_freeze_count() > 0
del baseline
session = Value()
session.cycle = session
session_ref = weakref.ref(session)
del session
gc.collect()
assert session_ref() is None
assert baseline_ref() is not None
guard.close()
guard.close()
gc.collect()
assert baseline_ref() is None
assert gc.get_freeze_count() == interpreter_baseline
""")

    def test_reference_counted_cache_entries_release_while_frozen(self):
        self.run_process("""
import weakref
from service.startup_gc import StartupGc
class Value: pass
cache = {0: Value()}
old = weakref.ref(cache[0])
guard = StartupGc()
guard.freeze()
cache[0] = Value()
assert old() is None
guard.close()
""")

    def test_extends_an_existing_baseline_without_unfreezing_its_owner(self):
        self.run_process("""
import gc, weakref
from service.startup_gc import StartupGc
class Value: pass
original = Value()
original.cycle = original
original_ref = weakref.ref(original)
gc.freeze()
count = gc.get_freeze_count()
del original
warmed = Value()
warmed.cycle = warmed
warmed_ref = weakref.ref(warmed)
guard = StartupGc()
guard.freeze()
assert gc.get_freeze_count() > count
del warmed
session = Value()
session.cycle = session
session_ref = weakref.ref(session)
del session
guard.freeze()  # A repeated call must not freeze request/session objects.
gc.collect()
assert session_ref() is None
assert original_ref() is not None
assert warmed_ref() is not None
guard.close()
gc.collect()
assert original_ref() is not None
assert warmed_ref() is not None
gc.unfreeze()
gc.collect()
assert original_ref() is None
assert warmed_ref() is None
""")
