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
assert gc.get_freeze_count() == 0
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

    def test_does_not_unfreeze_an_existing_owner(self):
        self.run_process("""
import gc
from service.startup_gc import StartupGc
gc.freeze()
count = gc.get_freeze_count()
guard = StartupGc()
guard.freeze()
guard.close()
assert gc.get_freeze_count() == count
gc.unfreeze()
""")
