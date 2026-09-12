"""Four slots retain their GPU until the complete task callback returns."""

import threading
import time

from scripts.server.run_four_slot_queue import run_slots


def test_two_slots_per_gpu_and_refill():
    lock = threading.Lock()
    active = {"a": 0, "b": 0}
    maximum = {"a": 0, "b": 0}
    seen = []

    def execute(row, gpu, slot):
        with lock:
            active[gpu] += 1
            maximum[gpu] = max(maximum[gpu], active[gpu])
            seen.append(row)
        time.sleep(0.01)
        with lock:
            active[gpu] -= 1
        return {"rc": 1 if row == "2" else 0}

    rows = [str(i) for i in range(9)]
    results = run_slots(rows, ["a", "b"], execute)
    assert sorted(seen) == sorted(rows)
    assert maximum == {"a": 2, "b": 2}
    assert len(results) == 9
