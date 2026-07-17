import unittest

from sglang.srt.utils.device_timer import DeviceTimer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, state):
        self.state = state

    def query(self):
        return self.state["ready"] and self.state["query_works"]

    def synchronize(self):
        self.state["ready"] = True


class _FakeInterval:
    def __init__(self, state, elapsed_ms, metadata):
        self.end_event = _FakeEvent(state)
        self._elapsed_ms = elapsed_ms
        self.metadata = metadata

    def elapsed_time(self):
        return self._elapsed_ms


class TestDeviceTimer(unittest.TestCase):
    def test_flush_waits_for_newest_event_and_reports_pending_intervals(self):
        reports = []
        timer = DeviceTimer(reporter=lambda **kwargs: reports.append(kwargs))
        # Mirrors Ascend events where synchronize() succeeds while query()
        # continues to report False.
        stream_state = {"ready": False, "query_works": False}
        timer._intervals.extend(
            [
                _FakeInterval(stream_state, 1000, {"mode": "first"}),
                _FakeInterval(stream_state, 2500, {"mode": "second"}),
            ]
        )

        timer._report()
        self.assertEqual(reports, [])

        self.assertEqual(timer.flush(), 2)
        self.assertEqual(
            reports,
            [
                {"t": 1.0, "mode": "first"},
                {"t": 2.5, "mode": "second"},
            ],
        )
        self.assertEqual(len(timer._intervals), 0)

        self.assertEqual(timer.flush(), 0)


if __name__ == "__main__":
    unittest.main()
