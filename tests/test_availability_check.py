from datetime import datetime
import unittest

from zoneinfo import ZoneInfo

from monitor.availability_check import AvailabilitySample, classify, should_notify


PRAGUE = ZoneInfo("Europe/Prague")


class TestAvailabilityCheck(unittest.TestCase):
    def test_classify_warn_threshold_defaults_to_92_percent(self) -> None:
        sample = AvailabilitySample(
            dispon_pripojeni=0.91,
            dispon_rizeni=1.0,
            month_start_utc=datetime(2026, 4, 1, tzinfo=PRAGUE),
            month_start_local=datetime(2026, 4, 1, tzinfo=PRAGUE),
        )

        self.assertEqual(classify(sample), "WARN")

    def test_should_notify_once_per_local_day_while_degraded(self) -> None:
        now = datetime(2026, 4, 30, 9, 0, tzinfo=PRAGUE)
        state = {"last_level": "WARN", "last_notified_date": "2026-04-29"}

        self.assertTrue(should_notify(state, "WARN", now))

    def test_should_not_repeat_degraded_notification_same_local_day(self) -> None:
        now = datetime(2026, 4, 30, 9, 0, tzinfo=PRAGUE)
        state = {"last_level": "WARN", "last_notified_date": "2026-04-30"}

        self.assertFalse(should_notify(state, "WARN", now))

    def test_should_notify_once_per_local_day_while_monitor_error(self) -> None:
        now = datetime(2026, 4, 30, 9, 0, tzinfo=PRAGUE)
        yesterday = {"last_level": "ERROR", "last_notified_date": "2026-04-29"}
        today = {"last_level": "ERROR", "last_notified_date": "2026-04-30"}

        self.assertTrue(should_notify(yesterday, "ERROR", now))
        self.assertFalse(should_notify(today, "ERROR", now))

    def test_should_not_repeat_same_event_after_level_flip_same_local_day(self) -> None:
        now = datetime(2026, 4, 30, 12, 0, tzinfo=PRAGUE)
        state = {
            "last_level": "OK",
            "last_notified_dates_by_level": {"WARN": "2026-04-30"},
        }

        self.assertFalse(should_notify(state, "WARN", now))
        self.assertTrue(should_notify(state, "CRIT", now))


if __name__ == "__main__":
    unittest.main()
