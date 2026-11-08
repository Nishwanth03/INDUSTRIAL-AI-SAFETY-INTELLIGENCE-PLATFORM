import unittest
from datetime import datetime, timedelta, timezone

from ppe_detection import (
    ISSUE_LABELS,
    PPEAlertValidator,
    PPEDetection,
    issue_for_class,
)


class PPEClassMappingTests(unittest.TestCase):
    def test_missing_ppe_labels_map_to_guidance_categories(self):
        self.assertEqual(issue_for_class("No-Helmet"), "helmet_missing")
        self.assertEqual(issue_for_class("NO-Safety Vest"), "vest_missing")
        self.assertEqual(issue_for_class("No-Boots"), "shoes_missing")

    def test_unrelated_label_is_not_a_violation(self):
        self.assertIsNone(issue_for_class("Worker"))
        self.assertEqual(len(ISSUE_LABELS), 3)


class PPEAlertValidatorTests(unittest.TestCase):
    def test_requires_consecutive_frames(self):
        validator = PPEAlertValidator(consecutive_frames=3)
        violation = PPEDetection("No-Helmet", 0.9, (1, 2, 3, 4))

        self.assertEqual(validator.update([violation], "camera-1"), [])
        self.assertEqual(validator.update([violation], "camera-1"), [])
        alerts = validator.update([violation], "camera-1")

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].event_type, "helmet_missing")
        self.assertEqual(alerts[0].source_id, "camera-1")

    def test_suppresses_repeats_until_violation_clears(self):
        validator = PPEAlertValidator(
            consecutive_frames=1,
            clear_frames=2,
            cooldown_seconds=0,
        )
        violation = PPEDetection("No-Vest", 0.8, (1, 2, 3, 4))

        self.assertEqual(len(validator.update([violation], "camera-1")), 1)
        self.assertEqual(len(validator.update([violation], "camera-1")), 0)
        validator.update([], "camera-1")
        validator.update([], "camera-1")
        self.assertEqual(len(validator.update([violation], "camera-1")), 1)

    def test_cooldown_suppresses_a_recent_repeat(self):
        validator = PPEAlertValidator(
            consecutive_frames=1,
            clear_frames=1,
            cooldown_seconds=60,
        )
        violation = PPEDetection("No-Boots", 0.75, (1, 2, 3, 4))
        start = datetime.now(timezone.utc)

        self.assertEqual(len(validator.update([violation], "camera-1", start)), 1)
        validator.update([], "camera-1", start + timedelta(seconds=1))
        self.assertEqual(
            validator.update([violation], "camera-1", start + timedelta(seconds=30)),
            [],
        )
        validator.update([], "camera-1", start + timedelta(seconds=31))
        self.assertEqual(
            len(validator.update([violation], "camera-1", start + timedelta(seconds=61))),
            1,
        )


if __name__ == "__main__":
    unittest.main()
