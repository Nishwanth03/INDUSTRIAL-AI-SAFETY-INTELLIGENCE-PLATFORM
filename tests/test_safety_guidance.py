import unittest

from safety_guidance import WORKER_GUIDANCE, get_worker_guidance


class WorkerGuidanceTests(unittest.TestCase):
    def test_guidance_exists_for_each_supported_issue(self):
        for issue_type, message in WORKER_GUIDANCE.items():
            with self.subTest(issue_type=issue_type):
                self.assertEqual(get_worker_guidance(issue_type), message)
                self.assertTrue(message.strip())

    def test_unknown_issue_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Unknown safety issue type"):
            get_worker_guidance("unknown")


if __name__ == "__main__":
    unittest.main()
