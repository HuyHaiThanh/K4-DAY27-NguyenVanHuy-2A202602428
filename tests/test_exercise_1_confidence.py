"""Offline regression checks for Exercise 1's structured analysis and routing."""

import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from common.schemas import PRAnalysis
from exercises import exercise_1_confidence as exercise


class ConfidenceGraphTests(unittest.TestCase):
    def test_graph_routes_all_buckets_and_boundaries(self):
        pr_url = "https://github.com/example/demo/pull/1"
        pr = SimpleNamespace(
            title="Add priority field",
            diff="+priority: int = 0",
            files_changed=["tasks.py"],
            head_sha="1234567890abcdef",
        )
        cases = [
            (0.0, "escalate", "pending_escalation"),
            (0.5799, "escalate", "pending_escalation"),
            (0.58, "human_approval", "pending_human_approval"),
            (0.65, "human_approval", "pending_human_approval"),
            (0.7299, "human_approval", "pending_human_approval"),
            (0.73, "auto_approve", "auto_approved"),
            (1.0, "auto_approve", "auto_approved"),
        ]
        for confidence, decision, action in cases:
            with self.subTest(confidence=confidence):
                analysis = PRAnalysis(
                    summary="Review of priority field",
                    confidence=confidence,
                    confidence_reasoning="Controlled score for routing verification",
                )
                output = StringIO()
                with (
                    patch.object(exercise, "fetch_pr", return_value=pr) as fetch,
                    patch.object(exercise, "get_llm") as get_llm,
                    patch.object(exercise, "console", Console(file=output)),
                ):
                    llm = get_llm.return_value.with_structured_output.return_value
                    llm.invoke.return_value = analysis
                    final = exercise.build_graph().invoke({"pr_url": pr_url})

                fetch.assert_called_once_with(pr_url)
                get_llm.return_value.with_structured_output.assert_called_once_with(PRAnalysis)
                llm.invoke.assert_called_once()
                messages = llm.invoke.call_args.args[0]
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertIn(pr.title, messages[1]["content"])
                self.assertIn(pr.diff, messages[1]["content"])
                self.assertEqual(final["analysis"], analysis)
                self.assertEqual(final["decision"], decision)
                self.assertEqual(final["final_action"], action)
                self.assertIn(decision.replace("_", " ").upper(), output.getvalue())


if __name__ == "__main__":
    unittest.main()
