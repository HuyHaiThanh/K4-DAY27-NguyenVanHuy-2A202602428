"""Offline checks for approval interrupts, resumes, and comment side effects."""

import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.types import Command
from rich.console import Console

from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_2_hitl as exercise


class HumanApprovalTests(unittest.TestCase):
    def setUp(self):
        self.pr_url = "https://github.com/example/demo/pull/1"
        self.pr = SimpleNamespace(
            title="Add priority",
            diff="+priority = 0\n" * 200,
            files_changed=["tasks.py"],
            head_sha="1234567890abcdef",
        )
        self.analysis = PRAnalysis(
            summary="Adds task priority",
            confidence=0.65,
            confidence_reasoning="Migration needs human confirmation",
            comments=[ReviewComment(
                file="tasks.py", line=1, severity="issue",
                body="How are existing tasks migrated?",
            )],
        )
        self.output = StringIO()
        self.enterContext(patch.object(exercise, "console", Console(file=self.output)))
        self.fetch = self.enterContext(patch.object(exercise, "fetch_pr", return_value=self.pr))
        get_llm = self.enterContext(patch.object(exercise, "get_llm"))
        self.llm = get_llm.return_value.with_structured_output.return_value
        self.llm.invoke.return_value = self.analysis
        self.post = self.enterContext(patch.object(exercise, "post_review_comment"))

    def test_pause_then_resume_each_choice_without_repeating_external_calls(self):
        for choice in ("approve", "reject", "edit"):
            with self.subTest(choice=choice):
                self.fetch.reset_mock()
                self.llm.invoke.reset_mock()
                self.post.reset_mock()
                app = exercise.build_graph()
                cfg = {"configurable": {"thread_id": choice}}
                result = app.invoke({"pr_url": self.pr_url, "thread_id": choice}, cfg)

                self.post.assert_not_called()
                payload = result["__interrupt__"][0].value
                self.assertEqual(payload, {
                    "kind": "approval_request",
                    "confidence": self.analysis.confidence,
                    "confidence_reasoning": self.analysis.confidence_reasoning,
                    "summary": self.analysis.summary,
                    "comments": [c.model_dump() for c in self.analysis.comments],
                    "diff_preview": self.pr.diff[:2000],
                })
                self.assertEqual(app.get_state(cfg).next, ("human_approval",))

                answer = {"choice": choice, "feedback": "Migration checked"}
                final = app.invoke(Command(resume=answer), cfg)

                self.assertNotIn("__interrupt__", final)
                self.assertEqual(app.get_state(cfg).next, ())
                self.assertEqual(final["human_choice"], choice)
                self.assertEqual(final["human_feedback"], answer["feedback"])
                self.fetch.assert_called_once_with(self.pr_url)
                self.llm.invoke.assert_called_once()
                if choice == "approve":
                    self.assertEqual(final["final_action"], "committed")
                    self.post.assert_called_once_with(
                        self.pr_url, exercise._render_comment_body(final),
                    )
                    self.assertIn("### Automated review (confidence 65%)", self.post.call_args.args[1])
                else:
                    self.assertEqual(final["final_action"], "rejected")
                    self.post.assert_not_called()

    def test_post_failure_is_not_reported_as_committed(self):
        self.post.side_effect = RuntimeError("GitHub unavailable")
        app = exercise.build_graph()
        cfg = {"configurable": {"thread_id": "post-failure"}}
        app.invoke({"pr_url": self.pr_url}, cfg)
        final = app.invoke(Command(resume={"choice": "approve"}), cfg)
        self.assertEqual(final["final_action"], "commit_failed")
        self.assertIsNone(final["human_feedback"])
        self.post.assert_called_once()

    def test_prompt_preserves_review_text_and_validates_choice(self):
        payload = {
            "confidence": 0.65,
            "confidence_reasoning": "Check [bold] markers",
            "summary": "Review [link] text",
            "comments": [{"severity": "issue", "file": "tasks.py", "line": 1, "body": "Keep [red] literal"}],
            "diff_preview": "+items[index] = [bold]",
        }
        with patch.object(exercise.console, "input", side_effect=["invalid", " EDIT ", " Needs migration "]):
            answer = exercise.prompt_human(payload)
        self.assertEqual(answer, {"choice": "edit", "feedback": "Needs migration"})
        output = self.output.getvalue()
        for expected in (
            "Approval request", "Check [bold] markers", "Review [link] text",
            "[issue] tasks.py:1", "Keep [red] literal", "+items[index] = [bold]",
        ):
            self.assertIn(expected, output)

    def test_main_handles_multiple_interrupts_with_the_same_config(self):
        payloads = [{"kind": "approval_request", "summary": str(i)} for i in range(2)]
        answers = [{"choice": "edit", "feedback": "Check migration"}, {"choice": "reject"}]
        app = Mock()
        app.invoke.side_effect = [
            {"__interrupt__": [SimpleNamespace(value=payloads[0])]},
            {"__interrupt__": [SimpleNamespace(value=payloads[1])]},
            {"final_action": "rejected"},
        ]
        with (
            patch.object(exercise, "build_graph", return_value=app),
            patch.object(exercise, "load_dotenv"),
            patch.object(exercise, "prompt_human", side_effect=answers) as prompt,
            patch("sys.argv", ["exercise_2_hitl.py", "--pr", self.pr_url]),
        ):
            exercise.main()
        self.assertEqual(prompt.call_count, 2)
        self.assertEqual([call.args[0] for call in prompt.call_args_list], payloads)
        calls = app.invoke.call_args_list
        self.assertEqual(len(calls), 3)
        cfg = calls[0].args[1]
        self.assertEqual(calls[0].args[0]["thread_id"], cfg["configurable"]["thread_id"])
        for call, answer in zip(calls[1:], answers):
            self.assertIsInstance(call.args[0], Command)
            self.assertEqual(call.args[0].resume, answer)
            self.assertIs(call.args[1], cfg)
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
