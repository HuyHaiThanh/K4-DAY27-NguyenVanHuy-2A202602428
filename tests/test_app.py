"""Streamlit interaction tests with real graph/checkpoints and mocked APIs."""

import asyncio
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rich.console import Console
from streamlit.testing.v1 import AppTest

import app as ui
from common.db import replay_events
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_4_audit as exercise


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitReviewTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {
            "HITL_DB_PATH": str(Path(directory) / "ui.db"), "GITHUB_USER": "ui-reviewer",
        }))
        self.enterContext(patch.object(exercise, "console", Console(file=StringIO())))
        self.pr_url = "https://github.com/example/demo/pull/1"
        pr = SimpleNamespace(title="Add priority", diff="+priority = 0", files_changed=["tasks.py"], head_sha="123456789")
        self.fetch = self.enterContext(patch.object(exercise, "fetch_pr", return_value=pr))
        self.analysis = PRAnalysis(
            summary="Adds priority", confidence=0.65, confidence_reasoning="Migration needs confirmation",
            comments=[ReviewComment(file="tasks.py", line=1, severity="issue", body="Confirm migration")],
        )
        get_llm = self.enterContext(patch.object(exercise, "get_llm"))
        self.llm = AsyncMock(return_value=self.analysis)
        get_llm.return_value.with_structured_output.return_value.ainvoke = self.llm
        self.post = self.enterContext(patch.object(exercise, "post_review_comment"))

    def new_app(self, thread=None):
        at = AppTest.from_file(str(APP_FILE), default_timeout=20)
        if thread:
            at.query_params["thread"] = thread
        at.run()
        self.assertFalse(at.exception)
        return at

    def click(self, at, label):
        next(button for button in at.button if button.label == label).click().run()
        self.assertFalse(at.exception)
        return at

    def start(self):
        at = self.new_app()
        at.text_input(key="pr_url_input").set_value(self.pr_url)
        return self.click(at, "Run review")

    def events(self, thread):
        return asyncio.run(replay_events(thread))

    def test_approve_posts_once_and_completed_page_reload_only_reads_checkpoint(self):
        at = self.start()
        thread = at.session_state["thread_id"]
        self.assertIn("Approval requested", at.subheader[0].value)
        self.assertEqual(at.code[0].value, "+priority = 0")
        self.post.assert_not_called()
        self.click(at, "Approve")
        self.assertEqual(at.session_state["thread_id"], thread)
        self.assertIn("committed", at.success[0].value)
        self.assertEqual(at.metric[0].value, "65%")
        self.assertTrue(at.get("link_button"))
        self.assertEqual(at.get("link_button")[0].proto.url, self.pr_url)
        self.post.assert_called_once()
        self.assertEqual(self.events(thread)[-1]["decision"], "approve")
        self.assertEqual(self.events(thread)[-2]["reviewer_id"], "ui-reviewer")
        at.run()
        restored = self.new_app(thread)
        self.assertTrue(restored.success)
        self.assertEqual(restored.text_input(key="pr_url_input").value, self.pr_url)
        self.post.assert_called_once()
        self.fetch.assert_called_once()
        self.llm.assert_awaited_once()

    def test_reject_and_edit_keep_feedback_without_posting(self):
        for choice in ("Reject", "Edit"):
            with self.subTest(choice=choice):
                at = self.start()
                thread = at.session_state["thread_id"]
                at.text_input(key=f"approval_feedback_{thread}").set_value("Need migration tests")
                self.click(at, choice)
                self.assertEqual(at.session_state["final"]["final_action"], "rejected")
                self.assertTrue(at.warning)
                self.assertFalse(at.success)
                event = self.events(thread)[-2]
                self.assertEqual(event["decision"], choice.lower())
                self.assertEqual(event["reason"], "Need migration tests")
                self.post.assert_not_called()

    def test_escalation_requires_answers_then_synthesizes_and_audits(self):
        self.analysis.confidence = 0.5
        self.analysis.risk_factors = ["Weak hashing", "Missing tests"]
        questions = ["auth.py: why MD5?", "storage.py: migration plan?"]
        self.analysis.escalation_questions = questions
        refined = PRAnalysis(summary="Refined review", confidence=0.6, confidence_reasoning="More context")
        self.llm.side_effect = [self.analysis, refined]
        at = self.start()
        thread = at.session_state["thread_id"]
        self.assertIn("Strong escalation", at.subheader[0].value)
        self.assertIn("Weak hashing", at.error[0].value)
        self.click(at, "Submit answers")
        self.assertTrue(at.warning)
        self.post.assert_not_called()
        self.llm.assert_awaited_once()
        for index, question in enumerate(questions):
            at.text_input(key=f"answer_{thread}_{index}").set_value(f"Answer for {question}")
        self.click(at, "Submit answers")
        self.assertTrue(at.success)
        self.assertEqual(at.metric[0].value, "60%")
        self.assertEqual(at.session_state["final"]["analysis"], refined)
        self.post.assert_called_once()
        self.assertIn("Reviewer answered escalation questions", self.post.call_args.args[1])
        self.assertEqual(len(self.events(thread)), 8)
        rows = asyncio.run(ui.recent_sessions())
        self.assertEqual(rows[0]["worst_risk"], "high")
        self.assertEqual(rows[0]["thread_id"], thread)

    def test_auto_success_and_failure_are_distinguished(self):
        self.analysis.confidence = 0.8
        at = self.start()
        self.assertTrue(at.success)
        self.assertEqual(at.metric[0].value, "80%")
        self.assertTrue(at.get("link_button"))
        self.post.side_effect = RuntimeError("GitHub unavailable")
        failed = self.start()
        self.assertFalse(failed.success)
        self.assertFalse(failed.get("link_button"))
        self.assertIn("Posting the comment failed", failed.error[0].value)
        self.assertEqual(failed.session_state["final"]["final_action"], "auto_commit_failed")

    def test_pending_session_survives_a_new_browser_session(self):
        original = self.start()
        thread = original.session_state["thread_id"]
        restored = self.new_app(thread)
        self.assertIn("Approval requested", restored.subheader[0].value)
        self.assertEqual(restored.session_state["thread_id"], thread)
        self.fetch.assert_called_once()
        self.llm.assert_awaited_once()
        self.click(restored, "Reject")
        self.assertEqual(self.events(thread)[-1]["decision"], "reject")
        self.post.assert_not_called()

    def test_sidebar_selection_restores_the_chosen_session(self):
        original = self.start()
        thread = original.session_state["thread_id"]
        original_dataframe = ui.st.dataframe
        selected = False

        def dataframe_with_one_selection(*args, **kwargs):
            nonlocal selected
            event = original_dataframe(*args, **kwargs)
            if not selected:
                selected = True
                return SimpleNamespace(selection=SimpleNamespace(rows=[0]))
            return event

        with patch.object(ui.st, "dataframe", side_effect=dataframe_with_one_selection):
            restored = self.new_app()
        self.assertEqual(restored.session_state["thread_id"], thread)
        self.assertEqual(restored.query_params["thread"], thread)
        self.assertIn("Approval requested", restored.subheader[0].value)
        self.fetch.assert_called_once()
        self.post.assert_not_called()

    def test_reload_can_continue_after_llm_failure_without_fetching_again(self):
        self.llm.side_effect = [RuntimeError("Temporary LLM error"), self.analysis]
        at = self.start()
        self.assertTrue(at.error)
        thread = at.session_state["thread_id"]
        restored = self.new_app(thread)
        self.assertTrue(restored.session_state["can_continue"])
        self.click(restored, "Continue review")
        self.assertIn("Approval requested", restored.subheader[0].value)
        self.fetch.assert_called_once()
        self.assertEqual(self.llm.await_count, 2)
        self.post.assert_not_called()

    def test_invalid_url_and_unknown_thread_do_not_start_a_review(self):
        at = self.new_app()
        at.text_input(key="pr_url_input").set_value("https://example.com/demo/pull/1")
        self.click(at, "Run review")
        self.assertTrue(at.error)
        restored = self.new_app("missing-thread")
        self.assertTrue(restored.error)
        self.fetch.assert_not_called()
        self.post.assert_not_called()

    def test_backend_rejects_unknown_resumes_and_cross_pr_sessions(self):
        with self.assertRaisesRegex(ValueError, "without a checkpoint"):
            asyncio.run(ui.run_graph(self.pr_url, "missing", {"choice": "approve"}))
        at = self.start()
        with self.assertRaisesRegex(ValueError, "different PR"):
            asyncio.run(ui.run_graph("https://github.com/example/demo/pull/2", at.session_state["thread_id"]))
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
