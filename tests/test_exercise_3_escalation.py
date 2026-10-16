"""Offline checks for contextual escalation and refined-review posting."""

import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command
from rich.console import Console

from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_3_escalation as exercise


class EscalationTests(unittest.TestCase):
    def setUp(self):
        self.pr_url = "https://github.com/example/demo/pull/2"
        self.pr = SimpleNamespace(
            title="Add login and sync",
            diff="+digest = md5(password)\n+query = f'INSERT {title}'",
            files_changed=["auth.py", "storage.py"],
            head_sha="abcdef1234567890",
        )
        self.questions = [
            "auth.py: why use MD5 for passwords?",
            "storage.py Storage.add: is title untrusted input?",
        ]
        self.initial = PRAnalysis(
            summary="Adds login and cloud sync",
            risk_factors=["Weak password hashing", "SQL injection"],
            comments=[ReviewComment(
                file="auth.py", line=1, severity="blocker", body="Replace MD5",
            )],
            confidence=0.5,
            confidence_reasoning="Missing security context and tests",
            escalation_questions=self.questions,
        )
        self.refined = PRAnalysis(
            summary="Security problems remain after clarification",
            confidence=0.45,
            confidence_reasoning="Answers reveal additional unknowns",
            comments=[ReviewComment(
                file="storage.py", line=2, severity="blocker", body="Use parameterized SQL",
            )],
        )
        self.output = StringIO()
        self.enterContext(patch.object(exercise, "console", Console(file=self.output)))
        self.fetch = self.enterContext(patch.object(exercise, "fetch_pr", return_value=self.pr))
        self.get_llm = self.enterContext(patch.object(exercise, "get_llm"))
        self.llm = self.get_llm.return_value.with_structured_output.return_value
        self.llm.invoke.side_effect = [self.initial, self.refined]
        self.post = self.enterContext(patch.object(exercise, "post_review_comment"))
        self.cfg = {"configurable": {"thread_id": "escalation-test"}}

    def test_escalation_resumes_with_full_context_and_posts_refined_review_once(self):
        app = exercise.build_graph()
        result = app.invoke({"pr_url": self.pr_url}, self.cfg)
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload, {
            "kind": "escalation",
            "pr_url": self.pr_url,
            "confidence": self.initial.confidence,
            "confidence_reasoning": self.initial.confidence_reasoning,
            "summary": self.initial.summary,
            "risk_factors": self.initial.risk_factors,
            "questions": self.questions,
        })
        self.assertEqual(app.get_state(self.cfg).next, ("escalate",))
        self.post.assert_not_called()
        self.assertEqual(self.llm.invoke.call_count, 1)
        answers = {
            self.questions[0]: "Prototype only; production hashing is undecided.",
            self.questions[1]: "Yes, title is user supplied.",
        }
        final = app.invoke(Command(resume=answers), self.cfg)

        self.assertNotIn("__interrupt__", final)
        self.assertEqual(app.get_state(self.cfg).next, ())
        self.assertEqual(final["escalation_answers"], answers)
        self.assertEqual(final["analysis"], self.refined)
        self.assertEqual(final["final_action"], "committed_after_escalation")
        self.fetch.assert_called_once_with(self.pr_url)
        self.assertEqual(self.llm.invoke.call_count, 2)
        self.assertEqual(self.get_llm.return_value.with_structured_output.call_count, 2)
        for call in self.get_llm.return_value.with_structured_output.call_args_list:
            self.assertEqual(call.args, (PRAnalysis,))
        prompt = self.llm.invoke.call_args.args[0][1]["content"]
        for context in (self.pr.title, self.pr.diff, self.initial.model_dump_json(indent=2)):
            self.assertIn(context, prompt)
        for question, answer in answers.items():
            self.assertIn(f"Q: {question}\nA: {answer}", prompt)
        self.post.assert_called_once_with(self.pr_url, exercise._render_comment_body(final))
        body = self.post.call_args.args[1]
        self.assertIn(self.refined.summary, body)
        self.assertIn("Use parameterized SQL", body)
        self.assertIn("Reviewer answered escalation questions", body)
        for question, answer in answers.items():
            self.assertIn(question, body)
            self.assertIn(answer, body)

    def test_failed_escalation_post_is_not_reported_as_committed(self):
        self.post.side_effect = RuntimeError("GitHub unavailable")
        app = exercise.build_graph()
        app.invoke({"pr_url": self.pr_url}, self.cfg)
        answers = {q: "Still unknown" for q in self.questions}
        final = app.invoke(Command(resume=answers), self.cfg)
        self.assertEqual(final["final_action"], "commit_failed")
        self.post.assert_called_once()

    def test_empty_questions_use_fallback_without_posting_or_mutating_analysis(self):
        self.initial.escalation_questions = []
        app = exercise.build_graph()
        result = app.invoke({"pr_url": self.pr_url}, self.cfg)
        questions = result["__interrupt__"][0].value["questions"]
        self.assertEqual(questions, ["What is the intent of this PR?", "Any migration concerns?"])
        self.assertEqual(self.initial.escalation_questions, [])
        self.post.assert_not_called()

    def test_existing_auto_and_human_branches_do_not_synthesize(self):
        for confidence, choice, action in (
            (0.8, None, "auto_approved"),
            (0.65, "approve", "committed"),
            (0.65, "reject", "rejected"),
            (0.65, "edit", "rejected"),
        ):
            with self.subTest(confidence=confidence, choice=choice):
                self.post.reset_mock()
                self.llm.invoke.reset_mock()
                self.llm.invoke.side_effect = None
                self.llm.invoke.return_value = self.initial.model_copy(update={"confidence": confidence})
                app = exercise.build_graph()
                result = app.invoke({"pr_url": self.pr_url}, self.cfg)
                if choice is not None:
                    self.assertEqual(result["__interrupt__"][0].value["kind"], "approval_request")
                    self.post.assert_not_called()
                    result = app.invoke(Command(resume={"choice": choice}), self.cfg)
                self.assertEqual(result["final_action"], action)
                self.llm.invoke.assert_called_once()
                self.assertEqual(self.post.call_count, int(action in ("auto_approved", "committed")))

    def test_escalation_terminal_collects_each_answer_and_preserves_markup_text(self):
        payload = {
            "kind": "escalation", "confidence": 0.5,
            "summary": "Review [bold] literal",
            "questions": ["auth.py: token [index]?", "storage.py: migration?"],
        }
        with patch("builtins.input", side_effect=[" First answer ", " Second answer "]):
            answers = exercise.handle_interrupt(payload)
        self.assertEqual(answers, dict(zip(payload["questions"], ["First answer", "Second answer"])))
        output = self.output.getvalue()
        for expected in ("Escalation conf=50%", payload["summary"], *payload["questions"]):
            self.assertIn(expected, output)


if __name__ == "__main__":
    unittest.main()
