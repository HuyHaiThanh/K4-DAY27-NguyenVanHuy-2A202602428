"""Integration tests using real SQLite audit rows and durable checkpoints."""

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console

from audit import replay
from common.db import db_conn, db_path, replay_events
from common.schemas import PRAnalysis, risk_level_for
from exercises import exercise_4_audit as exercise


class AuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {
            "HITL_DB_PATH": str(Path(directory) / "audit.db"),
            "GITHUB_USER": "lab-reviewer",
        }))
        self.output = StringIO()
        self.enterContext(patch.object(exercise, "console", Console(file=self.output)))
        self.pr_url = "https://github.com/example/demo/pull/1"
        pr = SimpleNamespace(title="Add priority", diff="+priority = 0", files_changed=["tasks.py"], head_sha="123456789")
        self.fetch = self.enterContext(patch.object(exercise, "fetch_pr", return_value=pr))
        self.analysis = PRAnalysis(summary="Adds priority", confidence=0.65, confidence_reasoning="Check migration [plan]")
        get_llm = self.enterContext(patch.object(exercise, "get_llm"))
        self.llm = AsyncMock(return_value=self.analysis)
        get_llm.return_value.with_structured_output.return_value.ainvoke = self.llm
        self.post = self.enterContext(patch.object(exercise, "post_review_comment"))

    async def invoke(self, thread, value=None):
        # A fresh saver/graph on every invocation exercises persisted checkpoints.
        async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
            await cp.setup()
            app = exercise.build_graph(cp)
            cfg = {"configurable": {"thread_id": thread}}
            return await app.ainvoke(value if value is not None else {
                "pr_url": self.pr_url, "thread_id": thread,
            }, cfg)

    async def assert_valid_events(self, thread):
        events = await replay_events(thread)
        allowed = {"auto", "approve", "reject", "edit", "escalate", "pending"}
        for event in events:
            self.assertEqual(event["agent_id"], exercise.AGENT_ID)
            self.assertIn(event["decision"], allowed)
            self.assertGreaterEqual(event["execution_time_ms"], 0)
            self.assertTrue(event["timestamp"])
            self.assertTrue(event["reason"])
            expected_risk = "med" if event["action"] == "fetch_pr" else risk_level_for(event["confidence"])
            self.assertEqual(event["risk_level"], expected_risk)
            if event["action"] in {"human_approval", "escalate"} and event["decision"] != "pending":
                self.assertEqual(event["reviewer_id"], "lab-reviewer")
            else:
                self.assertIsNone(event["reviewer_id"])
        async with db_conn() as conn:
            rows = await (await conn.execute(
                "SELECT thread_id, pr_url FROM audit_events WHERE thread_id = ?", (thread,),
            )).fetchall()
            self.assertTrue(all(r["thread_id"] == thread and r["pr_url"] == self.pr_url for r in rows))
            unsafe = await (await conn.execute(
                "SELECT COUNT(*) FROM audit_events WHERE risk_level = 'high' AND decision = 'auto'",
            )).fetchone()
            self.assertEqual(unsafe[0], 0)
        return events

    async def test_human_choices_persist_and_log_pending_again_on_resume(self):
        for choice in ("approve", "reject", "edit"):
            with self.subTest(choice=choice):
                self.fetch.reset_mock()
                self.llm.reset_mock()
                self.post.reset_mock()
                result = await self.invoke(choice)
                self.assertEqual(result["__interrupt__"][0].value["kind"], "approval_request")
                self.post.assert_not_called()
                before = await replay_events(choice)
                self.assertEqual(len(before), 4)
                final = await self.invoke(choice, Command(resume={"choice": choice, "feedback": "Migration checked"}))
                self.assertEqual(final["final_action"], "committed" if choice == "approve" else "rejected")
                events = await self.assert_valid_events(choice)
                self.assertEqual([e["action"] for e in events], [
                    "fetch_pr", "analyze", "route", "human_approval", "human_approval", "human_approval", "commit",
                ])
                self.assertEqual([e["decision"] for e in events[3:6]], ["pending", "pending", choice])
                self.assertEqual(events[5]["reason"], "Migration checked")
                self.assertEqual(events[-1]["decision"], choice)
                self.assertEqual(self.post.call_count, int(choice == "approve"))
                self.fetch.assert_called_once()
                self.llm.assert_awaited_once()

    async def test_auto_logs_low_risk_and_post_failure_outcome(self):
        self.analysis.confidence = 0.8
        self.post.side_effect = RuntimeError("GitHub unavailable")
        final = await self.invoke("auto")
        self.assertEqual(final["final_action"], "auto_commit_failed")
        events = await self.assert_valid_events("auto")
        self.assertEqual([e["action"] for e in events], ["fetch_pr", "analyze", "route", "auto_approve"])
        self.assertEqual([e["decision"] for e in events], ["pending", "pending", "auto", "auto"])
        self.assertIn("commit_failed", events[-1]["reason"])
        self.post.assert_called_once()

    async def test_escalation_logs_answers_and_new_confidence(self):
        self.analysis.confidence = 0.5
        self.analysis.escalation_questions = ["auth.py: why MD5?"]
        refined = PRAnalysis(summary="More unknowns", confidence=0.4, confidence_reasoning="Answers reveal missing context")
        self.llm.side_effect = [self.analysis, refined]
        await self.invoke("escalation")
        self.post.assert_not_called()
        answers = {"auth.py: why MD5?": "Prototype; production requirements unknown"}
        final = await self.invoke("escalation", Command(resume=answers))
        self.assertEqual(final["final_action"], "committed")
        events = await self.assert_valid_events("escalation")
        self.assertEqual([e["action"] for e in events], [
            "fetch_pr", "analyze", "route", "escalate", "escalate", "escalate", "synthesize", "commit",
        ])
        self.assertEqual(json.loads(events[5]["reason"]), answers)
        self.assertEqual(events[-2]["confidence"], refined.confidence)
        self.assertEqual(events[-1]["confidence"], refined.confidence)
        self.post.assert_called_once()
        self.assertIn("Reviewer answered escalation questions", self.post.call_args.args[1])
        prompt = self.llm.call_args.args[0][1]["content"]
        self.assertIn(self.analysis.model_dump_json(indent=2), prompt)

    async def test_run_resumes_saved_interrupt_and_does_not_restart_completed_thread(self):
        await self.invoke("resume")
        with patch.object(exercise, "handle_interrupt", return_value={"choice": "reject", "feedback": "No migration"}) as human:
            await exercise.run(self.pr_url, "resume")
            human.assert_called_once()
            events = await replay_events("resume")
            await exercise.run(self.pr_url, "resume")
            self.assertEqual(await replay_events("resume"), events)
            human.assert_called_once()
        self.fetch.assert_called_once()
        self.llm.assert_awaited_once()
        self.post.assert_not_called()
        with self.assertRaisesRegex(ValueError, "different PR"):
            await exercise.run("https://github.com/example/demo/pull/99", "resume")

    async def test_run_continues_failed_node_from_saved_checkpoint(self):
        self.llm.side_effect = [RuntimeError("LLM unavailable"), self.analysis]
        with self.assertRaisesRegex(RuntimeError, "LLM unavailable"):
            await self.invoke("failed-node")
        with patch.object(exercise, "handle_interrupt", return_value={"choice": "reject"}):
            await exercise.run(self.pr_url, "failed-node")
        self.fetch.assert_called_once()
        self.assertEqual(self.llm.await_count, 2)
        self.post.assert_not_called()
        events = await self.assert_valid_events("failed-node")
        self.assertEqual(events[-1]["decision"], "reject")
        self.assertEqual(sum(e["action"] == "fetch_pr" for e in events), 1)

    async def test_invalid_response_is_not_written_as_a_decision(self):
        await self.invoke("invalid")
        with self.assertRaisesRegex(ValueError, "approve, reject, or edit"):
            await self.invoke("invalid", Command(resume={"choice": "anything"}))
        await self.assert_valid_events("invalid")
        self.post.assert_not_called()

    async def test_replay_lists_worst_risk_by_severity_and_preserves_reason(self):
        self.analysis.confidence = 0.5
        await self.invoke("high-risk")
        output = StringIO()
        with patch.object(replay, "Console", return_value=Console(file=output, width=300)):
            await replay.list_threads()
            await replay.replay("high-risk")
        rendered = output.getvalue()
        self.assertIn("high-risk", rendered)
        self.assertIn("high", rendered)
        self.assertIn("Check migration [plan]", rendered)
        self.assertIn("decision=", rendered)
        self.assertIn("reviewer=", rendered)
        # Fetch has med risk, but the session's worst risk must still be high.
        table_output = StringIO()
        with patch.object(replay, "Console", return_value=Console(file=table_output, width=300)):
            await replay.list_threads()
        self.assertIn("high", table_output.getvalue())
        self.assertNotIn("med", table_output.getvalue())


if __name__ == "__main__":
    unittest.main()
