"""Exercise 5 — Streamlit approval UI backed by the Exercise 4 SQLite graph.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import re
import uuid
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


async def recent_sessions() -> list[dict]:
    """Return queryable session summaries, ordered by the latest audit event."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low'
                   END AS worst_risk,
                   MAX(timestamp) AS last_event
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]


def snapshot_result(snapshot) -> dict:
    """Expose pending interrupts without invoking or restarting the graph."""
    result = dict(snapshot.values)
    pending = [item for task in snapshot.tasks for item in task.interrupts]
    if pending:
        result["__interrupt__"] = pending
    return result


async def load_session(thread_id: str) -> tuple[dict, bool]:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.values:
            raise ValueError("Session not found. Select a recent session or start a review.")
        result = snapshot_result(snapshot)
        return result, bool(snapshot.next and "__interrupt__" not in result)


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Start once, resume an interrupt, or continue an unfinished checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(cfg)
        if snapshot.values:
            if snapshot.values["pr_url"] != pr_url:
                raise ValueError("The selected session belongs to a different PR.")
            if not snapshot.next:
                return snapshot_result(snapshot)
            if resume_value is not None:
                return await graph.ainvoke(Command(resume=resume_value), cfg)
            result = snapshot_result(snapshot)
            if "__interrupt__" in result:
                return result
            return await graph.ainvoke(None, cfg)
        if resume_value is not None:
            raise ValueError("Cannot resume a session without a checkpoint.")
        return await graph.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


def remember_result(result: dict, can_continue: bool = False) -> None:
    st.session_state.pr_url = result["pr_url"]
    st.session_state.can_continue = can_continue
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result if result.get("final_action") else None


def select_session(thread_id: str) -> None:
    result, can_continue = asyncio.run(load_session(thread_id))
    remember_result(result, can_continue)
    st.session_state.thread_id = thread_id
    st.session_state.loaded_thread = thread_id
    st.session_state.pr_url_input = result["pr_url"]
    st.query_params["thread"] = thread_id


def render_approval_card(payload: dict) -> dict | None:
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{comment.get('line') or '?'}` — {comment['body']}"
        )
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_input(
        "Feedback (optional)", key=f"approval_feedback_{st.session_state.thread_id}",
    )
    approve, reject, edit = st.columns(3)
    if approve.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if reject.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if edit.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    st.caption("Edit saves your feedback without posting a comment.")
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            question: st.text_input(
                question, key=f"answer_{st.session_state.thread_id}_{index}",
            ).strip()
            for index, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        if not answers or any(not answer for answer in answers.values()):
            st.warning("Please answer every question before submitting.")
        else:
            return answers
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    if action in {"auto_committed", "auto_approved", "committed", "committed_after_escalation"}:
        st.success(f"✓ {action} — comment posted")
        st.link_button("View comment on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        message = (
            "Edit requested — feedback saved; no comment posted."
            if final.get("human_choice") == "edit"
            else "Rejected — no comment posted."
        )
        st.warning(message)
    elif action in {"commit_failed", "auto_commit_failed"}:
        st.error("Posting the comment failed. Check your GitHub token and repository access.")
    else:
        st.info(f"final_action = {action}")
    analysis = final.get("analysis")
    if analysis:
        st.metric("Confidence", f"{analysis.confidence:.0%}")
        st.markdown(analysis.summary)
        st.caption(analysis.confidence_reasoning)
        for comment in analysis.comments:
            st.markdown(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` — {comment.body}")
    if final.get("escalation_answers"):
        with st.expander("Reviewer answers"):
            for question, answer in final["escalation_answers"].items():
                st.markdown(f"**{question}**")
                st.write(answer)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    st.title("HITL PR Review Agent")
    defaults = {
        "thread_id": None, "loaded_thread": None, "pr_url": "",
        "pr_url_input": "", "interrupt_payload": None, "final": None,
        "can_continue": False, "sidebar_version": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    requested_thread = st.query_params.get("thread")
    if requested_thread and requested_thread != st.session_state.loaded_thread:
        try:
            select_session(requested_thread)
        except Exception as exc:
            st.error(f"Could not restore session: {exc}")
            st.query_params.pop("thread", None)

    with st.sidebar:
        st.header("Recent sessions")
        try:
            rows = asyncio.run(recent_sessions())
            if rows:
                event = st.dataframe(
                    rows, hide_index=True, on_select="rerun", selection_mode="single-row",
                    column_config={"pr_url": st.column_config.LinkColumn("PR URL")},
                    key=f"recent_sessions_{st.session_state.sidebar_version}",
                )
                if event.selection.rows:
                    try:
                        select_session(rows[event.selection.rows[0]]["thread_id"])
                    except Exception as exc:
                        st.error(f"Could not open session: {exc}")
                    else:
                        st.session_state.sidebar_version += 1
                        st.rerun()
            else:
                st.caption("No review sessions yet.")
        except Exception as exc:
            st.error(f"Could not load recent sessions: {exc}")

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", key="pr_url_input",
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review")
    if submitted:
        parsed = urlparse(pr_url.strip())
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or not re.fullmatch(r"/[^/]+/[^/]+/pull/[1-9][0-9]*/?", parsed.path)
        ):
            st.error("Enter a GitHub PR URL such as https://github.com/owner/repo/pull/1.")
        else:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            st.session_state.loaded_thread = thread_id
            st.session_state.pr_url = pr_url.strip()
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.session_state.can_continue = False
            st.session_state.sidebar_version += 1
            st.query_params["thread"] = thread_id
            try:
                with st.spinner("Fetching PR + asking the LLM..."):
                    remember_result(asyncio.run(run_graph(st.session_state.pr_url, thread_id)))
            except Exception as exc:
                st.error(f"Review failed: {exc}. Reload this session to recover its saved progress.")
            else:
                st.rerun()

    if st.session_state.can_continue:
        st.info("This session has saved progress and is ready to continue.")
        if st.button("Continue review", type="primary"):
            try:
                with st.spinner("Continuing review..."):
                    remember_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id)))
            except Exception as exc:
                st.error(f"Could not continue review: {exc}")
            else:
                st.rerun()

    payload = st.session_state.interrupt_payload
    if payload is not None:
        if payload["kind"] == "approval_request":
            answer = render_approval_card(payload)
        elif payload["kind"] == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error("This session contains an unsupported approval request.")
            answer = None
        if answer is not None:
            try:
                with st.spinner("Resuming..."):
                    remember_result(asyncio.run(run_graph(
                        st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                    )))
            except Exception as exc:
                st.error(f"Could not resume review: {exc}. Reload the session to inspect its saved progress.")
            else:
                st.rerun()

    if st.session_state.final is not None:
        render_final(st.session_state.final)
    if st.session_state.thread_id:
        st.caption(f"thread_id = {st.session_state.thread_id}")
        st.code(f"uv run python -m audit.replay --thread {st.session_state.thread_id}", language="bash")


if __name__ == "__main__":
    main()
