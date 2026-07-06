import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("self_code_lifecycle_fixed.py")


def load_module(temp_dir: str):
    os.environ["MODEL_ID"] = "test-model"
    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = types.SimpleNamespace(create=None)

    fake_anthropic.Anthropic = FakeAnthropic
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None
    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv

    previous_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        spec = importlib.util.spec_from_file_location("s17_fixed", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)


class LifecycleFixedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.module = load_module(self.temp.name)
        self.module.IDLE_LOOP_INTERVAL = 0.001
        self.module.IDLE_TIMEOUT = 0.002
        self.module.time.sleep = lambda _seconds: None

    def tearDown(self):
        self.temp.cleanup()

    def lifecycle_events(self, name):
        path = Path(self.temp.name) / ".agent_logs" / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]

    def test_idle_auto_claim_injects_full_task(self):
        task = self.module.create_task("write tests", "cover lifecycle")
        self.module.BUS.read_inbox = lambda _agent: []
        messages = []

        result = self.module.idle_poll("alice", messages, "alice", "developer")

        claimed = self.module.load_task(task.id)
        self.assertEqual(result, "work")
        self.assertEqual(claimed.status, "in_progress")
        self.assertEqual(claimed.owner, "alice")
        self.assertIn(task.subject, messages[-1]["content"])
        self.assertIn("idle_to_work", self.lifecycle_events("alice"))

    def test_owner_checks_prevent_steal_and_wrong_completion(self):
        task = self.module.create_task("owned task")
        task.owner = "bob"
        task.status = "pending"
        self.module._save_task(task)

        self.assertIn("already owned by bob", self.module.claim_task(task.id, "alice"))
        task.status = "in_progress"
        self.module._save_task(task)
        self.assertIn("owned by bob", self.module.complete_task(task.id, owner="alice"))

    def test_idle_inbox_wakes_and_consumes_message(self):
        self.module.BUS.send("lead", "alice", "new task details")
        messages = []

        result = self.module.idle_poll("alice", messages, "alice", "developer")

        self.assertEqual(result, "work")
        self.assertIn("new task details", messages[-1]["content"])
        self.assertFalse((Path(self.temp.name) / ".mailbox" / "alice.jsonl").exists())

    def test_idle_shutdown_replies_and_stops(self):
        self.module.BUS.send(
            "lead", "alice", "stop", "shutdown_request",
            {"request_id": "req_test"},
        )

        result = self.module.idle_poll("alice", [], "alice", "developer")
        replies = self.module.BUS.read_inbox("lead")

        self.assertEqual(result, "shutdown")
        self.assertEqual(replies[0]["type"], "shutdown_response")
        self.assertEqual(replies[0]["metadata"]["request_id"], "req_test")

    def test_full_work_idle_work_complete_lifecycle(self):
        task = self.module.create_task("build feature", "implement and finish")
        responses = [
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text="initial work done")],
            ),
            types.SimpleNamespace(
                stop_reason="tool_use",
                content=[types.SimpleNamespace(
                    type="tool_use", name="complete_task",
                    input={"task_id": task.id}, id="tool_complete",
                )],
            ),
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text="task completed")],
            ),
        ]
        self.module.client.messages.create = lambda **_kwargs: responses.pop(0)

        class ImmediateThread:
            def __init__(self, target=None, **_kwargs):
                self.target = target
            def start(self):
                self.target()

        real_thread = self.module.threading.Thread
        self.module.threading.Thread = ImmediateThread
        try:
            result = self.module.spawn_teammate_thread("alice", "developer", "start")
        finally:
            self.module.threading.Thread = real_thread

        completed = self.module.load_task(task.id)
        events = self.lifecycle_events("alice")
        self.assertIn("autonomous", result)
        self.assertEqual(completed.status, "completed")
        self.assertNotIn("alice", self.module.active_teammates)
        self.assertIn("task_claimed", events)
        self.assertIn("task_completed", events)
        self.assertIn("thread_stopped", events)
        self.assertEqual(responses, [])

    def test_llm_error_cleans_active_registry(self):
        self.module.client.messages.create = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("model unavailable")
        )

        class ImmediateThread:
            def __init__(self, target=None, **_kwargs):
                self.target = target
            def start(self):
                self.target()

        real_thread = self.module.threading.Thread
        self.module.threading.Thread = ImmediateThread
        try:
            self.module.spawn_teammate_thread("bob", "developer", "start")
        finally:
            self.module.threading.Thread = real_thread

        self.assertNotIn("bob", self.module.active_teammates)
        events = self.lifecycle_events("bob")
        self.assertIn("llm_error", events)
        self.assertIn("thread_stopped", events)

    def test_llm_error_requeues_owned_task(self):
        task = self.module.create_task("interrupted task")
        self.module.claim_task(task.id, "bob")
        self.module.client.messages.create = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("model unavailable")
        )

        class ImmediateThread:
            def __init__(self, target=None, **_kwargs):
                self.target = target
            def start(self):
                self.target()

        real_thread = self.module.threading.Thread
        self.module.threading.Thread = ImmediateThread
        try:
            self.module.spawn_teammate_thread("bob", "developer", "continue")
        finally:
            self.module.threading.Thread = real_thread

        recovered = self.module.load_task(task.id)
        self.assertEqual(recovered.status, "pending")
        self.assertIsNone(recovered.owner)
        self.assertIn("tasks_requeued", self.lifecycle_events("bob"))

    def test_startup_recovery_requeues_orphan(self):
        task = self.module.create_task("orphaned task")
        self.module.claim_task(task.id, "dead-agent")

        recovered_ids = self.module.recover_orphaned_tasks(active_owners=set())
        recovered = self.module.load_task(task.id)

        self.assertIn(task.id, recovered_ids)
        self.assertEqual(recovered.status, "pending")
        self.assertIsNone(recovered.owner)

    def test_concurrent_mailbox_send_and_drain_loses_no_messages(self):
        expected = 100
        received = []

        def sender():
            for index in range(expected):
                self.module.BUS.send("lead", "alice", f"message-{index}")

        thread = self.module.threading.Thread(target=sender)
        thread.start()
        while thread.is_alive():
            received.extend(self.module.BUS.read_inbox("alice"))
        thread.join()
        received.extend(self.module.BUS.read_inbox("alice"))

        contents = {message["content"] for message in received}
        self.assertEqual(len(contents), expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)