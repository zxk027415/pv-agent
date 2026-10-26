"""Native Ragas contract checks, executed with the optional runtime via unittest."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage

from pvdiag.agent.llm import LLMReasoner
from pvdiag.eval.ragas import GroundedAnswer, run_ragas
from pvdiag.eval.ragas_inputs import prepare_plan
from pvdiag.eval.ragas_store import RagasStore
from pvdiag.eval.retrieval import QASet
from pvdiag.knowledge.store import KnowledgeStore
from pvdiag.observability.store import canonical

ROOT = Path(__file__).resolve().parents[1]


class NativeRagasContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = RagasStore(Path(self.temporary.name) / "jobs.sqlite3")
        self.dataset = QASet.model_validate_json((ROOT / "bench/kb_qa_dev.json").read_bytes())
        # A unique fixture label proves that the generator is isolated from it.
        self.dataset.questions[0].reference_answer = "PRIVATE_REFERENCE_LABEL_FOR_JUDGE_ONLY"
        self.corpus = KnowledgeStore()
        self.plan = prepare_plan(self.dataset, self.corpus, question_ids=["KB001"])
        self.calls = []
        self.answer_prompts = []
        self.edit_prompt_path = None
        self.behavior = "normal"
        self.fail_once = False
        owner = self

        class OfflineSDK:
            def __init__(self):
                self.precisions = 0

            def with_structured_output(self, schema, **kwargs):
                self.schema = schema
                return self

            async def ainvoke(self, messages):
                data = json.loads(messages[1][1])
                name = self.schema.__name__
                owner.calls.append((name, data))
                if name == "GroundedAnswer":
                    owner.answer_prompts.append(messages[0][1])
                    if owner.edit_prompt_path and len(owner.answer_prompts) == 1:
                        owner.edit_prompt_path.write_text("changed after first answer")
                    owner.assertEqual(set(data), {"task", "question", "contexts"})
                    owner.assertNotIn("PRIVATE_REFERENCE_LABEL", canonical(data))
                    parsed = GroundedAnswer(
                        answer="受控支持事实。受控不支持事实。",
                        cited_document_ids=[data["contexts"][0]["document_id"]],
                    )
                elif name == "ContextPrecisionOutput":
                    value = [1, 0, 1][self.precisions % 3]
                    self.precisions += 1
                    parsed = self.schema(
                        reason="Controlled fixture, not semantic evaluation.",
                        verdict=2 if owner.behavior == "invalid_binary" else value,
                    )
                elif name == "ContextRecallOutput":
                    if owner.fail_once:
                        owner.fail_once = False
                        raise RuntimeError("PRIVATE_PROVIDER_ERROR_MUST_NOT_LEAK")
                    parsed = self.schema(
                        classifications=[
                            {"statement": "受控支持事实", "reason": "fixture", "attributed": 1},
                            {"statement": "受控不支持事实", "reason": "fixture", "attributed": 0},
                        ]
                    )
                elif name == "StatementGeneratorOutput":
                    parsed = self.schema(
                        statements=[]
                        if owner.behavior == "empty_statements"
                        else ["受控支持事实", "受控不支持事实"]
                    )
                elif name == "NLIStatementOutput":
                    statements = [
                        {"statement": "受控支持事实", "reason": "fixture", "verdict": 1},
                        {"statement": "受控不支持事实", "reason": "fixture", "verdict": 0},
                    ]
                    parsed = self.schema(
                        statements=statements[:1]
                        if owner.behavior == "omit_statement"
                        else statements
                    )
                else:
                    raise AssertionError("unknown native Ragas schema")
                return {
                    "raw": AIMessage(
                        content="",
                        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                        response_metadata={"model_name": "offline-ragas-fixture"},
                    ),
                    "parsed": parsed,
                    "parsing_error": None,
                }

        self.sdk_patch = patch(
            "pvdiag.agent.llm.init_chat_model", side_effect=lambda *args, **kwargs: OfflineSDK()
        )
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)

    def evaluate(self, *, resume=False, budgets=None, plan=None, dataset=None, factories=None):
        budgets = budgets or {"answer": 2, "judge": 10}
        answer_factory, judge_factory = factories or (
            lambda: LLMReasoner("offline-answer", max_model_calls=30),
            lambda: LLMReasoner("offline-judge", max_model_calls=30),
        )
        return asyncio.run(
            run_ragas(
                plan or self.plan,
                dataset or self.dataset,
                self.corpus,
                self.store,
                "native-contract",
                answer_factory,
                judge_factory,
                budgets=budgets,
                allow_generation=True,
                allow_judge=True,
                resume=resume,
                execution_mode="offline_transport_contract",
            )
        )

    def test_native_formulas_use_ranked_binary_judgments_and_separate_answer(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["metrics"]["context_precision"], (1 + 2 / 3) / (2 + 1e-10))
        self.assertEqual(result["metrics"]["context_recall"], 0.5)
        self.assertEqual(result["metrics"]["faithfulness"], 0.5)
        self.assertEqual(result["usage"]["answer"]["model_calls"], 1)
        self.assertEqual(result["usage"]["judge"]["model_calls"], 6)
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(result["execution_mode"], "offline_transport_contract")
        self.assertEqual(os.environ["RAGAS_DO_NOT_TRACK"], "true")
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(len(self.store.all("span")), 8)

    def test_complete_resume_has_no_new_sdk_calls_and_keeps_usage(self):
        first = self.evaluate()
        second = self.evaluate(resume=True)
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(len(self.store.all("span")), 8)

    def test_failed_call_is_counted_and_resume_reuses_validated_partial_results(self):
        self.fail_once = True
        failed = self.evaluate()
        self.assertEqual(failed["status"], "incomplete")
        self.assertEqual(failed["cases"][0]["status"], "failed")
        self.assertIsNone(failed["metrics"]["context_precision"])
        restored = self.evaluate(resume=True)
        self.assertEqual(restored["status"], "completed")
        self.assertEqual(restored["usage"]["answer"]["model_calls"], 1)
        self.assertEqual(restored["usage"]["judge"]["model_calls"], 7)
        self.assertEqual(restored["usage"]["judge"]["usage_missing_calls"], 1)
        self.assertIsNone(restored["usage"]["judge"]["input_tokens"])
        self.assertNotIn("PRIVATE_PROVIDER_ERROR", canonical(self.store.snapshot()))
        self.assertEqual(len(self.calls), 8)

    def test_budget_cannot_be_reset_or_expanded_by_resume(self):
        self.evaluate()
        with self.assertRaisesRegex(ValueError, "context or budgets"):
            self.evaluate(resume=True, budgets={"answer": 3, "judge": 10})
        self.assertEqual(len(self.calls), 7)

    def test_nonbinary_judge_output_does_not_become_a_high_score(self):
        self.behavior = "invalid_binary"
        result = self.evaluate()
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all(v is None for v in result["metrics"].values()))
        self.assertEqual(result["usage"]["judge"]["model_calls"], 1)
        self.assertEqual(len(self.calls), 2)

    def test_omitted_faithfulness_statement_cannot_inflate_score(self):
        self.behavior = "omit_statement"
        result = self.evaluate()
        self.assertEqual(result["cases"][0]["status"], "failed")
        self.assertIsNone(result["metrics"]["faithfulness"])
        self.assertEqual(result["usage"]["judge"]["model_calls"], 6)

    def test_native_nan_is_reported_as_missing_with_explicit_coverage(self):
        self.behavior = "empty_statements"
        result = self.evaluate()
        self.assertEqual(result["cases"][0]["status"], "unscorable")
        self.assertIsNone(result["metrics"]["faithfulness"])
        self.assertEqual(result["coverage"]["faithfulness"], 0)
        self.assertEqual(result["usage"]["judge"]["model_calls"], 5)
        json.dumps(result, allow_nan=False)

    def test_empty_retrieval_never_constructs_a_model_or_fabricates_metrics(self):
        dataset = self.dataset.model_copy(deep=True)
        dataset.questions[0].question = "zzzxqv_unmatched_word"
        plan = prepare_plan(dataset, self.corpus, question_ids=["KB001"])
        self.assertEqual(plan.required_calls(), {"answer": 0, "judge": 0})

        def forbidden():
            self.fail("no-context runs must not construct any model")

        result = self.evaluate(
            plan=plan,
            dataset=dataset,
            budgets={"answer": 0, "judge": 0},
            factories=(forbidden, forbidden),
        )
        self.assertEqual(result["cases"][0]["status"], "no_context")
        self.assertTrue(all(v is None for v in result["metrics"].values()))
        self.assertEqual(self.calls, [])

    def test_answer_prompt_is_frozen_across_the_whole_job(self):
        prompt = Path(self.temporary.name) / "prompt.md"
        prompt.write_text("frozen controlled answer prompt")
        self.edit_prompt_path = prompt
        plan = prepare_plan(self.dataset, self.corpus, question_ids=["KB001", "KB002"])
        with patch("pvdiag.eval.ragas.ANSWER_PROMPT", prompt):
            result = self.evaluate(plan=plan, budgets={"answer": 2, "judge": 12})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.answer_prompts, ["frozen controlled answer prompt"] * 2)
        self.assertEqual(prompt.read_text(), "changed after first answer")


if __name__ == "__main__":
    with (
        patch(
            "socket.socket.connect", side_effect=AssertionError("external_network_forbidden")
        ) as connect,
        patch("socket.getaddrinfo", side_effect=AssertionError("external_dns_forbidden")) as dns,
    ):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeRagasContracts)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        print(
            json.dumps(
                {
                    "native_ragas_tests": outcome.testsRun,
                    "external_network_attempts": connect.call_count + dns.call_count,
                }
            )
        )
        raise SystemExit(
            0 if outcome.wasSuccessful() and connect.call_count + dns.call_count == 0 else 1
        )
