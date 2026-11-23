import importlib.util
import json
import unittest


SPEC = importlib.util.spec_from_file_location(
    "apply_mof_review_decisions",
    "apply_mof_review_decisions.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ApplyReviewTests(unittest.TestCase):
    @staticmethod
    def make_fixture(decisions):
        generated = [
            {"prompt": f"The Token{index} sample is harmless today.", "label": 0, "candidate_tokens": [f"Token{index}"]}
            for index in range(1, len(decisions) + 1)
        ]
        audit = [
            {
                "sample_id": str(index),
                "prompt": row["prompt"],
                "candidate_tokens": json.dumps(row["candidate_tokens"]),
                "label": "0",
            }
            for index, row in enumerate(generated, 1)
        ]
        reviews = []
        for index, decision in enumerate(decisions, 1):
            prompt = generated[index - 1]["prompt"]
            if decision == "edit":
                prompt = f"The Token{index} sample was manually reviewed today."
            reviews.append(
                {
                    "sample_id": str(index),
                    "prompt": prompt,
                    "candidate_tokens": json.dumps(generated[index - 1]["candidate_tokens"]),
                    "label": "0",
                    "manual_review": decision,
                }
            )
        return generated, audit, reviews

    def test_current_valid_distribution_derives_54_approved(self):
        decisions = ["keep"] * 40 + ["edit"] * 14 + ["reject"] * 9
        generated, audit, reviews = self.make_fixture(decisions)
        selected, counts = MODULE.validate_and_build(generated, audit, reviews)
        self.assertEqual(counts, {"keep": 40, "edit": 14, "reject": 9})
        self.assertEqual(len(selected), 54)
        self.assertEqual({row["manual_review_decision"] for row in selected}, {"keep", "edit"})

    def test_different_valid_distribution_is_accepted(self):
        generated, audit, reviews = self.make_fixture(["keep", "edit", "reject"])
        selected, counts = MODULE.validate_and_build(generated, audit, reviews)
        self.assertEqual(counts, {"keep": 1, "edit": 1, "reject": 1})
        self.assertEqual(len(selected), 2)

    def test_keep_uses_generated_and_edit_uses_review_prompt(self):
        generated = [
            {"prompt": "The Alpha sample is harmless today.", "label": 0, "candidate_tokens": ["Alpha"]},
            {"prompt": "The Beta sample is harmless today.", "label": 0, "candidate_tokens": ["Beta"]},
            {"prompt": "The Gamma sample is harmless today.", "label": 0, "candidate_tokens": ["Gamma"]},
        ]
        audit = [
            {"sample_id": str(i), "prompt": row["prompt"], "candidate_tokens": str(row["candidate_tokens"]).replace("'", '"'), "label": "0"}
            for i, row in enumerate(generated, 1)
        ]
        reviews = [
            {"sample_id": "1", "prompt": "Changed Alpha text.", "candidate_tokens": '["Alpha"]', "label": "0", "manual_review": "keep"},
            {"sample_id": "2", "prompt": "The Beta sample was manually edited today.", "candidate_tokens": '["Beta"]', "label": "0", "manual_review": "EDIT"},
            {"sample_id": "3", "prompt": "The Gamma sample is harmless today.", "candidate_tokens": '["Gamma"]', "label": "0", "manual_review": "reject"},
        ]
        selected, counts = MODULE.validate_and_build(generated, audit, reviews)
        self.assertEqual(counts, {"keep": 1, "edit": 1, "reject": 1})
        self.assertEqual(selected[0]["prompt"], generated[0]["prompt"])
        self.assertEqual(selected[0]["prompt_source"], "generated")
        self.assertEqual(selected[1]["prompt"], reviews[1]["prompt"])
        self.assertEqual(selected[1]["prompt_source"], "manual_edit")


if __name__ == "__main__":
    unittest.main()
