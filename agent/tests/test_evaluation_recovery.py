"""Synthetic CPU checks only; no claims about model accuracy."""
import copy
import importlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import unittest

from generative import evaluation_recovery as resume


class ShardingTests(unittest.TestCase):
    def setUp(self):
        import generative.evaluate as module
        self.module = importlib.reload(module)

    def tearDown(self):
        importlib.reload(self.module)

    def test_real_evaluator_matches_unsharded_in_both_entity_conditions(self):
        from generative.protocol import ProtocolConfig, dumps
        text = 'Fatigue is a core symptom of Condition A and supports diagnosis of Condition A.'
        graph = {'entities': [{'id': 'S1', 'label': 'Symptom', 'name': 'Fatigue'},
                              {'id': 'D1', 'label': 'Disease', 'name': 'Condition A'}],
                 'relations': [{'source': 'S1', 'target': 'D1', 'relation': 'is_core_symptom_of', 'evidence': text}]}
        records = [{'messages': [{'role': 'user', 'content': text},
                                {'role': 'assistant', 'content': dumps(graph)}]} for _ in range(11)]
        cached = [{'output': copy.deepcopy(graph)} for _ in records]
        cached[5]['output']['entities'].pop()  # Missing candidate must remain FN.
        config = ProtocolConfig(strict_evidence=True)

        def generate(batch):
            # Include a valid positive and a visible invalid emitted edge.
            decisions = [{'pair_id': p['pair_id'], 'relations': []} for p in batch['pairs']]
            for decision, pair in zip(decisions, batch['pairs']):
                if pair['source'] == 'S1' and pair['target'] == 'D1':
                    decision['relations'].append({'relation': 'is_core_symptom_of', 'evidence_span_ids': ['SENT_001']})
            if decisions:
                decisions[0]['relations'].append({'relation': 'INVALID_LABEL', 'evidence_span_ids': ['SENT_001']})
            return {'text': dumps({'decisions': decisions}), 'tokens': 23, 'truncated': False}

        for oracle in (False, True):
            module = importlib.reload(self.module)
            expected, expected_raw, expected_counts = module.run_records(records, cached, config, generate, oracle=oracle)
            expected_metrics = module.evaluate_records(records, expected)
            self.assertGreater(expected_metrics['free_text']['relation']['tp'], 0)
            self.assertGreater(expected_metrics['free_text']['relation']['fp'], 0)
            expected_rules = module.rule_ablation(records, expected)
            merged, raw, counts, seen = [None] * len(records), [], Counter(), set()
            for shard in range(4):
                module = importlib.reload(self.module)
                indices = list(range(shard, len(records), 4))
                resume.install_shard_hooks(module, indices, len(records))
                predictions, generated, audit = module.run_records(records, cached, config, generate, oracle=oracle)
                self.assertEqual(module.evaluate_records(records, predictions)['overall']['samples'], len(indices))
                module.rule_ablation(records, predictions)
                for i, row in zip(indices, predictions):
                    self.assertNotIn(i, seen); seen.add(i); merged[i] = row
                raw.extend(generated); counts.update(audit)
            module = importlib.reload(self.module)
            self.assertEqual(seen, set(range(len(records))))
            self.assertEqual(merged, expected)
            self.assertEqual(sorted(raw, key=lambda r: r['sample_id']), expected_raw)
            self.assertEqual(dict(counts), expected_counts)
            self.assertEqual(module.evaluate_records(records, merged), expected_metrics)
            self.assertEqual(module.rule_ablation(records, merged), expected_rules)

    def test_cache_mismatch_rejected(self):
        resume.install_shard_hooks(self.module, [0, 4], 8)
        with self.assertRaisesRegex(ValueError, 'alignment changed'):
            self.module.run_records([None] * 8, [None] * 7, None, None)


if __name__ == '__main__':
    unittest.main()
