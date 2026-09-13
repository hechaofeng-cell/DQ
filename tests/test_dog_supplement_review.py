import tempfile
import unittest
from pathlib import Path

from scripts.dog_supplement_review_server import Store


class SupplementReviewTests(unittest.TestCase):
    def test_fixed_queue_and_all_media_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            self.assertEqual(len(store.cases), 48)
            self.assertEqual(sum(c['group'] != 'random' for c in store.cases), 18)
            for case in store.cases:
                for kind in ('source', 'marked', 'crop'):
                    self.assertTrue(store.media(kind, case['image_id']).is_file())
            self.assertIsNone(store.media('source', '../invalid'))

    def test_save_preserves_history_and_original_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            case = next(c for c in store.cases if c['group'] == 'critical')
            features = store.features[case['image_id']]['features']
            value = {'image_id': case['image_id'], 'reviewer': 'test', 'target_match': 'yes', 'target_kind': 'real_dog', 'decision': 'accept', 'human_features': {s.split('=')[0]: features[s.split('=')[0]] for s in case['gap_states']}, 'notes': ''}
            store.save(value)
            store.save({**value, 'notes': 'second'})
            self.assertEqual(len(store.records()), 1)
            self.assertEqual(len(list((Path(directory) / 'history' / case['image_id']).glob('*.json'))), 2)

    def test_cannot_accept_nonliving_target(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            case = store.cases[0]
            with self.assertRaises(ValueError):
                store.save({'image_id': case['image_id'], 'reviewer': 'test', 'target_match': 'yes', 'target_kind': 'representation', 'decision': 'accept', 'human_features': {}})


if __name__ == '__main__':
    unittest.main()
