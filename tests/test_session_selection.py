import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy.io as sio

from data.seed import prepare_seed_features
from data.session_selection import parse_sessions, session_tag


class SessionSelectionTest(unittest.TestCase):
    def test_supported_selectors(self):
        self.assertEqual(parse_sessions(1), [1])
        self.assertEqual(parse_sessions('2'), [2])
        self.assertEqual(parse_sessions('123'), [1, 2, 3])
        self.assertEqual(parse_sessions('1,2,3'), [1, 2, 3])
        self.assertEqual(parse_sessions([1, 2, 3]), [1, 2, 3])
        self.assertEqual(session_tag([1, 2, 3]), '123')

    def test_invalid_selector(self):
        with self.assertRaises(ValueError):
            parse_sessions('4')

    def test_seed_preparation_combines_sessions_without_merging_subjects(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature = np.zeros((62, 4, 5), dtype=np.float32)
            for session in (1, 2):
                session_dir = root / str(session)
                session_dir.mkdir()
                sio.savemat(session_dir / f'1_session{session}.mat', {'de_LDS1': feature + session})

            arrays = prepare_seed_features(
                root, root / 'combined.npz', time_steps=2, sessions=[1, 2]
            )

            self.assertEqual(arrays['x'].shape, (6, 2, 62, 5))
            self.assertEqual(np.unique(arrays['subject_id']).tolist(), [0])
            self.assertEqual(np.unique(arrays['session_id']).tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
