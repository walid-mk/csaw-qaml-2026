import unittest
from pathlib import Path

import numpy as np

from kit.sdk import verify_assets


class KitContractTests(unittest.TestCase):
    def test_exact_public_model_inventory(self):
        manifest=verify_assets()
        expected={f'models/{family}/{kind}/run-{run}.npz'
                  for family in ('qnn','qcnn') for kind in ('original','hardened') for run in range(5)}
        actual={p for p in manifest['files'] if p.startswith('models/')}
        self.assertEqual(actual,expected)
        self.assertEqual(manifest['status'],'released')

    def test_training_and_attack_data_boundary(self):
        root=Path(__file__).resolve().parents[1]/'kit'
        with np.load(root/'training/road-signs-training.npz',allow_pickle=False) as data:
            self.assertEqual(set(data.files),{'ids','images','labels'})
            self.assertEqual(data['images'].shape,(889,3,32,32))
            self.assertEqual(data['images'].dtype,np.float64)
            self.assertEqual(set(data['labels'].tolist()),{0,1,2,3})
            train_ids=set(data['ids'].tolist())
        for family in ('qnn','qcnn'):
            for access in ('whitebox','blackbox'):
                with np.load(root/'inputs'/f'{family}-{access}.npz',allow_pickle=False) as data:
                    self.assertEqual(set(data.files),{'ids','images','targets'})
                    self.assertTrue(train_ids.isdisjoint(data['ids'].tolist()))
                    self.assertEqual(data['images'].shape,(len(data['ids']),3,32,32))
                    self.assertTrue(np.isfinite(data['images']).all())


if __name__=='__main__':unittest.main()
