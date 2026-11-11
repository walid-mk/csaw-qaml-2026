import importlib.util
import json
from pathlib import Path
import unittest
import torch
from client.defense import DefendedModel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('example_protocol', ROOT/'examples/custom-defense/protocol.py')
protocol = importlib.util.module_from_spec(spec); spec.loader.exec_module(protocol)


class DefenseTests(unittest.TestCase):
    def test_preprocessing_without_training_and_adaptive_gradient(self):
        x = torch.rand(2,3,32,32,dtype=torch.float64,requires_grad=True)
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3072,4,dtype=torch.float64))
        before = {k:v.clone() for k,v in model.state_dict().items()}
        state = protocol.fit(model,x,torch.tensor([0,1]),seed=2026,config={'strength':.1})
        wrapped = DefendedModel(model,protocol.preprocess,state=state,config={'strength':.1})
        logits = wrapped(x)
        self.assertEqual(tuple(logits.shape),(2,4))
        self.assertGreater(torch.autograd.grad(logits.sum(),x)[0].norm(),0)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_bpda_uses_true_forward_and_surrogate_backward(self):
        x=torch.tensor([[[[.13]]]],dtype=torch.float64,requires_grad=True)
        def quantize(images,**kwargs):return torch.round(images*4)/4
        wrapped=DefendedModel(torch.nn.Identity(),quantize,backward='bpda_identity')
        y=wrapped(x)
        self.assertEqual(y.item(),.25)
        self.assertEqual(torch.autograd.grad(y.sum(),x)[0].item(),1)

    def test_invalid_transform_rejected(self):
        for transform in (lambda x,**kw:x.float(),lambda x,**kw:x+2,lambda x,**kw:x[:0]):
            with self.assertRaises(ValueError):
                DefendedModel(torch.nn.Identity(),transform)(torch.zeros(1,3,32,32,dtype=torch.float64))

if __name__=='__main__':unittest.main()
