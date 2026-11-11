import importlib.util
from pathlib import Path
import json,unittest
from client.prediction import PredictionError
spec=importlib.util.spec_from_file_location('check',Path(__file__).resolve().parents[1]/'examples/check_prediction_access.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DeviceCheckTests(unittest.TestCase):
 def bundle(self):return {'purpose':'organizer_participant_device_check','check_id':'example-nonce','QAML_PREDICT_URL':'https://example.invalid/qaml','QAML_QUERY_TOKEN':'private-example-token','checks':[{'family':f,'ids':['example-id'],'images':[[[[0.0]]]]} for f in ('qnn','qcnn')]}
 def test_success_does_not_attest_device_automatically_or_echo_token(self):
  calls=[]
  def request(*args):
   calls.append(args);return {'probabilities':[[.25]*4],'queries':1,'policy':{'version':'qaml-limits-v1'},'policy_sha256':'test-policy'}
  result=module.run_check(self.bundle(),request)
  self.assertTrue(result['passed']);self.assertFalse(result['separate_device_attested']);self.assertEqual(len(calls),2)
  self.assertNotIn('private-example-token',json.dumps(result));self.assertEqual(result['check_id'],'example-nonce')
 def test_failure_no_retries_or_secret_error_text(self):
  calls=[]
  def request(*args):calls.append(1);raise PredictionError('private-example-token')
  result=module.run_check(self.bundle(),request)
  self.assertFalse(result['passed']);self.assertEqual(len(calls),2)
  self.assertNotIn('private-example-token',json.dumps(result))
 def test_wrong_bundle_rejected_before_request(self):
  bundle=self.bundle();bundle['purpose']='student-credential'
  with self.assertRaises(ValueError):module.run_check(bundle)
if __name__=='__main__':unittest.main()
