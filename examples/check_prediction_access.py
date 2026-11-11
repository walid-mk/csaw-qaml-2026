"""Run an organizer-issued connection check from a participant laptop.

Uses only the Python standard library and the bundled public prediction client.
The private check bundle contains a temporary organizer probe token, never a
GitHub key. No student allowance or competition score is used by that bundle.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

# Also works in the organizer's portable check folder containing client/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client.access import prediction_url
from client.prediction import predict, PredictionError


def run_check(bundle, request=predict):
    if bundle.get('purpose') != 'organizer_participant_device_check':
        raise ValueError('Use the issued temporary device-check bundle')
    checks = bundle['checks']
    if {x['family'] for x in checks} != {'qnn', 'qcnn'} or len(checks) != 2:
        raise ValueError('Expected one check for each model family')
    result = {'schema':1, 'purpose':bundle['purpose'], 'check_id':bundle['check_id'],
              'utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
              'platform':platform.system(), 'results':[], 'separate_device_attested':False}
    for check in checks:
        started = time.monotonic()
        try:
            response = request(prediction_url(bundle['QAML_PREDICT_URL'],check['family'],'untargeted',0),
                               bundle['QAML_QUERY_TOKEN'],check['ids'],check['images'])
            if response.get('policy',{}).get('version') != 'qaml-limits-v1':
                raise ValueError('Unexpected service policy; ask organizers')
            row = {'family':check['family'],'passed':True,'probabilities':response['probabilities'],
                   'queries':response['queries'],'policy_sha256':response['policy_sha256']}
        except (PredictionError, ValueError):
            # Do not echo URL, body, headers or token, including unexpected errors.
            row = {'family':check['family'],'passed':False,
                   'message':'Connection or response check failed; send this report privately. Do not automatically retry.'}
        row['seconds'] = round(time.monotonic()-started,3)
        result['results'].append(row)
    result['passed'] = all(x['passed'] for x in result['results'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('participant-device-result.json'))
    parser.add_argument('--separate-device',action='store_true',
                        help='Attest this is a participant laptop, not the prediction server')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new report path; keep previous results')
    with args.bundle.open() as file:bundle=json.load(file)
    result=run_check(bundle)
    result['separate_device_attested']=args.separate_device
    with args.output.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result,indent=2))
    print('Return this result file only. Keep the bundle private; it contains a temporary token.')
    return 0 if result['passed'] and args.separate_device else 1

if __name__=='__main__':raise SystemExit(main())
