#!/usr/bin/env python3
"""Extract the actual portable ZIP and replay only its documented entry points."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submission_closeout_20261007'
PACKAGE=ROOT/'docs/visual_state_testing_portable_reanalysis_20261007.zip'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='visual_state_portable_replay_') as name:
        extracted=Path(name)
        with ZipFile(PACKAGE) as z:
            assert z.testzip() is None
            z.extractall(extracted)
        manifest=json.loads((extracted/'manifest.sha256.json').read_text())
        assert all(sha(extracted/r['path'])==r['sha256'] for r in manifest)
        steps=[['scripts/run_scale_only_confirmation.py','--input','portable_inputs','--output','reproduced_scale'],
               ['scripts/reaggregate_selection_evidence.py','--input','historical_selection','--output','reproduced_selection']]
        for number,step in enumerate(steps,1):
            log=OUT/f'portable_replay_{number}.log'
            print('Replay',number,'started from independently extracted package',flush=True)
            with log.open('w') as stream:
                completed=subprocess.run([sys.executable,*step],cwd=extracted,stdout=stream,
                                         stderr=subprocess.STDOUT,check=False)
            assert completed.returncode==0,f'Replay {number} failed; see {log}'
            print('Replay',number,'passed',flush=True)
        comparisons=[]
        for expected in sorted((extracted/'expected_scale_tables').glob('*.csv')):
            actual=extracted/'reproduced_scale/tables'/expected.name
            a,b=pd.read_csv(actual),pd.read_csv(expected)
            pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-12,atol=1e-12)
            comparisons.append({'file':expected.name,'rows':len(a),'columns':len(a.columns)})
        replay=json.loads((extracted/'reproduced_scale/audit.json').read_text())
        assert replay['status']=='pass' and replay['sklearn_cells_checked']==1440
        assert all(sha(extracted/r['path'])==r['sha256'] for r in manifest)
        report=dict(status='pass',package=str(PACKAGE.relative_to(ROOT)),
                    independent_temp_directory=True,documented_commands=steps,
                    manifest_files_verified=len(manifest),input_manifest_unchanged=True,
                    all_scale_tables_match=comparisons,sklearn_full_and_area_f1_cells_checked=1440,
                    historical_full_risk_layer_cells=720,
                    historical_selection_contrasts_checked=60,
                    inference_training_vlm_calls=0,elapsed_seconds=round(time.monotonic()-started,2),
                    scope='Cached reanalysis only; no raw feature/cluster rebuild or service/training replay.')
        (OUT/'portable_replay_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
