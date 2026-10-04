from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from tests.factory import bundle, ROOT
from audit.report import numerical_results, decimal_endpoint
from audit.verify import verify


class PipelineTests(unittest.TestCase):
    def test_complete_document_independent_numerical_report(self):
        result,rows,parameters=verify(bundle())
        out=numerical_results(result,rows,parameters)
        for group in rows:
            self.assertEqual(set(out['intervals'][group]),set(rows[group]))
        self.assertEqual(sum(k.startswith('J_') for k in out['intervals']['feasible']),36)
        self.assertEqual(sum(k.startswith('F_') for k in out['intervals']['feasible']),6)
        self.assertIn('matrix',out['parameters_exact']['primal'])
        self.assertIn('gain_ratio',out['derived'])

    def test_decimal_display_rounds_outward_with_negative_values(self):
        from fractions import Fraction as F
        for x in (F(-1,3),F(1,3),F(-1,10**30),F(0),F(7,2)):
            self.assertLessEqual(F(decimal_endpoint(x)),x)
            self.assertGreaterEqual(F(decimal_endpoint(x,True)),x)

    def test_derived_bound_is_not_compared_to_fixed_target(self):
        from copy import deepcopy
        from tests.factory import embed
        from audit.exact import rational
        from fractions import Fraction as F
        b=deepcopy(bundle()); name='reference/reference-values.txt'
        lines=b['files'][name]['text'].splitlines()
        old=next(line.split() for line in lines if line.startswith('base '))
        lines=[f'base {rational(old[1])-F(1,100)} {rational(old[2])-F(1,100)}' if line.startswith('base ') else line for line in lines]
        b['files'][name]=embed('\n'.join(lines)+'\n')
        report,_,_=verify(b)
        self.assertLess(report['bound'][0],F('1.14'))
        self.assertGreater(report['ratios']['gain_ratio'][1],F('1.000004'))

    def test_native_failure_never_publishes_completion(self):
        from programs import common
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'programs').mkdir(); (root/'inputs').mkdir()
            with patch.multiple(common,ROOT=root,INPUTS=root/'inputs',RESULTS=root/'results',LOGS=root/'results/logs'), patch.object(common,'toolchain_state',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    with common.calculation('sine',('values.txt',)) as out:
                        (out/'values.txt').write_text('partial')
                        raise RuntimeError('simulated native rejection')
                self.assertFalse((root/'results/sine').exists())
                self.assertTrue(list((root/'results/logs').glob('sine-*')))

    def test_resume_checks_content_and_required_outputs(self):
        from programs import common
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'programs').mkdir(); (root/'inputs').mkdir()
            with patch.multiple(common,ROOT=root,INPUTS=root/'inputs',RESULTS=root/'results',LOGS=root/'results/logs'), patch.object(common,'toolchain_state',return_value={}):
                with common.calculation('sine',('v.txt',)) as out: (out/'v.txt').write_text('good')
                with common.calculation('sine',('v.txt',)) as out: self.assertIsNone(out)
                (root/'results/sine/v.txt').write_text('tampered')
                with self.assertRaisesRegex(RuntimeError,'stale'):
                    with common.calculation('sine',('v.txt',)): pass

    def test_workflow_requires_explicit_resume_and_owned_output(self):
        from programs.workflow import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'result';out.mkdir();(out/'user.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError,'empty'):
                run(out,inputs=ROOT/'inputs')
            self.assertEqual((out/'user.txt').read_text(),'keep')

    def test_failed_gate_retains_work_and_resume_reuses_only_bound_archive(self):
        # Orchestration only: no numerical executable is invoked by these mocks.
        from programs import workflow
        from programs.common import json_write, digest
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'result'
            def fake_export(data,target):
                target.write_bytes(b'mock archive')
                return {'sha256':digest(target)}
            def rejection(path,report,*args):
                json_write(report,{'status':'FAIL','error':'mock native rejection'})
                return 1
            with patch.object(workflow.subprocess,'run'), \
                 patch.object(workflow,'export',side_effect=fake_export), \
                 patch.object(workflow,'check_archive',side_effect=rejection):
                self.assertEqual(workflow.run(out,inputs=ROOT/'inputs'),1)
            self.assertTrue((out/'work').is_dir())
            self.assertFalse((out/'results.txt').exists())
            self.assertEqual(json.loads((out/'run.json').read_text())['status'],'FAILED')
            with self.assertRaisesRegex(ValueError,'already exists'):
                workflow.run(out,inputs=ROOT/'inputs')
            with patch.object(workflow.subprocess,'run',side_effect=AssertionError('must reuse archive')), \
                 patch.object(workflow,'check_archive',side_effect=rejection):
                self.assertEqual(workflow.run(out,inputs=ROOT/'inputs',resume=True),1)
            (out/'proof.tar.xz').write_bytes(b'changed')
            with patch.object(workflow,'check_archive',side_effect=AssertionError('must reject first')):
                self.assertEqual(workflow.run(out,inputs=ROOT/'inputs',resume=True),1)
            self.assertIn('changed proof archive',json.loads((out/'run.json').read_text())['error'])
