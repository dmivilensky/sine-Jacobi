import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.factory import ROOT
from audit.exact import records
from audit.checks import sine


@unittest.skipUnless(os.environ.get('SINEJACOBI_TEST_NATIVE')=='1','use compute.py test --native')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from programs.common import native
        cls.binary={name:native(name,compile_only=True,**({'source_file':ROOT/'tests/native/test_small.cpp'} if name=='test_small' else {})) for name in
                    ('probe','verify_reference','produce_transport','verify_transport','prepare_primal','verify_primal','verify_sine','test_small')}

    def test_runtime_and_all_entrypoints_compile(self):
        p=subprocess.run([self.binary['probe']],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('MPFR 4',p.stdout)

    def test_small_analytic_problems_and_trace_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); saved=[]
            for mode in ('summary','full'):
                out=base/mode;out.mkdir();env=os.environ.copy()
                env.update(SINEJACOBI_TRACE=str(out/'trace.txt'),SINEJACOBI_TRACE_MODE=mode)
                p=subprocess.run([self.binary['test_small'],out,ROOT/'inputs/transport.txt'],capture_output=True,text=True,env=env,timeout=30)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                saved.append((out/'flow.txt').read_bytes())
            self.assertEqual(*saved)
            self.assertLess((base/'summary/trace.txt').stat().st_size,(base/'full/trace.txt').stat().st_size)

    def test_real_sine_run_against_independent_rational_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);env=os.environ.copy()
            env.update(SINEJACOBI_TRACE=str(out/'trace.txt'),SINEJACOBI_TRACE_MODE='summary')
            p=subprocess.run([self.binary['verify_sine'],'160','1e-18',out],capture_output=True,text=True,env=env,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            values=records((out/'sine-values.txt').read_text(),'SINE_VALUES 1')
            J,result=sine((out/'sine-polynomial.txt').read_text(),values)
            self.assertEqual(J,3)
            self.assertLess(result['tail'][1],__import__('fractions').Fraction('2.358e-22'))
