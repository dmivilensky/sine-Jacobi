import copy
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from portable.cover import MAGIC, HEADER, CELL, END, Cell, Reader, encode_trees, leaves, scan, digest
from portable.verify import compare_intervals, compare_transport, manifest_check, check
from audit.exact import pt, F
from programs.proof import checker_sources, unpack, check_archive
from programs.common import json_write
from tests.factory import bundle, ROOT


def write_cover(path, half=None):
    """Small complete binary trees, with exact synthetic price intervals."""
    n, tree, upper = encode_trees([(0,'0',-.0625),(0,'1',-.03125),
                                  (1,'2',-.015625),(1,'3',-.0078125)])
    halves = range(2) if half is None else (half,)
    with Path(path).open('wb') as stream:
        stream.write(MAGIC+HEADER.pack(int(half is not None), half or 0, 0, 1))
        for h in halves: Cell(h,0,1,1,.25,0,upper,n,tree,'0 0').write(stream)
        stream.write(b'E'+END.pack(len(halves),len(halves)*n,1))


def write_fixture(root, synthetic=True):
    """Explicit fake data for control-flow tests, never a numerical witness."""
    checker_sources(root)
    b = copy.deepcopy(bundle())
    if not synthetic: b.pop('test_fixture')  # Used ONLY with mocked native gates below.
    shutil.copyfile(ROOT/'inputs/run.json', root/'run.json')
    data = root/'data'; (data/'second').mkdir(parents=True)
    cover = data/'second/state-cover.bin'
    with cover.open('wb') as f:
        f.write(MAGIC+HEADER.pack(0,0,0,1))
        count, tree, upper = encode_trees([(0,'',-.125),(1,'',-.125)])
        for half in range(2): Cell(half,0,1,1,0,0,upper,count,tree,'-0x1p-18 -0x1p-18').write(f)
        f.write(b'E'+END.pack(2,4,1))
    geometry = scan(cover); geometry['artifact'] = 'second/state-cover.bin'
    b['cover_geometry'] = geometry
    b['artifacts'] = {'second/state-cover.bin': {'sha256':digest(cover), 'bytes':cover.stat().st_size}}
    (data/'reference').mkdir(); (data/'primal_second').mkdir()
    json_write(root/'claims.json', b)
    def manifest():
        members = {str(p.relative_to(root)): {'sha256':digest(p),'bytes':p.stat().st_size}
                   for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
        json_write(root/'manifest.json', {'format':'SINEJACOBI_PROOF_3', 'members':members})
    manifest()
    return b, manifest


class CompactTests(unittest.TestCase):
    def test_export_real_archive_and_extract_without_repository(self):
        # The numerical reports are synthetic; this checks packaging only.
        from programs.proof import export
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'fixture';src.mkdir();b,_=write_fixture(src,synthetic=False)
            b['run_configuration']=json.loads((ROOT/'inputs/run.json').read_text())
            archive=p/'proof.tar.xz'
            with patch('programs.proof.collect',return_value=b):
                result=export(src/'data',archive)
            self.assertEqual(result['status'],'EXPORTED_NOT_VERIFIED')
            with unpack(archive) as packed:
                self.assertTrue((packed/'data/second/state-cover.bin').is_file())
                self.assertTrue((packed/'native/verify_transport.cpp').is_file())
                self.assertFalse((packed/'programs').exists())
                self.assertFalse(list(packed.rglob('*.tex')))
                self.assertFalse((packed/'paper').exists())
                self.assertNotIn('production_receipts',json.loads((packed/'claims.json').read_text()))

    def test_packed_geometry_and_leaf_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cover.bin';write_cover(path)
            result=scan(path)
            self.assertEqual((result['cells'],result['leaves']),(2,8))
            cells=list(Reader(path))
            self.assertEqual(list(leaves(cells[0])),[(0,'0'),(0,'1'),(1,'2'),(1,'3')])

    def test_tree_holes_overlap_and_axis_conflict_rejected(self):
        for rows in ([(0,'0',-.1),(1,'',-.1)],[(0,'',-.1),(0,'0',-.1),(1,'',-.1)],
                     [(0,'0',-.1),(0,'3',-.1),(1,'',-.1)],[(0,'0'*53,-.1),(1,'',-.1)]):
            with self.assertRaises(ValueError): encode_trees(rows)

    def test_binary_corruption_truncation_and_trailer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_cover(p/'good')
            raw=(p/'good').read_bytes(); tree=8+HEADER.size+1+CELL.size
            changes=[raw[:i] for i in (0,7,tree,len(raw)-1)] + [raw+b'junk']
            q=bytearray(raw);q[tree]|=3; changes.append(bytes(q))
            q=bytearray(raw);q[tree+1]|=128; changes.append(bytes(q))
            q=bytearray(raw);q[8+HEADER.size+1+1+5*8:8+HEADER.size+1+1+6*8]=struct.pack('<d',float('nan'));changes.append(bytes(q))
            for bad in changes:
                (p/'bad').write_bytes(bad)
                with self.assertRaises((ValueError,UnicodeError)): scan(p/'bad')

    def test_region_merge_and_schedule_from_binary(self):
        from programs.transport import merge_price_covers, verification_regions
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);regions=[]
            for half in range(2):
                d=p/str(half);d.mkdir();write_cover(d/'state-cover.bin',half)
                regions.append((half,0.,1.,d))
            merge_price_covers(regions,p/'state-cover.bin')
            self.assertEqual(scan(p/'state-cover.bin')['leaves'],8)
            self.assertEqual(len(verification_regions(p,8)),2)
            with self.assertRaises(ValueError): merge_price_covers(regions[::-1],p/'wrong.bin')

    def test_replay_intervals_cannot_be_overstated_or_drop_fields(self):
        self.assertEqual(compare_intervals('X\na 0 2\n','X\na 1 1\n','X','toy'),1)
        with self.assertRaises(ValueError): compare_intervals('X\na 1 1\n','X\na 0 2\n','X','toy')
        with self.assertRaises(ValueError): compare_intervals('X\na 0 2\nb 0 2\n','X\na 1 1\n','X','toy')
        compare_intervals('X\na 2 2\n','X\na 1 1\n','X','toy',('a',))

    def test_fixture_cannot_receive_full_pass(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as w:
            p=Path(tmp);write_fixture(p)
            with self.assertRaisesRegex(ValueError,'synthetic'):
                check(p,Path(w),1)

    def test_manifest_detects_missing_and_changed_witnesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p)
            manifest_check(p)
            target=p/'data/second/state-cover.bin';target.write_bytes(target.read_bytes()+b'X')
            with self.assertRaisesRegex(ValueError,'digest mismatch'): manifest_check(p)
            target.unlink()
            with self.assertRaisesRegex(ValueError,'missing'): manifest_check(p)

    def test_isolated_launcher_rejects_fixture_and_replaces_stale_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p);report=p/'verification.json';report.write_text('{"status":"OLD_PASS"}')
            run=subprocess.run([sys.executable,'-I','-B',p/'verify.py','--output',report],
                               cwd=p,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(json.loads(report.read_text())['status'],'FAIL')
            self.assertFalse((p/'programs').exists())

    def test_bad_archive_clears_old_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'bad.xz').write_bytes(b'bad');out=p/'report.json';out.write_text('{"status":"OLD_PASS"}')
            self.assertEqual(check_archive(p/'bad.xz',out),1)
            self.assertEqual(json.loads(out.read_text())['status'],'FAIL')

    def test_mocked_full_orchestration_requires_every_gate(self):
        # A control-flow test with fabricated reports. No numerical conclusion.
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as w:
            p=Path(tmp);b,_=write_fixture(p,synthetic=False);work=Path(w)
            def fake_process(args,work,name):
                if name=='reference':
                    (Path(args[-2])/'reference-values.txt').write_text(b['files']['reference/reference-values.txt']['text'])
                elif name=='primal': Path(args[-1]).write_text(b['files']['primal_second/values.txt']['text'])
                elif name=='sine':
                    out=Path(args[-1])
                    for n in ('sine-values.txt','sine-polynomial.txt'): (out/n).write_text(b['files']['sine/'+n]['text'])
                elif name.startswith('transport-'):
                    key=f"second/price-claims/{int(name.split('-')[1]):04d}/values.txt"
                    Path(args[9]).write_text(b['files'][key]['text'].replace('TRANSPORT_CLAIM_PART 1','VERIFIED_TRANSPORT_PART 1'))
                return {'seconds':0,'mocked':True}
            entries={k:k for k in ('verify_reference','verify_primal','verify_sine','verify_transport')}
            with patch('portable.verify.native_build',return_value=(entries,{'mocked':True})), \
                 patch('portable.verify.process',side_effect=fake_process):
                result=check(p,work,2)
            self.assertEqual(result['status'],'PASS_FULL_NUMERICS')
            self.assertEqual(len(result['native_timings']['transport_regions']),2)


@unittest.skipUnless(os.environ.get('SINEJACOBI_TEST_NATIVE')=='1','use compute.py test --native')
class PackedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from programs.common import native
        cls.binary=native('test_packed',compile_only=True,source_file=ROOT/'tests/native/test_packed.cpp')
        cls.reference=native('verify_reference',compile_only=True)
        cls.produce=native('produce_transport',compile_only=True)
        cls.transport=native('verify_transport',compile_only=True)

    def run_native(self,mode,path):
        env=os.environ.copy();env.update(SINEJACOBI_TRACE=str(path.parent/'trace'),SINEJACOBI_TRACE_MODE='summary')
        return subprocess.run([self.binary,mode,path],env=env,capture_output=True,text=True,timeout=30)

    def test_cpp_writer_python_reader_and_cpp_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'cover.bin';r=self.run_native('write',out)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(scan(out)['leaves'],8)
            decoded=self.run_native('decode',out);self.assertEqual(decoded.returncode,0,decoded.stderr)
            self.assertIn('TRANSPORT_COVER 2',decoded.stdout)
            reader=Reader(out);cells=list(reader)
            with (p/'again.bin').open('wb') as stream:
                stream.write(MAGIC+HEADER.pack(0,0,0,1))
                for cell in cells: cell.write(stream)
                stream.write(b'E'+END.pack(reader.cells,reader.count,reader.dilation))
            self.assertEqual(out.read_bytes(),(p/'again.bin').read_bytes())

    def test_cpp_rejects_corrupt_tree_and_extra_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'cover.bin';self.run_native('write',out);raw=out.read_bytes()
            pos=8+HEADER.size+1+CELL.size
            q=bytearray(raw);q[pos]|=3
            for bad in (bytes(q),raw[:-1],raw+b'junk'):
                out.write_bytes(bad);self.assertNotEqual(self.run_native('decode',out).returncode,0)

    def test_saved_ode_replay_rejects_modified_taylor_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'flow.txt';run=self.run_native('flow-write',p)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(self.run_native('flow-replay',p).returncode,0)
            lines=p.read_text().splitlines();lines[8]='0x1p+0 0x1p+0';p.write_text('\n'.join(lines)+'\n')
            run=self.run_native('flow-replay',p);self.assertNotEqual(run.returncode,0)
            self.assertIn('incoming initial interval',run.stderr)

    def test_real_state_bounds_reject_false_margin_after_binary_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cover.bin';self.assertEqual(self.run_native('write',p).returncode,0)
            run=self.run_native('state-replay',p);self.assertEqual(run.returncode,0,run.stderr)
            raw=bytearray(p.read_bytes());pos=8+HEADER.size+1+1+5*8
            raw[pos:pos+8]=struct.pack('<d',-1.);p.write_bytes(raw)
            run=self.run_native('state-replay',p);self.assertNotEqual(run.returncode,0)
            self.assertIn('state bound exceeds witness',run.stderr)

    def test_small_reference_and_real_regional_producer_to_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'reference.txt').write_text('CONTACT_REFERENCE 1\n1\n1\n0\n1\n1\n2\n1\n2\n')
            env=os.environ.copy();env.update(SINEJACOBI_TRACE=str(p/'trace'),SINEJACOBI_TRACE_MODE='summary')
            args=[self.reference,p/'reference.txt','128','8','1e-8','1e-6',p/'first']
            first=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(first.returncode,0,first.stderr)
            args[-1]=p/'replay';args.append(p/'first/support-filter-flow.txt')
            second=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(second.returncode,0,second.stderr)
            count=compare_intervals((p/'first/reference-values.txt').read_text(),
                                    (p/'replay/reference-values.txt').read_text(),'CONTACT_VALUES 1','reference')
            self.assertEqual(count,12)
            # Real producer on one short interval; dummy exterior cells are
            # deliberately OUTSIDE the reported verification domain.
            (p/'transport.txt').write_text('TRANSPORT_POLYNOMIAL 1\n2\n1\n2\n0\n0\n0\n0\n0\n')
            args=[self.produce,p/'reference.txt',p/'transport.txt',p/'first','128','8','1e-8','0.01',
                  '1e-6','16',p/'region','0','0x1p-3','0x1.8p-3','segment']
            produced=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(produced.returncode,0,produced.stderr)
            cells=list(Reader(p/'region/state-cover.bin'))
            self.assertEqual(len(cells),1)
            n,tree,upper=encode_trees([(0,'',-.125),(1,'',-.125)])
            full=p/'full.bin'
            with full.open('wb') as stream:
                stream.write(MAGIC+HEADER.pack(0,0,0,1))
                Cell(0,0,.125,1,0,0,upper,n,tree,'0 0').write(stream)
                cells[0].write(stream)
                Cell(0,.1875,1,1,0,0,upper,n,tree,'0 0').write(stream)
                Cell(1,0,1,1,0,0,upper,n,tree,'0 0').write(stream)
                stream.write(b'E'+END.pack(4,6+cells[0].count,1))
            report=p/'verified-region.txt'
            args=[self.transport,p/'reference.txt',p/'transport.txt',p/'first','128','8','1e-8','1e-6',
                  full,report,'0','0x1p-3','0x1.8p-3']
            accepted=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(accepted.returncode,0,accepted.stderr)
            from audit.checks import transport_report
            values=transport_report(report.read_text(),True)
            self.assertEqual(values['domain'],(0,F(1,8),F(3,16)))
            self.assertEqual(values['leaves'],cells[0].count)
