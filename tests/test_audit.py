import copy
import unittest
from audit.exact import F, rational, load_json, sub, scale, pt
from audit.checks import contraction, transport_report, merge_prices, polynomial, primal_parameters
from audit.verify import verify
from tests.factory import bundle, embed, ROOT


class ExactAuditTests(unittest.TestCase):
    def test_exact_literals_and_negative_scale(self):
        self.assertEqual(rational('-0x1.8p-2'),F(-3,8))
        self.assertEqual(scale((F(1),F(2)),F(-3)),(F(-6),F(-3)))
        with self.assertRaises(ValueError): rational(float('nan'))
        with self.assertRaises(ValueError): load_json('{"a":1,"a":2}')

    def test_complete_portable_mock_contract(self):
        report,rows,p=verify(bundle())
        self.assertGreater(report['ratios']['gain_ratio'][0],0)
        self.assertEqual(report['primal']['contraction'],0)
        self.assertEqual(p['J'],3)

    def test_missing_report_fails(self):
        b=copy.deepcopy(bundle()); del b['files']['second/price-claims/0001/values.txt']
        with self.assertRaisesRegex(ValueError,'exhaust'): verify(b)

    def test_changed_bytes_fail_before_arithmetic(self):
        b=copy.deepcopy(bundle()); b['files']['second/price-claim.txt']['text']+='extra\n'
        with self.assertRaisesRegex(ValueError,'digest mismatch'): verify(b)

    def test_dropped_jacobian_entry_fails(self):
        b=copy.deepcopy(bundle()); n='primal_second/values.txt'
        text=b['files'][n]['text']; text='\n'.join(line for line in text.splitlines() if not line.startswith('J 5 5 '))+'\n'
        b['files'][n]=embed(text)
        with self.assertRaisesRegex(ValueError,'incomplete residual'): verify(b)

    def test_unit_norm_summary_cannot_replace_matrix(self):
        b=copy.deepcopy(bundle()); n='primal_second/values.txt'
        lines=b['files'][n]['text'].splitlines()
        lines=['J 0 0 0x2p+0 0x2p+0' if l.startswith('J 0 0 ') else l for l in lines]
        b['files'][n]=embed('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ValueError,'norm underreported'): verify(b)

    def test_exact_sqrt_outward_rounding(self):
        from audit.exact import sqrt_interval
        for a in ((F(0),F(0)),(F(2),F(3)),(F(4),F(9)),(F(1,9),F(7,11))):
            lo,hi=sqrt_interval(a)
            self.assertLessEqual(lo*lo,a[0]); self.assertGreaterEqual(hi*hi,a[1])
        self.assertEqual(sqrt_interval((F(4),F(9))),(F(2),F(3)))
        with self.assertRaises(ValueError): sqrt_interval((F(-1),F(2)))

    def test_exact_problem_inputs_remain_consistent(self):
        q=primal_parameters((ROOT/'inputs/profile.txt').read_text())
        v=polynomial((ROOT/'inputs/transport.txt').read_text())
        self.assertEqual(q['scales'],v['scales'])
        self.assertEqual(len(v['terms']),6)

    def test_general_homothety_and_partition(self):
        def part(side,l,r,lam):
            return {'domain':(side,F(l),F(r)),'dilation':pt(F(lam)),
                    'undilated_price':pt(2),'price':pt(2*F(lam)),'cells':1,'leaves':2}
        parts=[part(0,0,1,'1/2'),part(1,0,1,'3/4')]
        merged={'dilation':pt(F(1,2)),'undilated_price':pt(4),'price':pt(2),'cells':2,'leaves':4}
        self.assertEqual(len(merge_prices(parts,merged)),2)
        parts[1]['domain']=(1,F(1,2),F(1))
        with self.assertRaisesRegex(ValueError,'gap'): merge_prices(parts,merged)

    def test_synthetic_fixture_does_not_certify_native_provenance(self):
        report,_,_=verify(bundle())
        self.assertIn('assumed valid saved enclosures',report['scope'])
        self.assertFalse(report['cover_geometry_available'])
