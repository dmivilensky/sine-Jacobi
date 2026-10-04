"""Derive numerical consequences from exact inputs and interval reports."""
import hashlib
from pathlib import PurePosixPath
import re
from .exact import F, need, rational, records, pt, add, sub, scale, encloses, mag, isum, rexp
from . import checks


def texts(bundle):
    need(bundle.get('format')=='SINEJACOBI_CERTIFICATES_2','unknown compact certificate format')
    result={}
    for name,item in bundle['files'].items():
        p=PurePosixPath(name)
        need(not p.is_absolute() and '..' not in p.parts and str(p)==name,'unsafe embedded path')
        need(set(item)=={'text','sha256'} and isinstance(item['text'],str),'malformed embedded report')
        need(hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256'],'embedded digest mismatch: '+name)
        result[name]=item['text']
    return result


def verify(bundle):
    files=texts(bundle)
    ref=checks.reference(files['reference/reference.txt'])
    poly=checks.polynomial(files['second/transport.txt'])
    primal=checks.primal_parameters(files['primal_second/profile.txt'])
    need(ref['scales']==poly['scales']==primal['scales'] and ref['beta']==primal['beta'],'incompatible mathematical classes')
    parts=[checks.transport_report(files[n],True) for n in sorted(files)
           if re.fullmatch(r'second/price-claims/[^/]+/values.txt',n)]
    transport=checks.transport_report(files['second/price-claim.txt'])
    domains=checks.merge_prices(parts,transport)
    rv=records(files['reference/reference-values.txt'],'CONTACT_VALUES 1')
    need(set(rv)=={'k','J','base','mass_1','mass_2','mass_3','H_0','H_signal_0','H_1','H_signal_1','S','S_signal'},'reference values schema')
    pv=records(files['primal_second/values.txt'],'PRIMAL_VALUES 1')
    root=checks.contraction(primal,pv)
    radius=files['primal_second/radius.txt'].split()
    need(len(radius)==2 and tuple(map(rational,radius))==pv['radius'],'radius witness mismatch')
    partition_cells=checks.partition(files['primal_second/partition.txt'])
    sv=records(files['sine/sine-values.txt'],'SINE_VALUES 1')
    need(set(sv)=={'energy','finite_sum','tail','gain_upper'},'sine values schema')
    J,sine=checks.sine(files['sine/sine-polynomial.txt'],sv)
    # k is separately checked against a rational enclosure of pi.
    from .exact import rm
    need(encloses(rv['k'],scale(rm(sine['pi'],sine['pi']),F(1,4))),'ground normalization k mismatch')
    xi0,xi1,tau,chi=poly['multipliers']; lam=transport['dilation'][0]
    terms={'determinant_0':scale(rv['H_signal_0'],xi0),
           'determinant_1':scale(rv['H_signal_1'],xi1),'second_trace':scale(rv['S_signal'],chi),
           'mass':scale(sub(pt(1),rv['mass_2']),-tau)}
    bound=sub(add(rv['base'],scale(isum(terms.values()),lam)),transport['price'])
    # Derive achieved bounds; there are no publication-specific target digits.
    need(bound[0]>0 and rv['base'][0]>0,'positive bounds required for energy ratios')
    need(bound[0]<=root['energy_ceiling'][1],'lower and feasible upper bounds contradict each other')
    from .exact import div, sqrt_interval
    ratios={'gain_ratio':sqrt_interval(div(sine['energy'],bound)),
            'baseline_gain_ratio':sqrt_interval(div(sine['energy'],rv['base']))}
    gaps={'reference_to_sine':sub(sine['energy'],rv['base']),
          'feasible_improvement_lower':sine['energy'][0]-root['energy_ceiling'][1],
          'primal_dual_gap_upper':root['energy_ceiling'][1]-bound[0]}
    geometry=bundle.get('cover_geometry')
    if geometry:
        need(geometry.get('format')=='SINEJACOBI_COVER_GEOMETRY_1','cover summary format')
        need((geometry['cells'],geometry['leaves'])==(transport['cells'],transport['leaves']),'cover summary counts')
        need(rational(geometry['dilation'])==lam and rational(geometry['max_recorded_upper'])<0,'cover summary signs/dilation')
        need(geometry['sha256']==bundle['artifacts'][geometry.get('artifact','second/state-cover.bin')]['sha256'],'cover identity mismatch')
        s=tuple(map(rational,geometry['undilated_price_sum'])); error=rational(geometry['summation_allowance'])
        need(error>=0 and encloses((s[0]-error,s[1]+error),transport['undilated_price']),'recorded cell prices do not contain the regional price claim')
    result={'status':'PASS_EXACT_CONSEQUENCES_TEST_FIXTURE' if bundle.get('test_fixture') else 'PASS_EXACT_CONSEQUENCES','scope':
            'Native ODE, quadrature, Jacobian and state bounds are assumed valid saved enclosures. '
            'The sine series and all consequences below are re-evaluated with rational arithmetic.',
            'bound':bound,'price':transport['price'],'terms':terms,'primal':root,
            'sine':{k:v for k,v in sine.items() if k!='coefficients'},'price_domains':domains,
            'partition_cells':partition_cells,'cover_geometry_available':geometry is not None,
            'cover_geometry':geometry,'counts':{'cells':transport['cells'],'leaves':transport['leaves']},
            'ratios':ratios,'gaps':gaps,
            'strict_feasible_improvement':gaps['feasible_improvement_lower']>0}
    rows={'reference':rv,'certificate':{'bound':bound,'price':transport['price']},'feasible':pv,'sine':sv}
    return result,rows,{'reference':ref,'polynomial':poly,'primal':primal,'J':J}
