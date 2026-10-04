"""Synthetic reports for interface tests. NOT numerical evidence for a production calculation."""
from pathlib import Path
import hashlib
from functools import lru_cache
from audit.exact import F, pt, rd, rm, rexp, scale, sine_enclosure

def hex_exact(value):
    """Exact rational token; fixtures need no producer serializer."""
    return str(F(value))

ROOT=Path(__file__).resolve().parents[1]


def encoded(magic,rows):
    return magic+'\n'+''.join(k.replace('_',' ')+' '+' '.join(map(hex_exact,rd(v)))+'\n'
                              if k.startswith(('F_','J_')) else k+' '+' '.join(map(hex_exact,rd(v)))+'\n'
                              for k,v in rows.items())


def embed(text): return {'text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}


@lru_cache(maxsize=1)
def bundle():
    s=sine_enclosure(); rv={k:pt(0) for k in ('J','H_0','H_1','H_signal_0','H_signal_1','S_signal')}
    rv.update({'k':scale(rm(s['pi'],s['pi']),F(1,4)),'base':rd(pt(F('1.14268'))),
               'mass_1':pt(1),'mass_2':pt(1),'mass_3':pt(1),'S':pt(1)})
    rho=F(1,2**60); E=rd(pt(F('1.1426920')))
    pv={'radius':pt(rho),'residual':pt(0),'contraction':pt(0),'central_energy':E,
        'energy_ceiling':rm(E,rexp(pt(6*rho)))}
    for i in range(6):
        pv[f'F_{i}']=pt(0)
        for j in range(6): pv[f'J_{i}_{j}']=pt(int(i==j))
    # Use an identity preconditioner and a test profile, not the production preconditioner.
    profile='GROUND_PRIMAL 1\n2704.261804447028\n1\n2\n32\n64\n8\n'+'0\n'*8
    profile+='6\n0\n1\n2\n3\n6\n7\n'+'\n'.join(str(int(i==j)) for i in range(6) for j in range(6))+'\n'
    partition='PRIMAL_QUADRATURE 1\n1\n0x0p+0 0x1p+0\n'
    price=pt(-F(1,2**17))
    def transport(part=None):
        p=price if part is None else scale(price,F(1,2))
        return ('TRANSPORT_CLAIM 1\n' if part is None else f'TRANSPORT_CLAIM_PART 1\ndomain {part} 0x0p+0 0x1p+0\n')+\
               'dilation 0x1p+0 0x1p+0\nprice '+' '.join(map(hex_exact,p))+'\nundilated_price '+\
               ' '.join(map(hex_exact,p))+f'\ncells {2 if part is None else 1}\nleaves {4 if part is None else 2}\n'
    sv={k:s[k] for k in ('finite_sum','energy')}; sv['tail']=pt(s['tail'][1]); sv['gain_upper']=(F(0),F(3))
    sinepoly='SINE_POLYNOMIAL 1\n3 127\n'+''.join(f'{i} {hex_exact(v)} {hex_exact(v)}\n' for i,v in enumerate(s['coefficients']))
    files={'reference/reference.txt':(ROOT/'inputs/reference.txt').read_text(),
        'reference/reference-values.txt':encoded('CONTACT_VALUES 1',rv),
        'second/transport.txt':(ROOT/'inputs/transport.txt').read_text(),
        'second/price-claim.txt':transport(),
        'primal_second/profile.txt':profile,
        'primal_second/partition.txt':partition,
        'primal_second/radius.txt':hex_exact(rho)+' '+hex_exact(rho)+'\n',
        'primal_second/values.txt':encoded('PRIMAL_VALUES 1',pv),
        'sine/sine-values.txt':encoded('SINE_VALUES 1',sv),'sine/sine-polynomial.txt':sinepoly}
    for h in range(2): files[f'second/price-claims/{h:04d}/values.txt']=transport(h)
    return {'format':'SINEJACOBI_CERTIFICATES_2','test_fixture':True,'files':{k:embed(v) for k,v in files.items()},
            'artifacts':{},'cover_geometry':None,'scope':'Synthetic interface test, not a native certificate.'}
