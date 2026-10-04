"""Independent readers and exact consequences of native enclosure reports."""
import re
from .exact import (F, need, rational, records, pt, add, sub, scale, mag,
                    encloses, isum, poly_mul, derivative, shift, trim, rm, rexp,
                    sine_enclosure)


class Tokens:
    def __init__(self,text): self.t=iter(text.split())
    def take(self):
        value=next(self.t,None); need(value is not None,'truncated parameters'); return value
    def count(self):
        s=self.take(); need(re.fullmatch(r'0|[1-9][0-9]*',s) is not None,'invalid integer'); return int(s)
    def number(self): return rational(self.take())
    def end(self): need(next(self.t,None) is None,'trailing parameters')


def reference(text):
    t=Tokens(text); need((t.take(),t.count())==('CONTACT_REFERENCE',1),'reference header')
    beta,b,gamma=[t.number() for _ in range(3)]
    support=[t.number() for _ in range(t.count())]
    scales=[t.number() for _ in range(t.count())]; t.end()
    need(beta>0 and gamma>=0 and b>0 and len(support)%2==1 and min(support)>0,'reference positivity')
    need(len(scales)==2 and 0<scales[0]<scales[1],'two ordered scales required')
    return dict(beta=beta,b=b,gamma=gamma,support=support,scales=scales)


def polynomial(text):
    t=Tokens(text); need((t.take(),t.count(),t.count())==('TRANSPORT_POLYNOMIAL',1,2),'polynomial header')
    scales=[t.number() for _ in range(2)]; multipliers=[t.number() for _ in range(4)]
    need(0<scales[0]<scales[1],'unordered polynomial scales')
    terms=[]
    for _ in range(t.count()):
        kind=t.take(); i=t.count(); j=t.count(); theta=t.number(); d=i+j
        need((kind=='h' and d==0) or (kind=='u' and d>0),'invalid monomial')
        q=[]
        for side in range(2):
            row=[]
            for k in range(4):
                n=t.count(); need(n>0,'empty polynomial'); row.append(trim([t.number() for _ in range(n)]))
            q.append(row)
        if kind=='h':
            p=q[0][1]; raw=[shift(derivative(p),1),shift(p,1),[F(0)],[F(0)]]
        else:
            p=shift(q[0][1],1)
            for _ in range(d-1):
                need(p[0]==0,'missing zero for endpoint cancellation'); p=trim(p[1:] or [F(0)])
            raw=[shift(derivative(p),d),shift(p,d-1),shift(p,d),shift(p,d+1)]
        for side,divisor in enumerate(([0,1],[2,-1])):
            for k in range(4):
                need(poly_mul(divisor,q[side][k])==trim(raw[k]),'endpoint/derivative polynomial identity')
        terms.append(dict(kind=kind,powers=(i,j),theta=theta,polynomial=p))
    t.end(); need(bool(terms),'empty storage potential')
    return dict(scales=scales,multipliers=multipliers,terms=terms)


def primal_parameters(text):
    t=Tokens(text); need((t.take(),t.count())==('GROUND_PRIMAL',1),'primal header')
    beta=t.number(); second=t.count(); scales=[t.number() for _ in range(t.count())]
    coeff=[t.number() for _ in range(t.count())]; d=t.count()
    directions=[t.count() for _ in range(d)]; C=[[t.number() for _ in range(d)] for _ in range(d)]; t.end()
    need(beta>0 and second==1 and len(scales)==2 and d==6,'retained six-equation primal class required')
    need(0<scales[0]<scales[1] and len(set(directions))==d and max(directions)<len(coeff),'primal dimensions')
    need(all(x.denominator&(x.denominator-1)==0 for x in coeff+[v for row in C for v in row]),'primal data must be dyadic')
    return dict(beta=beta,scales=scales,second=second,coefficients=coeff,directions=directions,matrix=C)


def transport_report(text,partial=False):
    lines=text.splitlines(); magic='VERIFIED_TRANSPORT_PART 1' if partial else 'VERIFIED_TRANSPORT 2'
    claim='TRANSPORT_CLAIM_PART 1' if partial else 'TRANSPORT_CLAIM 1'
    need(bool(lines) and lines[0] in (magic,claim),'wrong transport report/claim format')
    result={}; start=1
    if partial:
        a=lines[1].split(); need(len(a)==4 and a[0]=='domain' and a[1] in ('0','1'),'missing domain')
        l,r=map(rational,a[2:]); need(0<=l<r<=1,'invalid domain')
        result['domain']=(int(a[1]),l,r); start=2
    for line in lines[start:]:
        a=line.split(); need(bool(a) and a[0] not in result,'duplicate/empty transport field')
        if a[0] in ('dilation','price','undilated_price'):
            need(len(a)==3,'bad transport interval'); val=tuple(map(rational,a[1:])); need(val[0]<=val[1],'reversed transport interval')
        elif a[0] in ('cells','leaves'):
            need(len(a)==2 and re.fullmatch(r'[1-9][0-9]*',a[1]) is not None,'bad count'); val=int(a[1])
        else: raise ValueError('unknown transport field: '+a[0])
        result[a[0]]=val
    expected={'dilation','price','undilated_price','cells','leaves'}|({'domain'} if partial else set())
    need(set(result)==expected,'incomplete transport report')
    lo,hi=result['dilation']; need(0<lo==hi<=1 and result['leaves']>=2*result['cells'],'transport dimensions/dilation')
    need(encloses(result['price'],scale(result['undilated_price'],lo)),'inconsistent dilated price')
    return result


def merge_prices(parts,merged):
    need(bool(parts),'missing regional price claims')
    previous=(0,F(0)); total=pt(0); nc=nl=0; lam=F(1); domains=[]
    for part in parts:
        side,l,r=part['domain']; need((side,l)==previous,'gap, overlap or out-of-order price region')
        previous=(side+1,F(0)) if r==1 else (side,r)
        total=add(total,part['undilated_price']); lam=min(lam,part['dilation'][0])
        nc+=part['cells']; nl+=part['leaves']; domains.append(part['domain'])
    need(previous==(2,0),'price regions do not exhaust both halves')
    need(total==merged['undilated_price'] and pt(lam)==merged['dilation'] and
         scale(total,lam)==merged['price'] and (nc,nl)==(merged['cells'],merged['leaves']), 'exact price merge mismatch')
    return domains


def contraction(p,values):
    d=len(p['directions']); C=p['matrix']
    expected={'radius','residual','contraction','central_energy','energy_ceiling'}|{f'F_{i}' for i in range(d)}|{f'J_{i}_{j}' for i in range(d) for j in range(d)}
    need(set(values)==expected,'incomplete residual/Jacobian report')
    rho=values['radius'][0]
    need(values['radius'][1]==rho>0,'positive exact radius required')
    eta=max(mag(isum(scale(values[f'F_{j}'],C[i][j]) for j in range(d))) for i in range(d))
    rows=[sum(mag(sub(pt(int(i==j)),isum(scale(values[f'J_{a}_{j}'],C[i][a]) for a in range(d)))) for j in range(d)) for i in range(d)]
    r=max(rows)
    need(0<=values['residual'][0]<=values['residual'][1] and eta<=values['residual'][1],'residual norm underreported')
    need(0<=values['contraction'][0]<=values['contraction'][1]<1 and r<=values['contraction'][1],'Jacobian norm underreported')
    need(eta+r*rho<rho,'no strict self mapping')
    need(values['residual'][1]+values['contraction'][1]*rho<rho,'reported norm bounds do not establish self mapping')
    need(values['central_energy'][0]>0,'nonpositive central energy')
    ceiling=rm(values['central_energy'],rexp(pt(d*rho)))
    need(ceiling[1]<=values['energy_ceiling'][1],'energy ceiling is too small')
    return {'residual':eta,'row_sums':rows,'contraction':r,'radius':rho,
            'selfmap_slack':rho-eta-r*rho,'energy_ceiling':ceiling}


def partition(text):
    t=Tokens(text); need((t.take(),t.count())==('PRIMAL_QUADRATURE',1),'quadrature header')
    n=t.count(); need(n>0,'empty primal partition'); previous=F(0)
    for _ in range(n):
        l=t.number(); r=t.number()
        need(l==previous and l<r<=1,'primal quadrature gap or overlap')
        need(F(float(l))==l and F(float(r))==r,'inexact binary64 partition point'); previous=r
    t.end(); need(previous==1,'incomplete primal quadrature'); return n


def sine(polynomial_text,values):
    t=Tokens(polynomial_text); need((t.take(),t.count())==('SINE_POLYNOMIAL',1),'sine header')
    J=t.count(); n=t.count(); independent=sine_enclosure(J)
    need(n==len(independent['coefficients']),'sine polynomial size')
    for i,c in enumerate(independent['coefficients']):
        need(t.count()==i and t.number()==c and t.number()==c,'integer sine coefficient mismatch')
    t.end()
    need(encloses(values['finite_sum'],independent['finite_sum']),'sine sum fails rational re-evaluation')
    need(0<independent['tail'][0]<=independent['tail'][1]<=values['tail'][1],'sine tail fails')
    need(encloses(values['energy'],independent['energy']),'sine energy fails rational re-evaluation')
    return J, independent

