"""Document-independent exact values and outward decimal presentation."""
from fractions import Fraction


def decimal_endpoint(x, upper=False, digits=24):
    x=Fraction(x); q=10**digits
    n=-((-x.numerator*q)//x.denominator) if upper else (x.numerator*q)//x.denominator
    sign='-' if n<0 else ''; n=abs(n)
    return sign+str(n//q)+'.'+str(n%q).zfill(digits)


def interval(a):
    return {'exact':[str(x) for x in a],
            'decimal_outward':[decimal_endpoint(a[0]),decimal_endpoint(a[1],True)]}


def exact_tree(value):
    if isinstance(value,Fraction): return str(value)
    if isinstance(value,dict): return {str(k):exact_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [exact_tree(v) for v in value]
    return value


def numerical_results(report, rows, parameters):
    derived={k:interval(v) for k,v in report['ratios'].items()}
    derived['reference_to_sine']=interval(report['gaps']['reference_to_sine'])
    for k in ('feasible_improvement_lower','primal_dual_gap_upper'):
        v=report['gaps'][k]; derived[k]=interval((v,v))
    return {'format':'SINEJACOBI_NUMERICAL_RESULTS_1',
            'intervals':{g:{k:interval(v) for k,v in entries.items()} for g,entries in rows.items()},
            'derived':derived,'parameters_exact':exact_tree(parameters),
            'contraction_exact':exact_tree(report['primal']),
            'bound_terms':{k:interval(v) for k,v in report['terms'].items()},
            'sine_recomputed':{k:interval(v) for k,v in report['sine'].items()},
            'counts':report['counts'],'cover_geometry':report['cover_geometry'],
            'strict_feasible_improvement':report['strict_feasible_improvement']}

