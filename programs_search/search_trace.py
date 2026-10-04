"""Optional full diagnostic trace of numerical proposal evaluations.

Disabled by default; enable SINEJACOBI_SEARCH_TRACE_MODE=full for diagnostics.

Inputs are serialized before mutation. Arrays contain their full values;
model snapshots contain mathematical parameters, never process addresses.
Each call closes its stream so an I/O failure propagates to the stage."""
from functools import wraps
import json
import os
from pathlib import Path


def plain(value):
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,(list,tuple)): return [plain(x) for x in value]
    if isinstance(value,dict):
        return {str(k):plain(v) for k,v in value.items() if not callable(v)}
    if hasattr(value,'tolist'): return plain(value.tolist())
    # Mathematical parameters of model instances; never object addresses.
    return {key:plain(getattr(value,key)) for key in
            ('beta','b','gamma','coefficients','scales','degree','second','tolerance','target')
            if hasattr(value,key)}


def traced(function):
    @wraps(function)
    def evaluate(*args,**kwargs):
        path=os.environ.get('SINEJACOBI_SEARCH_TRACE')
        if not path or os.environ.get('SINEJACOBI_SEARCH_TRACE_MODE', 'off') != 'full':
            return function(*args,**kwargs)
        # Save the input before functions mutate a model's multipliers.
        inputs=plain(args),plain(kwargs)
        try: value=function(*args,**kwargs)
        except Exception as exc:
            with Path(path).open('a') as stream:
                stream.write(json.dumps({'function':function.__qualname__,'arguments':inputs,'error':str(exc)})+'\n')
            raise
        with Path(path).open('a') as stream:
            stream.write(json.dumps({'function':function.__qualname__,'arguments':inputs,'value':plain(value)})+'\n')
        return value
    return evaluate
