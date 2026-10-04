"""Record the actual runtime and host without affecting numerical acceptance."""
import os
import platform
import subprocess
import sys

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def runtime(jobs):
    info = {'python': sys.version, 'platform': platform.platform(),
            'architecture': platform.machine(), 'logical_cpus': os.cpu_count(),
            'jobs_requested': jobs,
            'thread_environment': {key: os.environ.get(key) for key in THREAD_VARIABLES}}
    if platform.system() == 'Darwin':
        names = {'cpu_model': 'machdep.cpu.brand_string', 'memory_bytes': 'hw.memsize',
                 'physical_cpus': 'hw.physicalcpu', 'logical_cpus': 'hw.logicalcpu',
                 'performance_cores': 'hw.perflevel0.physicalcpu',
                 'efficiency_cores': 'hw.perflevel1.physicalcpu'}
        for key, name in names.items():
            try:
                value = subprocess.check_output(['sysctl', '-n', name], text=True,
                                                stderr=subprocess.DEVNULL, timeout=5).strip()
                info[key] = int(value) if value.isdecimal() else value
            except (OSError, subprocess.SubprocessError):
                pass
    elif hasattr(os, 'sysconf'):
        try:
            info['memory_bytes'] = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        except (ValueError, OSError):
            pass
    return info
