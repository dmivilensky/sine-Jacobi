"""Streaming, exact binary64 cover format; two bits per tree node.

No recorded state bounds are trusted. The native reader reevaluates every leaf.
One common negative upper bound per spatial cell replaces per-leaf bounds.
All Taylor/ODE witnesses are retained separately, without lossy conversion.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
from pathlib import Path
import struct
from audit.exact import need, rational, pt, add, scale, mag

MAGIC = b'SJCV1\r\n\x00'
HEADER = struct.Struct('<BBdd')
CELL = struct.Struct('<B6dQI')
END = struct.Struct('<QQd')
MAX_LEAVES = 1 << 24


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def take(f, n):
    v = f.read(n)
    need(len(v) == n, 'truncated packed cover')
    return v


@dataclass
class Cell:
    half: int
    left: float
    right: float
    dilation: float
    intercept: float
    slope: float
    upper: float
    count: int
    tree: bytes
    integral: str

    def write(self, f):
        integral = self.integral.encode('ascii')
        f.write(b'C')
        f.write(CELL.pack(self.half, self.left, self.right, self.dilation,
                          self.intercept, self.slope, self.upper, self.count, len(integral)))
        f.write(self.tree)
        f.write(integral)


def leaves(cell):
    """Validate the complete pair of trees while yielding branch/path leaves."""
    node_count = 2 * cell.count - 2
    need(len(cell.tree) == (node_count + 3) // 4, 'packed tree length')
    pending = [(1, '', 0, 0), (0, '', 0, 0)]
    visited = count = 0
    while pending:
        need(visited < node_count, 'tree ends before both branches are complete')
        branch, word, du, dv = pending.pop()
        tag = (cell.tree[visited // 4] >> (2 * (visited % 4))) & 3
        visited += 1
        need(tag != 3, 'reserved tree tag')
        if tag == 0:
            count += 1
            yield branch, word
        else:
            du += tag == 1
            dv += tag == 2
            need(du <= 52 and dv <= 52, 'state coordinate depth exceeds exact contract')
            a = 2 * (tag - 1)
            pending.append((branch, word + str(a + 1), du, dv))
            pending.append((branch, word + str(a), du, dv))
    need(visited == node_count and count == cell.count, 'tree count or trailing nodes')
    used = node_count % 4
    need(not used or cell.tree[-1] >> (2 * used) == 0, 'nonzero tree padding')


def encode_trees(rows):
    trees = [{}, {}]
    count = 0
    worst = -math.inf
    for branch, word, upper in rows:
        need(branch in (0, 1) and math.isfinite(upper) and upper < 0, 'invalid state leaf')
        need(all(c in '0123' for c in word), 'invalid state path')
        need(sum(c in '01' for c in word) <= 52 and sum(c in '23' for c in word) <= 52,
             'state coordinate depth exceeds exact contract')
        node = trees[branch]
        for c in word:
            need('leaf' not in node, 'overlapping state leaf')
            axis, child = divmod(int(c), 2)
            need(node.get('axis', axis) == axis, 'inconsistent split axis')
            node['axis'] = axis
            node = node.setdefault(child, {})
        need(not node, 'duplicate or overlapping leaf')
        node['leaf'] = True
        worst = max(worst, upper)
        count += 1
        need(count <= MAX_LEAVES, 'too many leaves in one spatial cell')
    tags = []
    pending = list(reversed(trees))
    while pending:
        node = pending.pop()
        if 'leaf' in node:
            need(len(node) == 1, 'leaf with descendants')
            tags.append(0)
        else:
            need(set(node) == {'axis', 0, 1}, 'hole in conditional state cover')
            tags.append(node['axis'] + 1)
            pending.extend((node[1], node[0]))
    need(len(tags) == 2 * count - 2, 'tree node count')
    packed = bytearray((len(tags) + 3) // 4)
    for i, tag in enumerate(tags):
        packed[i // 4] |= tag << (2 * (i % 4))
    return count, bytes(packed), worst


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.complete = False

    def __iter__(self):
        need(not self.complete, 'cover reader already exhausted')
        with self.path.open('rb') as f:
            need(take(f, 8) == MAGIC, 'packed cover magic')
            self.partial, self.half, self.begin, self.end = HEADER.unpack(take(f, HEADER.size))
            need(self.partial in (0, 1) and self.half in (0, 1) and
                 0 <= self.begin < self.end <= 1, 'packed cover domain')
            need(self.partial or (self.half, self.begin, self.end) == (0, 0., 1.), 'full cover domain')
            half, left = self.half, self.begin
            cells = count = 0
            minimum = 1.
            while True:
                tag = take(f, 1)
                if tag == b'E':
                    self.cells, self.count, self.dilation = END.unpack(take(f, END.size))
                    need((self.cells, self.count) == (cells, count) and cells > 0, 'cover completion counts')
                    need(0 < self.dilation <= minimum, 'invalid final homothety')
                    need((half, left) == ((self.half, self.end) if self.partial else (2, 0.)),
                         'incomplete spatial cover')
                    need(f.read(1) == b'', 'trailing packed cover data')
                    self.complete = True
                    return
                need(tag == b'C', 'invalid packed cover record')
                h, l, r, dil, intercept, slope, upper, n, length = CELL.unpack(take(f, CELL.size))
                need(all(math.isfinite(x) for x in (l, r, dil, intercept, slope, upper)), 'nonfinite cover value')
                need(h == half and l == left and l < r <= (self.end if self.partial else 1.) and h < 2,
                     'spatial cells do not form an ordered complete partition')
                need(0 < dil <= 1 and upper < 0 and 2 <= n <= MAX_LEAVES, 'invalid cell bound, count or scale')
                need(0 < length <= 4096, 'invalid price interval length')
                tree = take(f, (2 * n + 1) // 4)
                integral = take(f, length).decode('ascii')
                a = integral.split()
                need(len(a) == 2 and rational(a[0]) <= rational(a[1]), 'invalid cell integral')
                yield Cell(h, l, r, dil, intercept, slope, upper, n, tree, integral)
                minimum = min(minimum, dil)
                cells += 1
                count += n
                left = r
                if not self.partial and r == 1:
                    half += 1
                    left = 0.


def scan(path, bits=128):
    need(isinstance(bits, int) and bits >= 64, 'invalid summation precision')
    reader = Reader(path)
    total, absolute, worst = pt(0), F(0), None
    for cell in reader:
        for _ in leaves(cell): pass
        value = scale(tuple(map(rational, cell.integral.split())), 1 / F(cell.dilation))
        total = add(total, value)
        absolute += mag(value)
        u = F(cell.upper)
        worst = u if worst is None else max(worst, u)
    need(not reader.partial, 'a full cover is required')
    ops = 2 * reader.cells + 16
    unit = F(1, 2 ** (bits - 1))
    need(ops * unit < 1, 'roundoff budget exhausted')
    return {'format': 'SINEJACOBI_COVER_GEOMETRY_1', 'cells': reader.cells, 'leaves': reader.count,
            'max_recorded_upper': str(worst), 'dilation': str(F(reader.dilation)),
            'undilated_price_sum': list(map(str, total)),
            'summation_allowance': str(ops * unit / (1 - ops * unit) * absolute),
            'sha256': digest(path), 'arithmetic_bits': bits,
            'scope': 'Packed geometry; all numerical inequalities require native replay.'}
