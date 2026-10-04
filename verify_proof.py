#!/usr/bin/env python3
"""Checker launcher copied as verify.py into a self-contained proof archive."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable.verify import main

if __name__ == '__main__':
    sys.exit(main())
