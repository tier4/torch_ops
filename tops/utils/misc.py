import os
import sys
from contextlib import contextmanager


@contextmanager
def capture_stdout():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout = old_stdout
