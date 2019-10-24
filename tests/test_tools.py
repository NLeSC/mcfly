import os

# For python 2.7 (which doesn't have a FileNotFoundError)
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError


def safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass