import os


def safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
