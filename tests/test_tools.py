import os


def save_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass