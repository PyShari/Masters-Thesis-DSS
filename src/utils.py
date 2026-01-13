import re

def safe_filename(text):
    """
    Convert a string into a filesystem-safe filename.
    Replaces all non-alphanumeric characters with underscores.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)

