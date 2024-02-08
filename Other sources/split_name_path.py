import os,sys

def f(img_path):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')
    filename = parts[len(parts)-1]
    

    return filename, parts[len(parts)-2]