import os

prefix = os.path.dirname(__file__) + "/../help/"

def resolveHelpLabel(label):
    filename = prefix + label + ".html"
    if os.path.exists(filename) and os.path.isfile(filename):
        f = open(filename, 'r')
        help = f.read()
        f.closed
        return help
    else:
        return ""
