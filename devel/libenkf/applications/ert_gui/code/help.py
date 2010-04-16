import os

prefix = os.path.dirname(__file__) + "/../help/"

def resolveHelpLabel(label):
    """
    Reads a HTML file from the help directory.
    The HTML must follow the specification allowed by QT here: http://doc.trolltech.com/4.6/richtext-html-subset.html
    """

    filename = prefix + label + ".html"
    if os.path.exists(filename) and os.path.isfile(filename):
        f = open(filename, 'r')
        help = f.read()
        f.closed
        return help
    else:
        return ""
