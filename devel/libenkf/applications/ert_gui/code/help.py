import os
import sys

prefix = os.path.dirname(__file__) + "/../help/"

def resolveHelpLabel(label):
    """
    Reads a HTML file from the help directory.
    The HTML must follow the specification allowed by QT here: http://doc.trolltech.com/4.6/richtext-html-subset.html
    """

#    if label.strip() == "":
#        raise AssertionError("NOOOOOOOOOOOOOOOOOOOOO!!!!!!!!!!!!")

    filename = prefix + label + ".html"
    if os.path.exists(filename) and os.path.isfile(filename):
        f = open(filename, 'r')
        help = f.read()
        f.close()
        return help
    else:
        #sys.stderr.write("Missing help file: " + label + "\n")
        return ""
