import os
import sys

prefix = os.path.dirname(__file__) + "/../help/"

def resolveHelpLabel(label):
    """
    Reads a HTML file from the help directory.
    The HTML must follow the specification allowed by QT here: http://doc.trolltech.com/4.6/richtext-html-subset.html
    """

    # This code can be used to find widgets with empty help labels
#    if label.strip() == "":
#        raise AssertionError("NOOOOOOOOOOOOOOOOOOOOO!!!!!!!!!!!!")

    path = prefix + label + ".html"
    if os.path.exists(path) and os.path.isfile(path):
        f = open(path, 'r')
        help = f.read()
        f.close()
        return help
    else:
        # This code automatically creates empty help files
#        sys.stderr.write("Missing help file: '%s'\n" % label)
#        if not label == "" and not label.find("/") == -1:
#            sys.stderr.write("Creating help file: '%s'\n" % label)
#            directory, filename = os.path.split(path)
#
#            if not os.path.exists(directory):
#                os.makedirs(directory)
#
#            file_object = open(path, "w")
#            file_object.write(label)
#            file_object.close()
        return ""
