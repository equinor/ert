import os, sys
def pyc_clean(dir):
    findcmd = 'find %s -name "*.pyc" -print' % dir
    count = 0
    for f in os.popen(findcmd).readlines():
        count += 1
        #print str(f[:-1])
        os.remove(str(f[:-1]))
    print "Removed %d .pyc files" % count

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    prefix = os.path.dirname(script_path)
    #print ("%s/../code" % (prefix))
    pyc_clean("%s/../code" % (prefix))