import os, sys
def pyc_clean(dir):
    findcmd = 'find %s -name "*.pyc" -print' % dir
    count = 0
    for f in os.popen(findcmd).readlines():
        count += 1

        # try / except here in case user does not have permission to remove old .pyc files
        try:                 
            os.remove(str(f[:-1]))
        except:
            pass

    print "Removed %d .pyc files" % count

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    prefix = os.path.dirname(script_path)
    pyc_clean("%s/../code" % (prefix))
