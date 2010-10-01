
def RH_version():
    RH  = open('/etc/redhat-release' , 'r').read().split()[6]
    return float( RH )
