#!/usr/bin/env python

import sys

if __name__ == '__main__':
    if int(sys.argv[1]) % int(sys.argv[2]) == 0:
        sys.exit(1)
    else:
        sys.exit(0)
