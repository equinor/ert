#!/bin/bash

# The only purpose of this funny little file is to serve as mock rms executable
# for the rms2013 test, which just verifies that we have removed 'python' from
# the PATH. Since we (hopefully ..) don't have Python in the path any longer the
# script must be implemented in another language.

echo $PATH > PATH
