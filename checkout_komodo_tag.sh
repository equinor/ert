#!/bin/bash
source /project/res/komodo/$KOMODO_VERSION/enable

GIT=/prog/sdpsoft/bin/git

# find and check out the code that was used to build libres for this komodo relase
echo "checkout tag from komodo"
EV=$(cat /project/res/komodo/$KOMODO_VERSION/$KOMODO_RELEASE | grep "libres:" -A2 | grep "version:")
EV=($EV)    # split the string "version: vX.X.X"
EV=${EV[1]} # extract the version
echo "Using libres version $EV"
$GIT checkout $EV
