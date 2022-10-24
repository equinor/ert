#!/bin/sh
#
# This scripts can be used as a caching proxy for the executable `qstat`
# assumed to be present in $PATH.
#
# qstat output (the cache) will be stored in a file on /tmp
#
# Assumptions:
#  * Failures in qstat (or this proxy) are handled upstream by retrying.
#  * This script is not re-entrant. When the proxyfile is not created, but
#    it is locked, it will exit with a nonzero exit code.
#  * This script is significantly faster (when cache is fresh) than the
#    real qstat executable.
#  * This script will be called simultaneously up to thousands of times pr
#    second.
#
# Usage:
#  qstat_proxy.sh [options] [job_id] [proxyfile]
#
# ERTs torque driver will always call qstat with (any) options (starting
# with a '-'-character) and then the job_id

# Cache age that will trigger a cache update, in seconds:
CACHE_TIMEOUT=2

# A cache file older than this should not be used and will give an error
# (if the backend is not available)
CACHE_INVALID=10

QSTAT=`which qstat 2>/dev/null`

QSTAT_OPTIONS=""
# while the first arg starts with a - character:
while [ "$1" != "${1#-}" ]; do
    QSTAT_OPTIONS="$QSTAT_OPTIONS $1"
    shift
done

if [ -z $QSTAT ]; then
    # clib/old_tests/job_queue tests require the backend in current working directory
    export PATH=.:$PATH
    QSTAT=`which qstat 2>/dev/null`
fi

if [ `uname` != "Linux" ]; then
    # Fallback if we are not on Linux
    $QSTAT $@
    exit $?
fi

file_age_seconds() {
    now=`date +%s`
    file_birth=`stat -L --format %Y $1`
    echo "$(($now - $file_birth))"
}

if [ -n "$2" ]; then
    # This is only for testing
    proxyfile=$2
else
    proxyfile=/tmp/${USER}-ert-qstat
fi

# If the file is not there, the first proxy invocation locks the output file and
# queries the backend. If this invocation can't get a lock, it will exit with
# an error code.
if [ ! -e "$proxyfile" ]; then
    flock --nonblock \
        --conflict-exit-code 1 \
        $proxyfile \
        --command "$QSTAT $QSTAT_OPTIONS > $proxyfile" || exit 1
fi
proxyage_seconds=`file_age_seconds $proxyfile`

# If it is due to update the proxyfile, try to update it.
# The other proxy invocations too late to get a lock will accept the cache.
if [ $proxyage_seconds -gt $CACHE_TIMEOUT ]; then
    flock --nonblock \
        --conflict-exit-code 0 \
        $proxyfile \
        --command "$QSTAT $QSTAT_OPTIONS > $proxyfile.tmp; mv $proxyfile.tmp $proxyfile"
fi

# The file is potentially updated:
proxyage_seconds=`file_age_seconds $proxyfile`

if [ $proxyage_seconds -gt $CACHE_INVALID ]; then
    echo "qstat_proxy: proxyfile $proxyfile was too old ($proxyage_seconds seconds). Giving up!"
    exit 1
fi

# Replicate qstat's error behaviour:
if [ -n "$1" ]; then
    grep -e "Job Id: ${1}" $proxyfile >/dev/null 2>&1 || {
        echo "qstat: Unknown Job Id $1" && cat $proxyfile >&2 && exit 1;
        }
fi

# Extract the job id from the proxyfile:
if [ -n "$1" ]; then
    awk "BEGIN { RS=\"Job Id: \"} /^$1/,/Job/ {printf \"Job Id: \"; print}" $proxyfile \
        | grep -v ^$
    exit 0
fi

# Empty $1, give out all we have (ert never asks for this)
cat $proxyfile

exit 0

# Example qstat output (when called with '-f'):
# Job Id: 15399.s034-lcam
#     Job_Name = DROGON-1
#     Job_Owner = combert
#     queue = hb120
#     job_state = H
# Job Id: 15400
#     Job_Name = DROGON-2
#     Job_Owner = barbert
#     queue = hb120
#     job_state = R
# Job Id: 15402.s034-lcam
#     Job_Name = DROGON-3
#     Job_Owner = foobert
#     queue = hb120
#     job_state = E
# Job Id: 15403.s034-lcam
#     Job_Name = DROGON-3
#     Job_Owner = foobert
#     queue = hb120
#     job_state = F

# NB: The F(inished) state is only reported if the "-x" option is supplied,
#     (this is also the default option to qstat in the driver)
