#!/bin/bash
set -euo pipefail

/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 640x480x24 -ac +extension GLX
sleep 3
herbstluftwm
