#!/bin/bash

export DISPLAY=:0

# Launch pydcop
pydcop agent -n {{agt_name}} -p 9001 --uiport 10001  --orchestrator {{controller_proxy_addr}}:9000 &

#launch ui web server
cd ~/pyDcop-ui/dist
python3 -m http.server 4001

sleep 2
chromium-browser --start-fullscreen  http://localhost:4001 &