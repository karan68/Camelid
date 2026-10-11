R=~/r801
while ! grep -q "^EXIT" $R/parity-final.txt 2>/dev/null; do sleep 20; done
export CI=true CHROME_PATH=$HOME/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome PUPPETEER_EXECUTABLE_PATH=$HOME/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome
cd ~/camelid-801/frontend
[ -d node_modules ] || npm ci --no-audit --no-fund > $R/npm.log 2>&1
cp $R/live801.mjs ./live801.mjs
run() { # label binary data port
  bash $R/serve.sh $2 $3 $4 > /dev/null 2>&1 || return
  for _ in $(seq 1 300); do curl -s 127.0.0.1:$4/v1/health | grep -q '"generation_ready":true' && break; sleep 1; done
  VITE_CAMELID_PROXY_TARGET=http://127.0.0.1:$4 setsid nohup node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4181 > $R/vite-$1.log 2>&1 < /dev/null &
  VITE=$!
  sleep 8
  node live801.mjs http://127.0.0.1:4181 $1 $R/live >> $R/live801.log 2>&1
  kill $VITE; kill $(cat $3/server.pid); sleep 5
}
rm -rf $R/live-before-data && mkdir -p $R/live-before-data && cp ~/f2a-500/run-799/data/documents_rag.sqlite3 $R/live-before-data/
run before ~/camelid-803-base $R/live-before-data 18850
run after ~/camelid-801-fix5 $R/final5 18851
rm -f ~/camelid-801/frontend/live801.mjs
echo DONE >> $R/live801.log
