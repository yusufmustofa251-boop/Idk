#!/bin/bash
# crawl -> train -> ulang terus. Jalanin: nohup ./run_loop.sh > log.txt 2>&1 &
round=1
while true; do
  echo ""
  echo "######## RONDE $round ########"
  python -u crawler.py --wiki 150
  python -u train.py --steps 500
  round=$((round+1))
done
