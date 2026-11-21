#!/bin/bash
# crawl -> train -> ulang terus. Jalanin: nohup ./run_loop.sh > log.txt 2>&1 &
while true; do
  python crawler.py --wiki 20
  python train.py --steps 1000
done
