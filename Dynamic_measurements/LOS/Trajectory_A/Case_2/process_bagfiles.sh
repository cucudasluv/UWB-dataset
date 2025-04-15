#!/bin/bash

# 상위 폴더에서 시작하여 모든 하위 폴더 순회

  
  # 해당 폴더에 있는 .bag 파일에 대해 명령어 실행
for bagfile in 2025*.bag ; do
    if [[ $bagfile == 2025*.bag ]]; then
      rostopic echo -b "$bagfile" -p /dwm1001/least_square > LS.csv
      rostopic echo -b "$bagfile" -p /dwm1001/eskf > ESKF.csv
    fi
done
  

