#!/usr/bin/env bash

# 実行時に指定された引数の数、つまり変数 $# の値が 3 でなければエラー終了。
if [ $# -ne 1 ]; then
  echo "実行するには2個の引数が必要です。"
  echo "./check_sample.sh dat/in_1st_my_wav check.txt "
  exit 1
fi

for FILE in $1/*.wav
do
    FILENAME=`echo ${FILE} | sed 's/\.[^\.]*$//'`
    echo -n ${FILENAME}.wav
    echo -n ' '
    ffprobe -hide_banner -v error -show_streams ${FILENAME}.wav | grep -e channels -e sample_rate -e bits_per_sample | tr '\n' ' '
    echo
done