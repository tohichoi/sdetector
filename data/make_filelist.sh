#!/bin/bash

find "$1" -name "*.jpg" | sed 's/.*/data\/\0/' | sort > "filelist-$1.txt"
