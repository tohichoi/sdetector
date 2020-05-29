#!/bin/bash

find image -type f -name "*.jpg" | xargs -0 | sed 's/.*/data\/\0/' | sort > filelist-all.txt

