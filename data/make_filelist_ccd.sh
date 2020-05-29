#!/bin/bash

find classification-ccd -type l -name "*.jpg" | xargs -0 | sed 's/.*/data\/\0/' | sort > filelist-ccd.txt 

