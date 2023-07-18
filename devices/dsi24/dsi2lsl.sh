#!/bin/bash

if [ -d dsi24_lib ]
then
	# run dsi2lsl program
	sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:dsi24_lib ./dsi24_lib/dsi2lsl $@
else
	# setup was not completed
	echo "Please run setup.sh first"
fi
