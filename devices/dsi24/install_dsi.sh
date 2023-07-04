#!/bin/bash

# check if the dsi2lsl binary already exists
if [ -f "dsi24_lib/dsi2lsl" ]; then
    echo "dsi2lsl binary already exists. Skipping setup."
    exit 0
fi

# create a build directory
mkdir dsi24_lib
cd dsi24_lib

# download LSL App-WearableSensing client (dsi2lsl)
wget https://raw.githubusercontent.com/labstreaminglayer/App-WearableSensing/master/CLI/dsi2lsl.c
wget https://raw.githubusercontent.com/labstreaminglayer/App-WearableSensing/master/CLI/lsl_c.h

# download DSI API from the WearableSensing website
wget -O DSI-API.zip https://wearablesensing.com/wp-content/uploads/2022/09/DSI_API_v1.18.2_11172021.zip
unzip DSI-API.zip
rm DSI-API.zip

# move relevant DSI API files into the dsi2lsl CLI directory
mv DSI_API_v1.18.2_11172021/DSI.h DSI_API_v1.18.2_11172021/DSI_API_Loader.c DSI_API_v1.18.2_11172021/libDSI-Linux-x86_64.so DSI_API_v1.18.2_11172021/DSI.py .
rm -r DSI_API_v1.18.2_11172021

# patch the dsi2lsl.c file to include use the correct LSL stream name
sed -i '199 c\  info = lsl_create_streaminfo((char*)streamName,"EEG",numberOfChannels,samplingRate,cft_float32,(char*)streamName);' dsi2lsl.c

# compile dsi2lsl
gcc -DDSI_PLATFORM=-Linux-x86_64 -o "dsi2lsl" dsi2lsl.c DSI_API_Loader.c -ldl -L /home/philipp/mambaforge/envs/nf/lib/ -llsl

# clean up directory
rm dsi2lsl.c DSI_API_Loader.c DSI.h lsl_c.h

# return to original directors
cd ..
