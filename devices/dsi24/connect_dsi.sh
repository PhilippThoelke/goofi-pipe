#!/bin/bash

# cache sudo credentials for connecting devices
sudo -v

# scan for bluetooth DSI devices
echo "Scanning for devices..."
SCAN_OUTPUT=$(hcitool scan 2>/dev/null)

SCANNING_MESSAGE=$(echo "$SCAN_OUTPUT" | grep -E '(Scanning|DSI24)' | awk 'NR==1 {print}')
DSI_ADDRESS=$(echo "$SCAN_OUTPUT" | grep -E '(Scanning|DSI24)' | awk '$2 ~ /DSI24/ {print $1}')

# make sure we are able to scan
if [ -z "$SCANNING_MESSAGE" ]; then
    echo "Scanning failed! Make sure bluetooth is enabled."
    exit -1
fi

# check if we found a device
if [ -n "$DSI_ADDRESS" ]; then
    echo "Found DSI at $DSI_ADDRESS"
else
    echo "No DSI headset found"
    exit -2
fi


# make sure we're not already connected to the device
RFCOMM_PORT=$(rfcomm | grep $DSI_ADDRESS | awk '{print $1}' | tr -d ':')

if [ -n "$RFCOMM_PORT" ]; then
    # disconnect from DSI
    echo -e "\nAlready connected to this device, disconnecting..."
    while read -r PORT; do
        echo "Disconnecting from $PORT"
        sudo rfcomm release $PORT
    done <<< "$RFCOMM_PORT"
fi


# find an empty rfcomm port
echo -e "\nLooking for an empty rfcomm port to connect the headset to"
CURRENT_PORT=0
RFCOMM_PORT=""
while true; do
    echo -n "Trying port rfcomm$CURRENT_PORT..."
    if [ -z "$(rfcomm | grep "rfcomm$CURRENT_PORT")" ]; then
        # found a port
        RFCOMM_PORT="rfcomm$CURRENT_PORT"
        echo "success"
        break
    fi
    echo "failed"

    CURRENT_PORT=$((CURRENT_PORT + 1))
done


# connect to the headset
echo -e "\nConnecting to DSI at $DSI_ADDRESS on port $RFCOMM_PORT"
sudo rfcomm bind /dev/$RFCOMM_PORT $DSI_ADDRESS 1
echo "Done!"


echo -n -e "\nStart LSL stream? [y/n]: "
read USER_INPUT

if [ "$USER_INPUT" = "y" ]; then
    echo "Starting LSL stream..."
    ./dsi2lsl.sh --port=/dev/$RFCOMM_PORT
fi
