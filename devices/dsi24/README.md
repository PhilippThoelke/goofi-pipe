> __Warning__
> This setup is currently only tested on Linux. It likely won't work on Windows, it might on Mac.

# Setup for the DSI24 EEG headset by Wearable Sensing
## Installing the API
Run the `install_dsi.sh` script to automatically download the API from WearableSensing and compile it into an executable. The script will create a directory called `dsi24_lib` in the current working directory, containing the `dsi2lsl` executable and a Python API for accessing device info.

## Connecting to a device
Run the `connect_dsi.sh` script to search for available DSI24 devices via bluetooth and connect on an available rfcomm port. After a successful connection, the script will ask to directly start streaming data to LSL. Select `n` if you want to check electrode impedance before starting the stream.

## Checking electrode impedance
Run the `impedance.py` script to bring up a visualization of electrode impedance. Note that due to how Wearable Sensing designed the API, this script has to be run as root. Make sure that you still have access to the correct Python environment when running `sudo python impedance.py`.

## Streaming data to LSL
To stream raw data from the headset to LSL, run `dsi2lsl.sh`, which will utilize the compiled `dsi24_lib/dsi2lsl`. Alternatively, you can simply select `y` when running `connect_dsi.sh` to start streaming data immediately after connecting to a device. Note that depending on the directory hierarchy, you might have to adjust the path to the `dsi2lsl` executable and to the `dsi24_lib` library in `dsi2lsl.sh`.

# TODO
- make the installation more robust
    - standardize the directory structure
    - make sure that the scripts can be run from anywhere
    - add option to select a device to connect if multiple are available
- provide setup scripts for Windows
- test installation on Mac
