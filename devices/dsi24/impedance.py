from dsi24_lib.DSI import (
    Headset,
    NullMessageCallback,
    SampleCallback,
)
import sys
from threading import Thread
import numpy as np
from matplotlib import pyplot as plt, colors, cm
import mne
from mne.viz import plot_topomap
from mne.channels import make_standard_montage


# the COM port to use
if len(sys.argv) > 1:
    # if specified, get the port from the command line
    port = sys.argv[1]
else:
    # default port
    port = "/dev/rfcomm0"


# establish connection to the device
print("Connecting to DSI-24...", end="")
h = Headset()
h.SetMessageCallback(NullMessageCallback)
h.Connect(port)
print("done")

# create mne.Info object for DSI-24
montage = make_standard_montage("standard_1020")
info = mne.create_info(
    [ch.GetName() for ch in h.Channels() if ch.GetName() in montage.ch_names],
    sfreq=300,
    ch_types="eeg",
)
info.set_montage(montage)

# initialize impedances array
impedances = np.full(len(info.ch_names), float("nan"))


def viz_loop():
    def on_close(event):
        running[0] = False

    # set up plot
    plt.ion()
    plt.subplots()

    # stop running if the close event occured
    plt.gcf().canvas.mpl_connect("close_event", on_close)

    # initialize colormap
    cmap = colors.ListedColormap(["green", "orange", "red"])
    norm = colors.BoundaryNorm([0, 1, 10], 4, extend="max")
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable, ax=plt.gca())

    # drawing loop
    running = [True]
    while running[0]:
        plt.cla()
        # get channel names
        names = [f"{ch}\n{imp:.2f}" for ch, imp in zip(info.ch_names, impedances)]

        # plot impedances
        plot_topomap(
            impedances,
            info,
            contours=False,
            image_interp="nearest",
            show_names=True,
            sensors=False,
            names=names,
            cmap=cmap,
            cnorm=norm,
            axes=plt.gca(),
            show=False,
        )

        # redraw everything
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()


# start visualization in a separate thread
viz_thread = Thread(target=viz_loop)
viz_thread.start()


@SampleCallback
def impedance_callback(headset_ptr, *args):
    # callback from the device receiving data samples
    h = Headset(headset_ptr)
    impedances[:] = [h.GetSourceByName(ch).GetImpedanceEEG() for ch in info.ch_names]


# set up callback function to receive data
h.SetSampleCallback(impedance_callback, 0)

# start streaming from the device with the impedance driver
print("starting stream...")
h.StartImpedanceDriver()
h.StartDataAcquisition()

# keep running while the visualization thread is still alive
while viz_thread.is_alive():
    h.Idle(0.1)

# close the data stream
h.StopDataAcquisition()
