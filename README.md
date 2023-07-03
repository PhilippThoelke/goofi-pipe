TODO: Update README
---

# Real-time EEG Processing Pipeline

This is a modular processing and streaming pipeline for real-time EEG. In its current state it is able to listen to EEG data streamed through LSL, apply a range of processing to the data (e.g. PSD, Lempel-Ziv) and output the processed features to several output channels (e.g. visualization, OSC stream).

## Architecture
The main component of this pipeline is the `Manager` class. This class manages a single input stream, multiple data processing/feature extraction steps and multiple output channels. Internally it creates a dynamic buffer of raw EEG, which gets populated by a single `DataIn` source. This can be a stream of raw EEG data coming from an EEG headset in real-time (`data_in.EEGStream`) or from a previous EEG recording (`data_in.EEGRecording`). Once the internal buffer is filled, a list of `Processor` objects extracts features from the raw EEG.

A `Processor` object, that is an object that derives from the `Processor` parent-class, takes the buffer of raw EEG data and a selection of channels, and adds the features to a feature dictionary. Examples of `Processor`s are power spectral density (`processors.PSD`) or Lempel-Ziv complexity (`processors.LempelZiv`).

After applying all `Processor`s, the `Manager` class applies a `Normalization` strategy to the features, which can for example be a static baseline or running z-transform.

Finally, a list of `DataOut` objects serve to output the data to be used elsewhere. This can for example be an OSC stream (`data_out.OSCStream`) or a direct visualization of raw EEG (`data_out.PlotRaw`) or extracted features (`data_out.PlotProcessed`). Note that the built-in visualization tools are meant for debugging purposes only since these plots can slow down the processing loop.

## Usage
This simple example script creates a `Manager` instance and defines the data input stream, processing steps and output channels.

```python
from manager import Manager
import data_in
import processors
import data_out

mngr = Manager(
    data_in=data_in.EEGRecording.make_eegbci(),
    processors=[
        processors.PSD("delta"),
        processors.PSD("theta"),
        processors.PSD("alpha", include_chs=["O1", "Oz", "O2"]),
        processors.PSD("beta", include_chs=["P3", "P4"]),
        processors.PSD("gamma"),
        processors.LempelZiv(include_chs=["Fp1", "Fp2"]),
    ],
    data_out=[data_out.OSCStream("127.0.0.1", 5005), data_out.PlotProcessed()],
)
mngr.run()
```

- `data_in`: The input EEG is streamed from an example EEG dataset. The utility function `data_in.EEGRecording.make_eegbci()` downloads and initializes EEG recordings from the PhysioNet EEG BCI dataset.
- `processors`: We create a range of `Processor`s to extract spectral power from five frequency bands and Lempel-Ziv complexity. Additionally, _alpha_ and _beta_ power and _LempelZiv_ are only computed on a limited set of channels, while _delta_, _theta_ and _gamma_ are averaged across all channels.
- `data_out`: The extracted features are sent to two different outputs, an OSC stream to localhost on port 5005 and a debug visualization of the features in real time.

Finally, we call the blocking `mngr.run()` method, which starts the processing loop. Processing and output modules only start running once the `Manager`'s internal buffer is filled. The buffer size can be adjusted using the `buffer_seconds` argument in the `Manager`'s constructor, which is set to five seconds by default.

# TODO
- implement processor cache to avoid recomputing features that have not changed (e.g. power spectrum for PSD and 1/f)
    - e.g. add PowerSpectrum processor that is required by PSD and 1/f
    - set cache to dirty at the end of every update
- all processors return channel-wise features in addition to average
    - find standardized OSC address format (e.g. /device/feature/channel)
- implement threaded Processors into abstract base class (default can be either eager or timed evaluation)
- improve normalization by allowing for different normalization strategies for different features
    - implement normalizers as wrappers of processors
    - can take single processor or list
- modular preprocessing step (e.g. notch filter, band pass)
- clean up visualization code (replace by more efficient library in the long run)
- add more features (e.g. 1/f slope)
- find a standardized way to handle multiple channels (e.g. average vs channel-wise)
- integrate Crown EEG headset
