<p align="center">
<img src=https://github.com/PhilippThoelke/goofi-pipe/assets/36135990/60fb2ba9-4124-4ca4-96e2-ae450d55596d width="150">
</p>

<h1 align="center">goofi-pipe</h1>
<h3 align="center">Generative Organic Oscillation Feedback Isomorphism Pipeline</h3>

\
🚀 Welcome to goofi-pipe, our modular real-time data processing machine! Equipped to deal with a broad spectrum of biosignals, this pipeline is designed for flexible creative prototyping integrative biofeedback systems.

🎯 Presently, it can handle data input from LSL, OSC, and Serial, putting the focus on live processing of electrophysiological data streams. Want to connect your brain to generative AI models? goofi-pipe's got you covered! It plays well with generative AI models like Stable Diffusion/DALL-E and GPT-3/4 to generate visual and textual representations features extracted from the data.

🔬 Make your prototyping journey faster with the ability to visualize processed data on the go. Plus, seamless integration with your hardware and software is a breeze, thanks to streaming capabilities via OSC and sockets.


## Installation
Clone the repository and install the required packages using `pip`:
```bash
git clone git@github.com:PhilippThoelke/goofi-pipe.git
cd goofi-pipe
pip install -e .
```

## Device integration
Refer to the [`devices`](https://github.com/PhilippThoelke/goofi-pipe/tree/main/devices) directory for device-specific setup instructions.

## Architecture
The main component of this pipeline is the `Manager` class. This class manages a dictionary of input streams (e.g. EEG, Serial, ...), a list of data processing/feature extraction steps (e.g. PSD, complexity, AI-models, ...), feature normalization and list of output channels (e.g. visualization, OSC, ...). By default, the pipeline has data buffers of 5 seconds and runs at 10Hz, meaning that new features are extracted 10 times a second, using the last 5 seconds of data.

Internally the pipeline handles buffers of raw time-series data, which is populated with data from `DataIn` sources. Once these buffers are full, a range of `Processor`s extracts features from the raw data.

A `Processor` object, that is an object that derives from the `Processor` parent-class, simply works on a single buffer of raw data and extracts some feature, which is subsequently added to the output dictionary.

After applying all `Processor`s, the `Manager` class applies a `Normalization` strategy to a subset features to rescale them to a standardized range.

Finally, a list of `DataOut` objects serve to output the data to be used elsewhere. This can for example be an OSC stream (`data_out.OSCStream`), or visualization of raw time series (`data_out.PlotRaw`) or extracted features (`data_out.PlotProcessed`). Note that the built-in visualization tools are meant for prototyping only, as currently the streams can be quite demanding and limit the processing speed of the ful pipeline.

> **Note**
> Check out the [`data_in`](https://github.com/PhilippThoelke/goofi-pipe/blob/main/goofi/data_in.py), [`processors`](https://github.com/PhilippThoelke/goofi-pipe/blob/main/goofi/processors.py), [`normalization`](https://github.com/PhilippThoelke/goofi-pipe/blob/main/goofi/normalization.py), [`data_out`](https://github.com/PhilippThoelke/goofi-pipe/blob/main/goofi/data_out.py) files for complete lists of implemented features.

## Usage
Check out the [`scripts`](https://github.com/PhilippThoelke/goofi-pipe/tree/main/scripts) directory for some example configurations of the pipeline, or have a look at the following code, extracting a broad range of features from pre-recorded EEG:

```python
from goofi import data_in, data_out, manager, normalization, processors

# configure the pipeline through the Manager class
mngr = manager.Manager(
    data_in={
        "eeg": data_in.EEGRecording.make_eegbci()  # stream some pre-recorded EEG from a file
    },
    processors=[
        # global delta power
        processors.PSD("delta"),
        # global theta power
        processors.PSD("theta"),
        # occipital alpha power (eyes open/closed)
        processors.PSD("alpha", include_chs=["O1", "Oz", "O2"]),
        # parietal beta power (motor activity)
        processors.PSD("beta", include_chs=["P3", "P4"]),
        # global gamma power
        processors.PSD("gamma"),
        # pre-frontal Lempel-Ziv complexity
        processors.LempelZiv(include_chs=["Fp1", "Fp2"]),
        # map EEG oscillations to emission spectra
        processors.Bioelements(channels={"eeg": ["C3"]}),
        # extract colors from harmonic ratios of EEG oscillations
        processors.Biocolor(channels={"eeg": ["C3"]}),
        # ask GPT-3 to write a line of poetry based on EEG features (requires OpenAI API key)
        processors.TextGeneration(
            processors.TextGeneration.POETRY_PROMPT,
            "/eeg/biocolor/ch0_peak0_name",
            "/eeg/bioelements/ch0_bioelements",
        ),
    ],
    normalization=normalization.WelfordsZTransform(),  # apply a running z-transform to the features
    data_out=[
        data_out.OSCStream("127.0.0.1", 5005),  # stream features on localhost
        data_out.PlotProcessed(),  # visualize the extracted features
    ],
)

# start the pipeline
mngr.run()
```

- `data_in`: dictionary of stream labels and `DataIn` data sources
- `processors`: a list of `Processor`s to extract a range of features from the EEG
- `normalization` a normalization strategy for the extracted features, should be an instance of `Normalization`
- `data_out`: a list of data outputs (`DataOut`) to use the extracted features

Finally, we call the blocking `mngr.run()` method, which starts the processing loop. Processing and output modules only start running once the `Manager`'s internal buffer is filled. The buffer size can be adjusted using the `buffer_seconds` argument in the `Manager`'s constructor, which is set to 5 seconds by default.

TO-DO
---
This is a list of current TO-DOs for continued development of goofi-pipe
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
