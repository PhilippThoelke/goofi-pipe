<p align="center">
<img src=https://github.com/PhilippThoelke/goofi-pipe/assets/36135990/60fb2ba9-4124-4ca4-96e2-ae450d55596d width="150">
</p>

<h1 align="center">goofi-pipe</h1>
<h3 align="center">Generative Organic Oscillation Feedback Isomorphism Pipeline</h3>

# Installation
## Running the pipeline
If you only want to run goofi-pipe and not edit any of the code, make sure you activated the desired Python environment with Python>=3.8 and run the following commands in your terminal:
```bash
pip install git+https://github.com/PhilippThoelke/goofi-pipe # install goofi-pipe
goofi-pipe # start the application
```

## Development
In your terminal, make sure you activated the desired Python environment with Python>=3.8, and that you are in the directory where you want to install goofi-pipe. Then, run the following commands:
```bash
git clone git@github.com:PhilippThoelke/goofi-pipe.git # download the repository
cd goofi-pipe # navigate into the repository
pip install -e . # install goofi-pipe in development mode
goofi-pipe # start the application to make sure the installation was successful
```

# Basic Usage

## Accessing the Node Menu

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/358a897f-3947-495e-849a-e6d7ebce2238" width="small">
</p>

To access the node menu, simply double-click anywhere within the application window or press the 'Tab' key. The node menu allows you to add various functionalities to your pipeline. Nodes are categorized for easy access, but if you're looking for something specific, the search bar at the top is a handy tool.

## Common Parameters and Metadata

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/23ba6df7-7f28-4505-acff-205e42e48dcb" alt="Common Parameters" width="small">
</p>

**Common Parameters**: All nodes within goofi have a set of common parameters. These settings consistently dictate how the node operates within the pipeline.

- **AutoTrigger**: This option, when enabled, allows the node to be triggered automatically. When disabled,
the node is triggered when it receives input.
  
- **Max_Frequency**: This denotes the maximum rate at which computations are set for the node.

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/54604cfb-6611-4ce8-92b2-0b353584c5f5" alt="Metadata" width="small">
</p>

**Metadata**: This section conveys essential information passed between nodes. Each output node will be accompanied by its metadata, providing clarity and consistency throughout the workflow.

Here are some conventional components present in the metadata

- **Channel Dictionary**: A conventional representation of EEG channels names.
  
- **Sampling Frequency**: The rate at which data samples are measured. It's crucial for maintaining consistent data input and output across various nodes.

- **Shape of the Output**: Details the format and structure of the node's output.


## Playing with Pre-recorded EEG Signal using LslStream

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/db340bd9-07af-470e-a791-f3c2dcf4935e" width="small">
</p>

This image showcases the process of utilizing a pre-recorded EEG signal through the `LslStream` node. It's crucial to ensure that the `Stream Name` in the `LslStream` node matches the stream name in the node receiving the data. This ensures data integrity and accurate signal processing in real-time.

# Patch examples

## Basic Signal Processing Patch

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/52f85dd4-6395-4eb2-a347-6cf489d659da" width="medium">
</p>

This patch provides a demonstration of basic EEG signal processing using goofi-pipe.

1. **EegRecording**: This is the starting point where the EEG data originates. 

2. **LslClient**: The `LslClient` node retrieves the EEG data from `EegRecording`. Here, the visual representation of the EEG data being streamed in real-time is depicted. By default, the multiple lines in the plot correspond to the different EEG channels.

3. **Buffer**: This node holds the buffered EEG data.

4. **Psd**: Power Spectral Density (PSD) is a technique to measure a signal's power content versus frequency. In this node, the raw EEG data is transformed to exhibit its power distribution across distinct frequency bands.

5. **Math**: This node is employed to execute mathematical operations on the data. In this context, it's rescaling the values to ensure a harmonious dynamic range between 0 and 1, which is ideal for image representation. The resultant data is then visualized as an image.

One of the user-friendly features of goofi-pipe is the capability to toggle between different visualizations. By 'Ctrl+clicking' on any plot within a node, you can effortlessly switch between a line plot and an image representation, offering flexibility in data analysis.

## Sending Power Bands via Open Sound Control (OSC)

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/97576017-a737-47b9-aac6-bd0d00e0e7e9" width="medium">
</p>

Expanding on the basic patch, the advanced additions include:

- **Select**: Chooses specific EEG channels.
- **PowerBandEEG**: Computes EEG signal power across various frequency bands.
- **ExtendedTable**: Prepares data for transmission in a structured format.
- **OscOut**: Sends data using the Open-Sound-Control (OSC) protocol.

These nodes elevate data processing and communication capabilities.

## Real-Time Connectivity and Spectrogram

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/7c63a869-d20a-4f41-99fe-eb0931cebdc9" width="medium">
</p>

This patch highlights:

- **Connectivity**: Analyzes relationships between EEG channels, offering selectable methods like `wPLI`, `coherence`, `PLI`, and more.

- **Spectrogram**: Created using the `PSD` node followed by a `Buffer`, it provides a time-resolved view of the EEG signal's frequency content.

## Principal Component Analysis (PCA)
![PCA](https://github.com/PhilippThoelke/goofi-pipe/assets/36135990/d239eed8-4552-4256-9caf-d7c2fbb937e9)

Using PCA (Principal Component Analysis) allows us to reduce the dimensionality of raw EEG data, while retaining most of the variance. We use the first three components and visualize their trajectory, allowing us to identify patterns in the data over time. The topographical maps show the contrbution of each channel to the first four principal components (PCs).

## Realtime Classification

leverage the multimodal framework of goofi, state-of-the-art machine learning classifiers can be built on-the-fly to predict behavior from an array of different sources. Here's a brief walkthrough of three distinct examples:

### 1. Raw EEG Signal Classification
![EEG Signal Classification](https://github.com/PhilippThoelke/goofi-pipe/assets/36135990/2da6b555-9f79-40c7-9bd8-1f863dcf4137)
This patch captures raw EEG signals using the `EEGrecording` and `LslStream`module. The classifier module allows
to capture data from different states indicated by the user from *n* features, which in the present case are the 64 EEG channels. Some classifiers allow for visualization of feature importance. Here we show a topomap of the distribution of features importances on the scalp. The classifier outputs probability of being in each of the states in the training data. This prediction is smoothed using a buffer for less jiterry results.  
![Classifier parameters](https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/da2a86e3-efc8-4088-8d52-fb8c528dfb87)

### 2. Audio Input Classification
![Audio Input Classification](https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/4e50b13e-185d-414e-a39d-f6d39dc3e57f)
The audio input stream captures real-time sound data, which can also be passed through a classifier. Different sonic states can be predicted in realtime.

### 3. Video Input Classification
![Video Input Classification](https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/e7988ae9-cd2c-4b9f-907a-f438fd52328b)
![image_classification2](https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/77d33f2e-014f-4e3b-99fb-179f4bca1db0)
In this example, video frames are extracted using the `VideoStream` module. Similarly, prediction of labelled visual states can be achieved in realtime.
The images show how two states (being on the left or the right side of the image) can be detected using classification

These patches demonstrate the versatility of our framework in handling various types of real-time data streams for classification tasks.

## Musical Features using Biotuner

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/b426ce44-bf23-4b88-a772-5d183dc36a93" width="medium">
</p>

This patch presents a pipeline for processing EEG data to extract musical features:

- Data flows from the EEG recording through several preprocessing nodes and culminates in the **Biotuner** node, which specializes in deriving musical attributes from the EEG.

- **Biotuner** Node: With its sophisticated algorithms, Biotuner pinpoints harmonic relationships, tension, peaks, and more, essential for music theory analysis.

<p align="center">
<img src="https://github.com/PhilippThoelke/goofi-pipe/assets/49297774/042692ae-a558-48f2-9693-d09e33240373" width="medium">
</p>

Delving into the parameters of the Biotuner node:

- `N Peaks`: The number of spectral peaks to consider.
- `F Min` & `F Max`: Defines the frequency range for analysis.
- `Precision`: Sets the precision in Hz for peak extraction.
- `Peaks Function`: Method to compute the peaks, like EMD, fixed band, or harmonic recurrence.
- `N Harm Subharm` & `N Harm Extended`: Configures number of harmonics used in different computations.
- `Delta Lim`: Defines the maximal distance between two subharmonics to include in subharmonic tension computation.

For a deeper understanding and advanced configurations, consult the [Biotuner repository](https://github.com/AntoineBellemare/biotuner).


# Data Types and Node Categories

## Data Types

To simplify understanding, we've associated specific shapes with data types at the inputs and outputs of nodes:

- **Circles**: Represent arrays.
- **Triangles**: Represent strings.
- **Squares**: Represent tables.


## Node Categories

The nodes used in goofi-pipe can be categorized as follows:

1. **Inputs**:
   - `AudioStream`
   - `LslClient`
   - `SerialStream`
   - `VideoStream`
   - and other related input nodes.

2. **Array**:
   - Specific to array manipulation techniques such as:
     - `Join`
     - `Math`
     - `Reduce`
     - `Select`
     - `Transpose`
   
3. **Signal**:
   - Pertains to signal processing techniques such as:
     - `Buffer`
     - `Smooth`
     - `Resample`
     - `Filter`
     - `Psd`
     - `Hilbert`
     - `WelfordZTransform`
     - `StatisBaseline`
     - among others.
     
4. **Analysis**:
   
   - **Biosignal Analysis**:
     - Techniques related to biological signals include:
       - `Powerbands`
       - `LempelZiv`
       - `Connectivity`
       - `CardiacRespiration`
       - and more.
       
   - **Music Theory Analysis**:
     - Techniques related to musical theory include:
       - `Biotuner`
       - `Biorhythms`
       - `Spectromorphology`
       - `Transitional Harmony`
       - and others.

5. **Outputs**:
   - `OscoOut`
   - `WriteCsv`
   - `AudioOut`
   - `SharedMemOut`
   
6. **Misc**:
   - Miscellaneous nodes that don't fit into the above categories.


