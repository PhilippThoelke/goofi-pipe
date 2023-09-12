import threading
import time
from typing import List, Optional, Union

import mne
import numpy as np
import serial
import serial.tools.list_ports
import sounddevice as sd
from mne.datasets import eegbci
from mne.io import BaseRaw, concatenate_raws, read_raw
from mne_realtime import LSLClient, MockLSLStream

from goofi.utils import DataIn

mne.set_log_level(False)


class DummyStream(DataIn):
    """
    Dummy stream for testing purposes.

    Parameters:
        data (str): the type of data to generate, one of "normal", "zeros", "arange", "exp"
        n_channels (int): number of channels to generate
        sfreq (int): sampling frequency
        buffer_seconds (int): number of seconds to buffer
    """

    def __init__(
        self, data="normal", n_channels=5, sfreq=100, buffer_seconds=5, frequency=1
    ):
        super().__init__(buffer_seconds)
        assert data in ["normal", "zeros", "arange", "exp", "osc"]

        self.data = data
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.frequency = frequency  # New: specific frequency for 'osc' data
        self.last_receive = time.time() - buffer_seconds

        # fill the buffer so we don't need to wait for the first receive
        self.update()

    @property
    def info(self) -> mne.Info:
        return mne.create_info(
            [f"ch{i}" for i in range(self.n_channels)], self.sfreq, "eeg"
        )

    def update(self):
        # Dummy update function to comply with the original design.
        pass

    def receive(self) -> np.ndarray:
        n_samples = int((time.time() - self.last_receive) * self.sfreq)
        self.last_receive = time.time()

        if self.data == "normal":
            dat = np.random.normal(size=(self.n_channels, n_samples))
        elif self.data == "zeros":
            dat = np.zeros((self.n_channels, n_samples))
        elif self.data == "arange":
            dat = (
                np.arange(n_samples)
                .repeat(self.n_channels)
                .reshape(n_samples, self.n_channels)
                .T
            )
        elif self.data == "exp":
            dat = (
                np.exp(np.arange(n_samples) / self.sfreq)
                .repeat(self.n_channels)
                .reshape(n_samples, self.n_channels)
                .T
            )
        elif self.data == "osc":
            # New: generate an oscillating sine wave with a specified frequency
            t = np.linspace(0, n_samples / self.sfreq, n_samples)
            dat = np.sin(2 * np.pi * self.frequency * t)
            dat = np.tile(dat, (self.n_channels, 1))

        return dat.astype(np.float32) * 1e-6


class SerialStream(DataIn):
    def __init__(
        self,
        sfreq: int,
        buffer_seconds: int = 5,
        auto_select: bool = True,
        port: Optional[str] = None,
    ):
        super(SerialStream, self).__init__(buffer_seconds=buffer_seconds)
        self.sfreq = sfreq
        self.auto_select = auto_select
        self.port = port
        self.serial_buffer = []
        self.buffer_times = []
        self.last_processed_time = None
        self.lock = threading.Lock()
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def read_serial(self):
        if self.auto_select:
            ser = serial.Serial(SerialStream.detect_serial_port(), 115200, timeout=1)
        else:
            ser = serial.Serial(self.port, 115200, timeout=1)
        if not ser.isOpen():
            ser.open()

        ESC = b"\xDB"
        END = b"\xC0"
        ESC_END = b"\xDC"
        ESC_ESC = b"\xDD"

        packet = bytearray()
        while True:
            c = ser.read()
            if c == END:
                if packet:  # ignore empty packets
                    if len(packet) == 2:
                        value = packet[0] << 8 | packet[1]
                        current_time = time.time()
                        with self.lock:
                            self.serial_buffer.append(value)
                            self.buffer_times.append(current_time)
                    else:
                        print("Invalid packet received")
                    packet = bytearray()
            elif c == ESC:
                c = ser.read()
                if c == ESC_END:
                    packet.append(0xC0)
                elif c == ESC_ESC:
                    packet.append(0xDB)
                else:
                    print("Invalid escape sequence")
            elif len(c) > 0:
                packet.append(c[0])

    @property
    def info(self) -> mne.Info:
        return mne.create_info(
            ch_names=["serial"],
            ch_types=["bio"],
            sfreq=self.sfreq,
        )

    def receive(self) -> np.ndarray:
        with self.lock:
            # Copy the buffer and times list
            data = np.array(self.serial_buffer)
            times = np.array(self.buffer_times)

            if len(data) < 2:
                return None

            # Calculate new times for interpolation
            if self.last_processed_time is None:
                new_times_start = times[0]
            else:
                new_times_start = self.last_processed_time

            # Calculate the number of samples based on the sampling frequency and the time span
            num_samples = int((times[-1] - new_times_start) * self.sfreq)
            if num_samples == 0:
                return None

            # Create the new time array
            new_times = np.linspace(new_times_start, times[-1], num_samples)

            # Clear the buffer and times list
            self.serial_buffer.clear()
            self.buffer_times.clear()

        # Resample the data to a common sampling frequency
        new_data = np.interp(new_times, times, data)

        # Update the last processed time
        self.last_processed_time = new_times[-1]
        return new_data[None]

    def detect_serial_port(name="Arduino"):
        ports = serial.tools.list_ports.comports()
        print(ports)
        for port in ports:
            if name in port.description or "Serial" in port.description:
                return port.device
        return None


class EEGStream(DataIn):
    """
    Incoming LSL stream with raw EEG data.

    Parameters:
        host (str): the LSL stream's hostname
        port (int): the LSL stream's port (if None, use the LSLClient's default port)
        pick_eeg (bool): if True, only pick EEG channels
        buffer_seconds (int): the number of seconds to buffer incoming data
    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        pick_eeg: bool = True,
        buffer_seconds: int = 5,
    ):
        super(EEGStream, self).__init__(buffer_seconds=buffer_seconds)
        self.pick_eeg = pick_eeg
        self.pick_idxs = None

        # start LSL client
        self.client = LSLClient(host=host, port=port)
        self.client.start()
        self.data_iterator = self.client.iter_raw_buffers()

    @property
    def info(self) -> mne.Info:
        """
        Returns the MNE info object corresponding to this EEG stream
        """
        if self.pick_eeg:
            # exclude the aux channels
            self.pick_idxs = mne.pick_channels(
                self.client.get_measurement_info()["ch_names"],
                include=[],
                exclude=["X1", "X2", "X3", "TRG", "A1", "A2"],
            )
            # only pick EEG channels
            self.pick_idxs = [
                idx
                for idx in self.pick_idxs
                if idx in mne.pick_types(self.client.get_measurement_info(), eeg=True)
            ]
            return mne.pick_info(self.client.get_measurement_info(), self.pick_idxs)
        return self.client.get_measurement_info()

    def receive(self) -> np.ndarray:
        """
        Returns newly acquired samples from the EEG stream as a NumPy array
        with shape (Channels, Time). If there are no new samples None is returned.
        """
        data = next(self.data_iterator)
        if data.size == 0:
            return None
        if self.pick_eeg:
            data = data[self.pick_idxs]
            # re-reference to the average of all EEG channels
            # data = data - np.mean(data, axis=0, keepdims=True)
        return data


class EEGRecording(EEGStream):
    """
    Stream previously recorded EEG from a file. The data is loaded from the file and streamed
    to a mock LSL stream, which is then accessed via the EEGStream parent-class.

    Parameters:
        raw (str, BaseRaw): file-name of a raw EEG file or an instance of mne.io.BaseRaw
    """

    def __init__(self, raw: Union[str, BaseRaw]):
        # load raw EEG data
        if not isinstance(raw, BaseRaw):
            raw = read_raw(raw)
        raw.load_data().pick("eeg")

        # start the mock LSL stream to serve the EEG recording
        host = "mock-eeg-stream"
        self.mock_stream = MockLSLStream(host, raw, "eeg")
        self.mock_stream.start()

        # start the LSL client
        super(EEGRecording, self).__init__(host=host)

    @staticmethod
    def make_eegbci(
        subjects: Union[int, List[int]] = 1,
        runs: Union[int, List[int]] = [1, 2],
    ):
        """
        Static utility function to instantiate an EEGRecording instance using
        the PhysioNet EEG BCI dataset. This function automatically downloads the
        dataset if it is not present.
        See https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#mne-datasets-eegbci-load-data
        for information about the dataset and a description of different runs.

        Parameters:
            subjects (int, List[int]): which subject(s) to load data from
            runs (int, List[int]): which run(s) to load from the corresponding subject
        """
        raw = concatenate_raws([read_raw(p) for p in eegbci.load_data(subjects, runs)])
        eegbci.standardize(raw)
        return EEGRecording(raw)


# sd.default.device = 'Microphone Array (Realtek Audio)'
sd.default.samplerate = 44100
import mne
import numpy as np
import sounddevice as sd
from scipy.signal import resample


class AudioStream(DataIn):
    def __init__(self, channels=1, sfreq=44100, buffer_seconds=5, device=None):
        super().__init__(buffer_seconds)
        self.channels = channels
        self.sfreq = sfreq
        self.device = device
        self.target_sfreq = 1000  # New target sampling rate
        try:
            self.stream = sd.InputStream(
                samplerate=self.sfreq, channels=self.channels, device=self.device
            )
            self.stream.start()
        except Exception as e:
            print(f"Error initializing audio stream: {e}")
            self.stream = None

    def downsample(self, samples: np.ndarray) -> np.ndarray:
        """Downsample the audio signal."""
        num_samples = int(samples.shape[1] * (self.target_sfreq / self.sfreq))
        return resample(samples, num_samples, axis=1)

    @property
    def info(self) -> mne.Info:
        ch_types = ["eeg"]
        ch_names = ["audio"]
        return mne.create_info(ch_names, self.target_sfreq, ch_types)

    def receive(self) -> np.ndarray:
        if not self.stream:
            print("Audio stream is not available.")
            return None

        samples, overflowed = self.stream.read(int(self.sfreq * self.buffer_seconds))
        if overflowed:
            print("Warning: Audio buffer overflowed!")

        downsampled_samples = self.downsample(samples.T)
        return downsampled_samples

    @staticmethod
    def list_audio_devices():
        print(sd.query_devices(kind="input"))

    def __del__(self):
        if hasattr(self, "stream") and self.stream:
            self.stream.stop()
            self.stream.close()

from pythonosc import dispatcher, osc_server
import numpy as np
import mne

class OSCStream(DataIn):
    def __init__(self, address: str="/osc_address", port: Optional[int]=8000, buffer_seconds: int=5):
        super().__init__(buffer_seconds)
        self.address = address
        self.port = port
        self.buffer = []
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map(self.address, self.osc_callback)
        
        self.server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", self.port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        
    def osc_callback(self, unused_addr, *args):
        self.buffer.append(args)

    @property
    def info(self) -> mne.Info:
        ch_types = ['eeg']
        ch_names = ['OSC']
        return mne.create_info(ch_names, 1000, ch_types)

    def receive(self) -> np.ndarray:
        if len(self.buffer) == 0:
            print("No data received yet.")
            return None

        buffer_length = len(self.buffer)
        channel_count = len(self.buffer[0])
        
        data = np.zeros((channel_count, buffer_length))
        for i in range(buffer_length):
            for j in range(channel_count):
                data[j, i] = self.buffer[i][j]
        
        self.buffer = []  # Clear the buffer
        return data

    def start(self):
        print(f"Listening for OSC messages on port {self.port}...")
        self.server_thread.start()

    def __del__(self):
        self.server.shutdown()
        
# list_audio_devices()
# print('-------------------------------------------')
# print(sd.query_devices(kind='input'))
# print('-------------------------------------------')

# Create an instance of the audio stream
# audio_stream = AudioStream(channels=1, sfreq=44100, buffer_seconds=5)

# Get new samples
# samples = audio_stream.receive()

# Your samples are now downsampled to 20kHz.


# #device_info = sd.query_devices(5, 'input')
# #print(device_info['default_samplerate'])

# # Create an instance of the audio stream
# audio_stream = AudioStream(channels=2, sfreq=44100, buffer_seconds=5)

# # Get new samples
# samples = audio_stream.receive()
