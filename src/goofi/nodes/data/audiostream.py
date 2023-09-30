import time
import sounddevice as sd
import numpy as np
from goofi.data import DataType
from goofi.node import Node
from goofi.params import IntParam, FloatParam, StringParam


class AudioStream(Node):
    def config_params():
        return {
            "audio": {
                "sfreq": IntParam(44100, 8000, 192000),
                "buffer_seconds": FloatParam(5.0, 1.0, 60.0),
                "channels": IntParam(1, 1, 10),
                "device": StringParam("", options=AudioStream.list_audio_devices()),
                "convert_to_mono": True,
            },
            "common": {
                "autotrigger": True
            }
        }
        
    def config_output_slots():
        return {"out": DataType.ARRAY}
    
    def setup(self):
        self.sfreq = self.params.audio.sfreq.value
        self.buffer_seconds = self.params.audio.buffer_seconds.value
        self.channels = self.params.audio.channels.value
        device = self.params.audio.device.value or None
        
        self.buffer = np.zeros((self.channels, int(self.sfreq * self.buffer_seconds)))
        
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                samplerate=self.sfreq, 
                channels=self.channels, 
                device=device
            )
            self.stream.start()
        except Exception as e:
            print(f"Error initializing audio stream: {e}")
            self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.buffer = np.roll(self.buffer, shift=-frames, axis=1)
        self.buffer[:, -frames:] = indata.T  # Assuming you want to transpose the samples
        
    def process(self):
        if not self.stream:
            print("Audio stream is not available.")
            return None
          
        data = np.copy(self.buffer)
        # Convert to mono if required
        convert_to_mono = self.params.audio.convert_to_mono.value
        if convert_to_mono and self.channels > 1:
            data = np.mean(data, axis=0, keepdims=False)
            # ensure that the data is 1D
            data = np.squeeze(data)     
        meta = {"sfreq": self.sfreq, "dim0": ["audio"]}
        return {"out": (data, meta)}
    
    @staticmethod
    def list_audio_devices():
        devices = sd.query_devices(kind="input")
        device_names = []
        for device in devices:
            if isinstance(device, dict) and 'name' in device:
                device_names.append(device['name'])
            elif isinstance(device, str):
                device_names.append(device)
        print("Available Audio Devices:")
        for name in device_names:
            print(name)
        return device_names
