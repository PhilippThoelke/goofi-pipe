import openai
import io
import soundfile as sf
from goofi.node import Node
import numpy as np
from goofi.params import StringParam, FloatParam
from goofi.data import Data, DataType
import librosa
client = openai.OpenAI()

class SpeechSynthesis(Node):
    def config_input_slots():
        return {"text": DataType.STRING, "voice": DataType.ARRAY}

    def config_output_slots():
        return {"speech": DataType.ARRAY, "transcript": DataType.STRING}

    def config_params():
        return {
            "speech_generation": {
                "openai_key": StringParam("openai.key"),
                "speed": FloatParam(0.25, 0.1, 2)
                }
        }

    def setup(self):
        key = self.params["speech_generation"]["openai_key"].value
        with open(key, "r") as f:
            openai.api_key = f.read().strip()

    def process(self, text: Data, voice: Data,):
        speech = None
        transcript = ''

        if text is None and voice is None:
            return {"speech": (np.array([]), {}), "transcript": ("", {})}

        if text is not None:
            audio_generator = self.synthesize_speech_stream(text.data, self.params["speech_generation"]["speed"].value)
            speech = self.bytes_to_array(b''.join([speech_bytes for speech_bytes in audio_generator]))  # Convert bytes to numpy array

        elif voice is not None:
            transcript = self.transcribe_voice(voice.data) or ""  # Ensure transcript is a string

        return {"speech": (speech, {}), "transcript": (transcript, {})}

    def synthesize_speech_stream(self, text, speed):
        response = openai.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=text,
                        speed=speed
                        )
        # This would be a generator yielding audio chunks
        for chunk in response.iter_bytes():
            yield chunk
    
    def bytes_to_array(self, audio_bytes):
        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as sf_file:
                audio_array = np.array(sf_file.read(dtype='float32'))
                # Resample if the sample rate is different from 44100 Hz
                if sf_file.samplerate != 44100:
                    audio_array = self.resample_audio(audio_array, sf_file.samplerate, 44100)
                return audio_array

    def resample_audio(self, audio, input_rate, output_rate):
        # Resample audio from input_rate to output_rate
        return librosa.resample(audio, orig_sr=input_rate, target_sr=output_rate)

    def transcribe_voice(self, voice_buffer):
        # Convert the numpy array buffer to a WAV file in memory
        with io.BytesIO() as audio_file:
            sf.write(audio_file, voice_buffer, 44100, format='wav')
            audio_file.seek(0)

            # Send the audio file to OpenAI for transcription
            response = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            return response["choices"][0]["text"]

    