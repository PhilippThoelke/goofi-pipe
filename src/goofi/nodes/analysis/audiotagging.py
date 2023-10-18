import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam, BoolParam


class AudioTagging(Node):
    def config_input_slots():
        # Defining two input slots for two input signals
        return {"audioIn": DataType.ARRAY}

    def config_output_slots():
        # Defining two output slots for the resampled signals
        return {"tags": DataType.STRING,
                'probabilities': DataType.ARRAY,
                'embedding': DataType.ARRAY}

    def config_params():
        return {"selection": {"threshold": FloatParam(0.5, 0.0, 1.0),
                             "number_of_tags": IntParam(3, 1, 527),}}
    
    def setup(self):
        import librosa
        import panns_inference
        from panns_inference import AudioTagging, SoundEventDetection, labels
        self.at = AudioTagging(checkpoint_path=None, device='cpu')
        self.labels = labels

    def process(self, audioIn: Data):
        if audioIn.data is None:
            return None
        
        #check if sf is 32k
        if audioIn.meta["sfreq"] != 32000:
            raise ValueError("Sampling frequency must be 32k")
        
        threshold = self.params["selection"]["threshold"].value
        n_tags = self.params["selection"]["number_of_tags"].value
        audioIn.data = audioIn.data.reshape(1,-1)

        (clipwise_output, embedding) = self.at.inference(audioIn.data)

        best_labels = np.argsort(clipwise_output[0])[::-1]
        tags = np.array(self.labels)[best_labels]
        probabilities = clipwise_output[0][best_labels]
        embedding = embedding.squeeze()
        
        formatted_tags = "\n".join([x for i, x in enumerate(tags[:n_tags-1]) if probabilities[i]>threshold])
        formatted_probs = np.array([x for i, x in enumerate(probabilities[:n_tags-1]) if probabilities[i]>threshold])
        
    
        return {"tags": (formatted_tags, audioIn.meta),
                "probabilities": (formatted_probs, audioIn.meta),
                "embedding": (embedding, audioIn.meta)}
