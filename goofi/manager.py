import time
from typing import Dict, List, Union

from goofi.normalization import StaticBaselineNormal
from goofi.processors import SignalStd
from goofi.utils import DataIn, DataOut, Normalization, Processor


class Manager:
    """
    Central class to manage an EEG input stream, several processing steps,
    feature normalization and multiple output channels.

    Parameters:
        data_in (List[DataIn]): a list of DataIn channels (e.g. EEGStream, EEGRecording)
        processors (List[Processor]): a list of Processor instances (e.g. PSD, LempelZiv)
        normalization (Normalization): the normalization strategy to apply to the extracted features
        data_out (List[DataOut]): a list of DataOut channels (e.g. OSCStream, PlotProcessed)
        frequency (int): frequency of the data processing loop (-1 to run as fast as possible)
        noise_threshold (float or Dict[str,float]): threshold for detecting if a device is in use
    """

    def __init__(
        self,
        data_in: Dict[str, DataIn],
        processors: List[Processor],
        normalization: Normalization,
        data_out: List[DataOut],
        frequency: int = 10,
        noise_threshold: Union[Dict[str, float], float] = 4,
    ):
        if isinstance(data_in, dict):
            for name in data_in.keys():
                assert not name.startswith(" ") and not name.endswith(
                    " "
                ), f"DataIn names cannot start or end with a space (got '{name}')"

        self.data_in = data_in
        self.processors = processors
        self.normalization = normalization
        self.data_out = data_out
        self.noise_threshold = noise_threshold

        # compute signal standard deviation to estimate noise level
        self.std_processor = SignalStd()
        self.std_normalization = StaticBaselineNormal(30)

        # auxiliary attributes
        self.frequency = frequency
        self.too_slow_count = 0
        self.filling_buffer = len(self.data_in) > 0

    def update(self):
        """
        Fetches new data from the input channels, processes the data and
        outputs the processed data to the output channels.
        """
        # fetch raw data
        if len(self.data_in) > 0 and any(
            d.update() == -1 for d in self.data_in.values()
        ):
            return
        elif self.filling_buffer:
            print("done")
            self.filling_buffer = False

        # process raw data (feature extraction)
        processed, intermediates, normalize_mask = {}, {}, {}

        # assess if the device is in use or just producing noise
        noise_container = {}
        if len(self.data_in) > 0:
            self.std_processor(self.data_in, noise_container, {})
            self.std_normalization.normalize(noise_container)

        for key in self.data_in.keys():
            if len(self.data_in) > 0:
                if isinstance(self.noise_threshold, dict):
                    thresh = self.noise_threshold[key]
                else:
                    thresh = self.noise_threshold
                processed[f"/{key}/in-use"] = float(
                    noise_container[f"/{key}/signal-std"] < thresh
                )
                normalize_mask[f"/{key}/in-use"] = False

        for processor in self.processors:
            normalize_mask.update(processor(self.data_in, processed, intermediates))

        # extract the features that need normalization
        finished = {
            lbl: feat for lbl, feat in processed.items() if not normalize_mask[lbl]
        }
        unfinished = {
            lbl: feat for lbl, feat in processed.items() if normalize_mask[lbl]
        }

        # normalize extracted features
        self.normalization.normalize(unfinished)
        finished.update(unfinished)
        finished = {lbl: finished[lbl] for lbl in processed.keys()}

        # update data outputs
        for out in self.data_out:
            out.update(self.data_in, finished, intermediates)

    def run(self, n_iterations: int = -1):
        """
        Start the fetching and processing loop and limit the loop to run at a
        constant update rate.

        Parameters:
            n_iterations (int): number of iterations to run the loop for (-1 for infinite)
        """
        print("Filling buffer(s)...", end="")

        last_time = time.time()
        it = 0
        while it != n_iterations:
            # receive, process and output data
            self.update()
            it += 1

            # limit the loop to run at a constant update rate
            if self.frequency > 0:
                # ensure a constant sampling frequency
                current_time = time.time()
                sleep_dur = 1 / self.frequency - current_time + last_time
                if sleep_dur >= 0:
                    time.sleep(sleep_dur)
                else:
                    self.too_slow_count += 1
                    print(
                        f"Processing too slow to run at {self.frequency}Hz"
                        f" ({self.too_slow_count})"
                    )
                last_time = time.time()


if __name__ == "__main__":
    from goofi import data_in, data_out, manager, normalization, processors

    # configure the pipeline through the Manager class
    mngr = manager.Manager(
        data_in={
            "eeg": data_in.EEGRecording.make_eegbci()  # stream some pre-recorded EEG from a file
        },
        processors=[
            # global delta power
            processors.PSD(label="delta"),
            # global theta power
            processors.PSD(label="theta"),
            # global alpha power
            processors.PSD(fmin=8, fmax=12, label="global-alpha"),
            # occipital alpha power (eyes open/closed)
            processors.PSD(label="alpha", channels={"eeg": ["O1", "Oz", "O2"]}),
            # parietal beta power (motor activity)
            processors.PSD(label="beta", channels={"eeg": ["P3", "P4"]}),
            # global gamma power
            processors.PSD(label="gamma"),
            # theta/alpha ratio
            processors.Ratio("/eeg/theta", "/eeg/global-alpha", label="theta/alpha"),
            # pre-frontal Lempel-Ziv complexity
            processors.LempelZiv(channels={"eeg": ["Fp1", "Fp2"]}),
            # map EEG oscillations to emission spectra
            processors.Bioelements(channels={"eeg": ["C3"]}),
            # extract colors from harmonic ratios of EEG oscillations
            processors.Biocolor(channels={"eeg": ["C3"]}),
            # ask GPT-3 to write a line of poetry based on EEG features (requires OpenAI API key)
            processors.TextGeneration(
                processors.TextGeneration.POETRY_PROMPT,
                "/eeg/biocolor/ch0_peak0_name",
                "/eeg/bioelements/ch0_bioelements",
                keep_conversation=True,
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
