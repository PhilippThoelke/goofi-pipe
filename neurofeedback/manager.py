import time
from typing import Dict, List

import numpy as np
from utils import DataIn, DataOut, Normalization, Processor


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
    """

    def __init__(
        self,
        data_in: Dict[str, DataIn],
        processors: List[Processor],
        normalization: Normalization,
        data_out: List[DataOut],
        frequency: int = 10,
    ):
        self.data_in = data_in
        self.processors = processors
        self.normalization = normalization
        self.data_out = data_out

        # auxiliary attributes
        self.frequency = frequency
        self.too_slow_count = 0
        self.filling_buffer = True

    def update(self):
        """
        Fetches new data from the input channels, processes the data and
        outputs the processed data to the output channels.
        """
        # fetch raw data
        if any(d.update() == -1 for d in self.data_in.values()):
            return
        elif self.filling_buffer:
            print("done")
            self.filling_buffer = False

        # process raw data (feature extraction)
        processed, intermediates, normalize_mask = {}, {}, {}
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
            out.update(self.data_in, finished)

    def run(self):
        """
        Start the fetching and processing loop and limit the loop to run at a
        constant update rate.
        """
        print("Filling buffer(s)...", end="")

        last_time = time.time()
        while True:
            # receive, process and output data
            self.update()

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
    import data_in
    import data_out
    import normalization
    import processors

    mngr = Manager(
        data_in={
            "file": data_in.EEGRecording.make_eegbci(),
        },
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Ratio("/file/alpha", "/file/theta", "alpha/theta"),
            processors.Biocolor(channels={"file": ["C3"]}),
            processors.Biotuner(channels={"file": ["C3", "C4", "O1", "O2"]}),
        ],
        normalization=normalization.StaticBaselineNormal(duration=30),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5005),
            data_out.PlotRaw("file"),
            data_out.PlotProcessed(),
        ],
    )

    mngr.run()
