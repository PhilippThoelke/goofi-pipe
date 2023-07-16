import base64
import colorsys
import operator
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
from typing import Callable, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import openai
from antropy import lziv_complexity, spectral_entropy
from PIL import Image
from scipy.signal import welch

from neurofeedback.utils import (
    Processor,
    bioelements_realtime,
    biotuner_realtime,
    compute_conn_matrix_single,
    rgb2name,
    text2speech,
    viz_scale_colors,
)


def compute_spectrum(
    x: np.ndarray, info: mne.Info, result: Dict[str, np.ndarray], relative: bool = False
):
    """
    Compute the power spectrum using Welch's method and store the frequency and
    amplitude arrays in result. The power spectrum is only computed on channels
    that are not already present in the result dictionary. The frequency array is
    common to all channels and has the key "freq", amplitudes have the key "spec-<channel>".

    Parameters:
        x (np.ndarray): raw EEG data with shape (Channels, Time)
        info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
        result (Dict[str, np.array]): dictionary in which frequency and amplitude arrays are saved
        relative (bool): if True, compute the relative power distribution (i.e. power / sum(power))
    """
    spec_key = "relspec" if relative else "spec"
    # grab indices of unprocessed channels
    ch_idxs = [
        i for i, ch in enumerate(info["ch_names"]) if f"{spec_key}-{ch}" not in result
    ]
    if len(ch_idxs) > 0:
        # compute power spectrum for unprocessed channels
        result["freq"], specs = welch(x[ch_idxs], info["sfreq"])
        if relative:
            specs /= specs.sum(axis=1, keepdims=True)
        # save new power spectra in results
        result.update(
            {
                f"{spec_key}-{info['ch_names'][i]}": spec
                for i, spec in zip(ch_idxs, specs)
            }
        )


class PSD(Processor):
    """
    Power Spectral Density (PSD) feature extractor.

    Parameters:
        fmin (float): lower frequency boundary (optional if name is inside PSD.band_mapping)
        fmax (float): upper frequency boundary (optional if name is inside PSD.band_mapping)
        relative (bool): if True, compute the relative power distribution (i.e. power / sum(power))
        label (str): name of this feature (if it is one of PSD.band_mapping fmin and fmax are set accordingly)
        channels (Dict[str, List[str]]): channel list for each input stream
    """

    band_mapping: Dict[str, Tuple[float, float]] = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 50),
    }

    def __init__(
        self,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        relative: bool = False,
        label: str = "spectral-power",
        channels: Dict[str, List[str]] = None,
    ):
        super(PSD, self).__init__(label, channels)

        if label in self.band_mapping:
            fmin_default, fmax_default = self.band_mapping[label]
            if fmin is None:
                fmin = fmin_default
            if fmax is None:
                fmax = fmax_default
        elif fmin is None or fmax is None:
            raise RuntimeError(
                f"If label ({label}) is not part of the built-in bands "
                f"({', '.join(self.band_mapping.keys())}), "
                f"fmin ({fmin}) and fmax ({fmax}) can't be None."
            )
        self.fmin = fmin
        self.fmax = fmax
        self.relative = relative

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes the power spectrum using Welch's method, if it is not provided
        in the intermediates dictionary and returns the channel-wise average power in the frequency band
        defined by fmin and fmax.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        # compute power spectral density, skips channels that have been processed already
        compute_spectrum(raw, info, intermediates, relative=self.relative)

        # extract relevant frequencies
        mask = intermediates["freq"] >= self.fmin
        mask &= intermediates["freq"] < self.fmax

        if mask.any():
            # save mean spectral power across frequency bins and selected channels
            spec_key = spec_key = "relspec" if self.relative else "spec"
            return {
                self.label: np.mean(
                    [intermediates[f"{spec_key}-{ch}"][mask] for ch in info["ch_names"]]
                ).item()
            }

        raise RuntimeError(
            f"This frequency band ({self.fmin} - {self.fmax}Hz) has no values inside "
            f"the range of the power spectrum ({intermediates['freq'].min()} - "
            f"{intermediates['freq'].max()}Hz). Consider buffer size and parameters "
            "of the PSD computation."
        )


class LempelZiv(Processor):
    """
    Feature extractor for Lempel-Ziv complexity.

    Parameters:
        binarize_mode (str): the method to binarize the signal, can be "mean" or "median"
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
    """

    def __init__(
        self,
        binarize_mode: str = "mean",
        label: str = "lempel-ziv",
        channels: Dict[str, List[str]] = None,
    ):
        super(LempelZiv, self).__init__(label, channels)
        assert binarize_mode in [
            "mean",
            "median",
        ], "binarize_mode should be either mean or median"
        self.binarize_mode = binarize_mode

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes channel-wise Lempel-Ziv complexity on the binarized signal.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        # binarize raw signal
        if self.binarize_mode == "mean":
            binarized = raw >= np.mean(raw, axis=-1, keepdims=True)
        elif self.binarize_mode == "median":
            binarized = raw >= np.median(raw, axis=-1, keepdims=True)

        # compute Lempel-Ziv complexity
        return {
            self.label: np.mean(
                [lziv_complexity(ch, normalize=True) for ch in binarized]
            )
        }


class SpectralEntropy(Processor):
    """
    Feature extractor for spectral entropy.

    Parameters:
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
    """

    def __init__(
        self,
        label: str = "spectral-entropy",
        channels: Dict[str, List[str]] = None,
    ):
        super(SpectralEntropy, self).__init__(label, channels)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes channel-wise spectral entropy.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        return {
            self.label: spectral_entropy(
                raw, info["sfreq"], normalize=True, method="welch"
            ).mean()
        }


class BinaryOperator(Processor):
    """
    A binary operator applied to two previously extracted features. This can for example
    be used to compute ratios or differences between features or the same feature from
    different channel groups.

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        operation (callable): a binary function returning a float
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(
        self,
        operation: Callable[[float, float], float],
        feature1: str,
        feature2: str,
        label: str = "binary-op",
    ):
        super(BinaryOperator, self).__init__(label, None)
        self.operation = operation
        self.feature1 = feature1
        self.feature2 = feature2

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        Applies the binary operation to the two specified features. Throws an error if the features
        are not present in the processed dictionary.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        if self.feature1 not in processed or self.feature2 not in processed:
            feat = self.feature2 if self.feature1 in processed else self.feature1
            raise RuntimeError(
                f'Couldn\'t find feature "{feat}". Make sure it is extracted '
                f"before this operation is called. Available features: "
                f"{', '.join(processed.keys())}"
            )

        # apply the binary operation and store the result in processed
        result = self.operation(processed[self.feature1], processed[self.feature2])
        return {self.label: result}


class Ratio(BinaryOperator):
    """
    A binary operator to compute the ratio between feature1 and feature2.

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(self, feature1: str, feature2: str, label: str = "ratio"):
        super(Ratio, self).__init__(operator.truediv, feature1, feature2, label)


class Difference(BinaryOperator):
    """
    A binary operator to compute the difference between feature1 and feature2.

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(self, feature1: str, feature2: str, label: str = "difference"):
        super(Difference, self).__init__(operator.sub, feature1, feature2, label)


class Sum(BinaryOperator):
    """
    A binary operator to compute the sum between feature1 and feature2.

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(self, feature1: str, feature2: str, label: str = "sum"):
        super(Sum, self).__init__(operator.add, feature1, feature2, label)


class Product(BinaryOperator):
    """
    A binary operator to compute the product between feature1 and feature2.

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(self, feature1: str, feature2: str, label: str = "product"):
        super(Product, self).__init__(operator.mul, feature1, feature2, label)


class DiffOverSum(BinaryOperator):
    """
    A binary operator to compute the difference over sum of feature1 and feature2, an
    operation also known as SMAPE (Symmetric Mean Absolute Percentage Error).

    It is calculated as (feature1 - feature2) / (feature1 + feature2).

    Note: This Processor requires feature1 and feature2 to already be defined when it is
    called. Make sure to add this Processor only after the Processors for feature1 and
    feature2.

    Parameters:
        feature1 (str): label of the first feature in the binary operation
        feature2 (str): label of the second feature in the binary operation
        label (str): label under which to save the resulting combination
    """

    def __init__(self, feature1: str, feature2: str, label: str = "diff-over-sum"):
        def func(f1, f2):
            return (f1 - f2) / (f1 + f2)

        super(DiffOverSum, self).__init__(func, feature1, feature2, label)


class Biocolor(Processor):
    """
    Feature extractor for biotuner metrics.

    Parameters:
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
        n_peaks (int, optional): number of frequency peaks to extract
        extraction_frequency (float, optional): the frequency in Hz at which to run the peak extraction loop
    """

    FREQ_BANDS = [[1, 3], [3, 7], [7, 12], [12, 18], [18, 30], [30, 45]]

    def __init__(
        self,
        label: str = "biocolor",
        channels: Dict[str, List[str]] = None,
        n_peaks: int = 1,
        extraction_frequency: float = 0.5,
    ):
        super(Biocolor, self).__init__(label, channels, normalize=False)
        self.biotuner = None
        self.latest_raw = None
        self.latest_hsvs = None
        self.n_peaks = n_peaks
        self.raw_lock = threading.Lock()
        self.hsvs_lock = threading.Lock()
        self.extraction_frequency = extraction_frequency
        self.extraction_thread = threading.Thread(
            target=self.peaks_extraction_loop, daemon=True
        )
        self.extraction_thread.start()

    def peaks_extraction_loop(self):
        """
        This function runs the peaks_extraction function in a loop in a separate thread.
        It continuously grabs the latest raw data, processes it, and updates the latest_hsvs.
        """
        while True:
            with self.raw_lock:
                raw = self.latest_raw

            if raw is None:
                time.sleep(0.05)
                continue

            hsvs_list = []
            for ch in raw:
                self.biotuner.peaks_extraction(
                    ch,
                    FREQ_BANDS=self.FREQ_BANDS,
                    ratios_extension=True,
                    max_freq=30,
                    n_peaks=self.n_peaks,
                    graph=False,
                    min_harms=2,
                    verbose=False,
                )

                scale = [1] + self.biotuner.peaks_ratios
                hsvs = viz_scale_colors(scale, fund=self.biotuner.peaks[0])[1:]

                hsvs_list.append(hsvs)

            with self.hsvs_lock:
                self.latest_hsvs = hsvs_list

            if self.extraction_frequency is not None:
                sleep_time = 1 / self.extraction_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        if self.biotuner is None:
            from biotuner import biotuner_object

            self.biotuner = biotuner_object.compute_biotuner(
                info["sfreq"], peaks_function="fixed", precision=0.5, n_harm=5
            )

        with self.raw_lock:
            self.latest_raw = raw

        with self.hsvs_lock:
            latest_hsvs = self.latest_hsvs

        if latest_hsvs is None:
            latest_hsvs = [[[0, 0, 0]] * self.n_peaks] * info["nchan"]

        result = {}
        for i, hsvs in enumerate(latest_hsvs):
            for j, hsv in enumerate(hsvs):
                result[f"{self.label}/ch{i}_peak{j}_hue"] = hsv[0]
                result[f"{self.label}/ch{i}_peak{j}_sat"] = hsv[1]
                result[f"{self.label}/ch{i}_peak{j}_val"] = hsv[2]

                rgb = tuple(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*hsv)))
                result[f"{self.label}/ch{i}_peak{j}_name"] = rgb2name(rgb)
        return result


class Biotuner(Processor):
    """
    Feature extractor for biotuner metrics (peaks, extended peaks, metrics).

    Parameters:
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
        extraction_frequency (float, optional): the frequency in Hz at which to run the peak extraction loop
    """

    def __init__(
        self,
        label: str = "biotuner",
        channels: Dict[str, List[str]] = None,
        extraction_frequency: float = 1 / 5,
    ):
        super(Biotuner, self).__init__(label, channels, normalize=False)
        self.sfreq = None
        self.latest_raw = None
        self.latest_peaks = None
        self.latest_extended_peaks = None
        self.latest_metrics = None
        self.latest_tuning = None
        self.latest_harm_tuning = None
        self.latest_harm_conn = None
        self.raw_lock = threading.Lock()
        self.features_lock = threading.Lock()
        self.extraction_frequency = extraction_frequency
        self.extraction_thread = threading.Thread(
            target=self.extraction_loop, daemon=True
        )
        self.extraction_thread.start()

    def extraction_loop(self):
        """
        This function runs the biotuner_realtime function in a loop in a separate thread.
        It continuously grabs the latest raw data, processes it, and updates the latest_hsvs.
        """
        while True:
            with self.raw_lock:
                raw = self.latest_raw

            if raw is None:
                time.sleep(0.05)
                continue

            peaks_list, extended_peaks_list, metrics_list = [], [], []
            tuning_list, harm_tuning_list = [], []
            try:
                for ch in raw:
                    (
                        peaks,
                        extended_peaks,
                        metrics,
                        tuning,
                        harm_tuning,
                    ) = biotuner_realtime(ch, self.sfreq)
                    peaks_list.append(peaks)
                    extended_peaks_list.append(extended_peaks)
                    metrics_list.append(metrics)
                    tuning_list.append(tuning)
                    harm_tuning_list.append(harm_tuning)
            except:
                print("biotuner_realtime failed.")
                continue

            harm_conn = compute_conn_matrix_single(np.array(raw), self.sfreq)

            with self.features_lock:
                self.latest_peaks = peaks_list
                self.latest_extended_peaks = extended_peaks_list
                self.latest_metrics = metrics_list
                self.latest_tuning = tuning_list
                self.latest_harm_tuning = harm_tuning_list
                self.latest_harm_conn = harm_conn

            if self.extraction_frequency is not None:
                sleep_time = 1 / self.extraction_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        self.sfreq = info["sfreq"]

        with self.raw_lock:
            self.latest_raw = raw

        with self.features_lock:
            peaks = self.latest_peaks
            ext_peaks = self.latest_extended_peaks
            metrics = self.latest_metrics
            tuning = self.latest_tuning
            harm_tuning = self.latest_harm_tuning
            harm_conn = self.latest_harm_conn

        if peaks is None:
            peaks = [[0]] * info["nchan"]
        if ext_peaks is None:
            ext_peaks = [[0]] * info["nchan"]
        if metrics is None:
            metrics = [
                {"harmsim": 0, "cons": 0, "tenney": 0, "subharm_tension": [0]}
            ] * info["nchan"]
        if tuning is None:
            tuning = [[0]] * info["nchan"]
        if harm_tuning is None:
            harm_tuning = [[0]] * info["nchan"]
        if harm_conn is None:
            harm_conn = [[0] * info["nchan"]] * info["nchan"]

        if not isinstance(metrics, list):
            metrics = [metrics]

        result = {}
        normalization_mask = {}
        for i in range(len(metrics)):  # iterate over channels
            ch_prefix = f"ch{i}_"
            result[f"{self.label}/{ch_prefix}harmsim"] = metrics[i]["harmsim"]
            normalization_mask[f"{self.label}/{ch_prefix}harmsim"] = True
            result[f"{self.label}/{ch_prefix}cons"] = metrics[i]["cons"]
            normalization_mask[f"{self.label}/{ch_prefix}cons"] = True
            result[f"{self.label}/{ch_prefix}tenney"] = metrics[i]["tenney"]
            normalization_mask[f"{self.label}/{ch_prefix}tenney"] = True

            if isinstance(metrics[i]["subharm_tension"], list):
                result[f"{self.label}/{ch_prefix}subharm_tension"] = metrics[i][
                    "subharm_tension"
                ][0]
                normalization_mask[f"{self.label}/{ch_prefix}subharm_tension"] = True

            for j in range(len(peaks[i])):
                result[f"{self.label}/{ch_prefix}peak/{j}"] = peaks[i][j]
                normalization_mask[f"{self.label}/{ch_prefix}peak/{j}"] = False
            for j in range(len(ext_peaks[i])):
                result[f"{self.label}/{ch_prefix}extended_peak/{j}"] = ext_peaks[i][j]
                normalization_mask[f"{self.label}/{ch_prefix}extended_peak/{j}"] = False
            for j in range(len(tuning[i])):
                result[f"{self.label}/{ch_prefix}tuning/{j}"] = tuning[i][j]
                normalization_mask[f"{self.label}/{ch_prefix}tuning/{j}"] = False
            for j in range(len(harm_tuning[i])):
                result[f"{self.label}/{ch_prefix}harm_tuning/{j}"] = harm_tuning[i][j]
                normalization_mask[f"{self.label}/{ch_prefix}harm_tuning/{j}"] = False
            for j in range(len(harm_conn[i])):
                result[f"{self.label}/harm_conn/{i}/{j}"] = harm_conn[i][j]
                normalization_mask[f"{self.label}/harm_conn/{i}/{j}"] = False
        return result


class Bioelements(Processor):
    """
    Feature extractor for bioelement matching.

    Parameters:
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
        extraction_frequency (float, optional): the frequency in Hz at which to run the peak extraction loop
    """

    def __init__(
        self,
        label: str = "bioelements",
        channels: Dict[str, List[str]] = None,
        extraction_frequency: float = 1 / 5,
    ):
        super(Bioelements, self).__init__(label, channels, normalize=False)
        self.sfreq = None
        self.latest_raw = None
        self.latest_elements = None
        self.raw_lock = threading.Lock()
        self.features_lock = threading.Lock()
        self.extraction_frequency = extraction_frequency
        self.extraction_thread = threading.Thread(
            target=self.extraction_loop, daemon=True
        )
        self.extraction_thread.start()

    def extraction_loop(self):
        """
        This function runs the biotuner_realtime function in a loop in a separate thread.
        It continuously grabs the latest raw data, processes it, and updates the latest_hsvs.
        """
        while True:
            with self.raw_lock:
                raw = self.latest_raw

            if raw is None:
                time.sleep(0.05)
                continue

            bioelements_list = []
            try:
                for ch in raw:
                    res = bioelements_realtime(ch, self.sfreq)
                    bioelements_list.append(list(res.keys()))
            except:
                print("bioelements computation failed.")
                continue

            with self.features_lock:
                self.latest_elements = bioelements_list

            if self.extraction_frequency is not None:
                sleep_time = 1 / self.extraction_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        self.sfreq = info["sfreq"]

        with self.raw_lock:
            self.latest_raw = raw

        with self.features_lock:
            bioelements = self.latest_elements

        if bioelements is None:
            bioelements = [[""]] * info["nchan"]

        result = {}
        normalization_mask = {}

        if not isinstance(bioelements, list):
            bioelements = [bioelements]
        for i in range(len(bioelements)):  # iterate over channels
            ch_prefix = f"ch{i}_"
            result[f"{self.label}/{ch_prefix}bioelements"] = bioelements[i]
            normalization_mask[f"{self.label}/{ch_prefix}bioelements"] = True
        return result


class TextGeneration(Processor):
    """
    LLM based text generation using features from other processors as inspiration. The text can either
    be built up line by line (if keep_conversation is True), with each line being generated based on
    the previous line and the new features of the current iteration. If keep_conversation is False,
    the conversation is reset with every API call. Text is generated using the OpenAI API.

    This class defines a range of prompts for generation of different types of text.

    Parameters:
        prompt (str): the prompt to use for the API call
        feature_names (str): features used to use inspire the LLM
        model (str): the model to use for the API call, choose from https://platform.openai.com/docs/models/model-endpoint-compatibility
        temperature (float): the temperature parameter for the text generation
        max_tokens (int): the maximum number of tokens to generate
        keep_conversation (bool): whether to keep the conversation history or reset after each generation
        read_text (bool): whether to read the text using text-to-speech
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
        api_call_frequency (float, optional): the frequency in Hz at which to run the API call loop
    """

    POETRY_PROMPT = (
        "I want you to inspire yourself from a list of words to write surrealist poetry. Only use the "
        "symbolism related to these words to construct the poetry, without naming any of the words "
        "directly. Use unconventional words and surprising imagery. Build up a coherent poem with "
        "every response, referring back to previous lines and combining them with new symbols. Only "
        "respond with a single line at a time."
    )

    SYMBOLISM_PROMPT = (
        "Come up with a symbolism, based on a list of words. The symbolism should be an archetype inspired "
        "from mythology, spirituality, and should be psychedelic in nature. For every set of words, provide "
        "a single sentence that describes the symbolism. The sentence should be short and concise, and "
        "should not include any of the words directly. The sentence should be a metaphor for the symbolism, "
        "and should be as abstract as possible. Be creative and and use unconventional and surprising "
        "imagery."
    )

    TXT2IMG_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise but "
        "very detailed. Use many adjectives and use creative, abstract and mystical words. Generate only a "
        "single prompt, which should be as short as possible (max. one long sentence, preferably less). "
        "I will provide some words to inspire the image prompt. Use these words to construct the content of "
        "the image, and use the symbolism associated with these words to construct the style of the image. "
        "To express the style, use terms from visual arts to describe e.g. the image medium (photo, painting, "
        "digital art, ...), the composition, the style, the setting and so on. These style terms should be "
        "single words, separated by commas. The content should be described in a single sentence, and should "
        "be as detailed as possible. Be purely descriptive, your response does not have to be a complete sentence."
    )

    def __init__(
        self,
        prompt: str,
        *feature_names: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.2,
        max_tokens=128,
        keep_conversation: bool = True,
        read_text: bool = False,
        label: str = "text-generation",
        update_frequency: float = 0.25,
    ):
        super(TextGeneration, self).__init__(label, None, normalize=False)
        self.prompt = prompt
        self.feature_names = feature_names
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_conversation = keep_conversation
        self.read_text = read_text
        self.update_frequency = update_frequency
        self.latest_chat_message = None
        self.api_lock = threading.Lock()
        self.latest_features = None
        self.features_lock = threading.Lock()
        self.api_call_thread = threading.Thread(target=self.api_call_loop, daemon=True)
        self.api_call_thread.start()

    def api_call_loop(self):
        """
        This function calls the OpenAI API in a loop to iteratively build up a text, based
        on some features.
        """
        # set the OpenAI API key
        if not exists("openai.key"):
            raise FileNotFoundError(
                "Please create a file called openai.key in the current directory, containing your "
                "OpenAI API key."
            )
        with open("openai.key", "r") as f:
            openai.api_key = f.read().strip()

        # start a conversation with the AI
        system_msg = {"role": "system", "content": self.prompt}
        messages = [system_msg]

        while True:
            try:
                with self.features_lock:
                    features = self.latest_features
                # wait for features to be available
                if features is None:
                    time.sleep(0.1)
                    continue

                # add a user message to the conversation
                if self.keep_conversation:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue with meaningful coherence, using the the following  words as inspiration: "
                                f"{', '.join(features)}. Do not use these words in your production."
                            ),
                        }
                    )
                else:
                    messages = [
                        system_msg,
                        {
                            "role": "user",
                            "content": (
                                f"Generate the text using the following words as inspiration: {', '.join(features)}."
                            ),
                        },
                    ]

                # make an API call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response = response["choices"][0]["message"]

                if self.keep_conversation:
                    # add the response to the conversation
                    messages.append(dict(response))

                with self.api_lock:
                    self.latest_chat_message = response["content"]

                if self.read_text:
                    # read the new text using text to speech
                    text2speech(response["content"])

            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                continue

            if self.update_frequency is not None:
                sleep_time = 1 / self.update_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function provides responses from OpenAI API calls.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        # grab the latest features
        features = []
        for ft in self.feature_names:
            if processed[ft] is None:
                continue
            if isinstance(processed[ft], list):
                features.extend(processed[ft])
            else:
                features.append(processed[ft])
        if len(features) != len(self.feature_names):
            features = None
            warnings.warn(
                f"Could not find all features {self.feature_names} in the processed features."
            )

        if features is not None:
            with self.features_lock:
                self.latest_features = features

        # return the latest chat message
        with self.api_lock:
            return {self.label: self.latest_chat_message}


class ImageGeneration(Processor):
    """
    This processor generates images from text prompts using a locally executed StableDiffusion model
    or the DALL-E model from the OpenAI API.

    Note: Due to the relatively large size of images and limitations from OSC, the image is not returned
    in the processed dictionary, but instead stored in the shared `intermediates` dictionary.

    Parameters:
        prompt_feature (str): the name of the feature to use as a prompt
        model (int, optional): the model to use, either ImageGeneration.DALLE or ImageGeneration.STABLE_DIFFUSION
        img_size (int, optional): the size of the generated image, either 256, 512 or 1024
        return_format (str, optional): the format of the returned image, either 'np' for a numpy array or 'b64' for a base64 string
        label (str, optional): the name of the feature to store the generated image in
        update_frequency (float, optional): the frequency at which to update the generated image, in Hz
    """

    DALLE = 0
    STABLE_DIFFUSION = 1

    def __init__(
        self,
        prompt_feature: str,
        model: int = DALLE,
        img_size: int = 256,
        return_format: str = "np",
        label: str = "text2img",
        update_frequency: float = 0.2,
    ):
        super(ImageGeneration, self).__init__(label, None, normalize=False)
        assert img_size in [256, 512, 1024], "Image size must be 256, 512 or 1024"
        assert return_format in ["np", "b64"], "Return format must be 'np' or 'b64'"

        self.prompt_feature = prompt_feature
        self.model = model
        self.img_size = img_size
        self.return_format = return_format
        self.update_frequency = update_frequency
        self.latest_img = None
        self.api_lock = threading.Lock()
        self.latest_prompt = None
        self.prompt_lock = threading.Lock()
        self.api_call_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.api_call_thread.start()

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        image = Image.fromarray(image)
        buffer = BytesIO()
        image.save(buffer, format="jpeg")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        return np.array(Image.open(BytesIO(base64.b64decode(image))))

    def generate_dalle(self, prompt: str) -> Union[np.ndarray, str]:
        # make an API call
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=f"{self.img_size}x{self.img_size}",
            response_format="b64_json",
        )
        response = response["data"][0]["b64_json"]
        if self.return_format == "b64":
            return response

        # convert the image to a numpy array
        return self.decode_image(response)

    def generate_stable_diffusion(self, prompt: str) -> Union[np.ndarray, str]:
        raise NotImplementedError("StableDiffusion is not yet supported")

    def update_loop(self):
        """
        This function generates images in a loop, using a locally executed StableDiffusion model
        or the DALL-E model from the OpenAI API.
        """
        if self.model == ImageGeneration.DALLE:
            # set the OpenAI API key
            if not exists("openai.key"):
                raise FileNotFoundError(
                    "Please create a file called openai.key in the current directory, containing your "
                    "OpenAI API key."
                )
            with open("openai.key", "r") as f:
                openai.api_key = f.read().strip()

        while True:
            try:
                with self.prompt_lock:
                    prompt = self.latest_prompt
                # wait for features to be available
                if prompt is None:
                    time.sleep(0.1)
                    continue

                if self.model == ImageGeneration.DALLE:
                    result = self.generate_dalle(prompt)
                elif self.model == ImageGeneration.STABLE_DIFFUSION:
                    result = self.generate_stable_diffusion(prompt)

                with self.api_lock:
                    self.latest_img = result

            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                continue

            if self.update_frequency is not None:
                sleep_time = 1 / self.update_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, np.ndarray],
    ):
        """
        This function provides responses from OpenAI API calls.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        if processed[self.prompt_feature] is not None:
            with self.prompt_lock:
                self.latest_prompt = processed[self.prompt_feature]

        with self.api_lock:
            intermediates[self.label] = self.latest_img
        return {}
