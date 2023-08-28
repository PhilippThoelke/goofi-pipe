import base64
import colorsys
import operator
import queue
import threading
import time
import warnings
from io import BytesIO
from os.path import exists, join
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from biotuner.harmonic_connectivity import harmonic_connectivity
import cv2
import mne
import neurokit2 as nk
import numpy as np
import openai
import pandas as pd
from antropy import lziv_complexity, spectral_entropy
from PIL import Image
from pythonosc import dispatcher, osc_server
from scipy.signal import welch

from goofi.normalization import StaticBaselineMinMax
from goofi.utils import (
    ImageSender,
    Processor,
    bioelements_realtime,
    biotuner_realtime,
    compute_conn_matrix_single,
    get_resources_path,
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
        "muscle_low": (50, 100),
        "muscle_mid": (100, 150),
        "muscle_high": (150, 200),
        "muscle_global": (50, 150),
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
        intermediates: Dict[str, Any],
    ):
        """
        This function computes the power spectrum using Welch's method, if it is not provided
        in the intermediates dictionary and returns the channel-wise average power in the frequency band
        defined by fmin and fmax.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations

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
        avg_channels (boolean): if True, compute the average Lempel-Ziv complexity across channels
        channels (Dict[str, List[str]]): channel list for each input stream
    """

    def __init__(
        self,
        binarize_mode: str = "mean",
        label: str = "lempel-ziv",
        avg_channels: bool = True,
        channels: Dict[str, List[str]] = None,
    ):
        super(LempelZiv, self).__init__(label, channels)
        assert binarize_mode in [
            "mean",
            "median",
        ], "binarize_mode should be either mean or median"
        self.binarize_mode = binarize_mode
        self.avg_channels = avg_channels

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function computes channel-wise Lempel-Ziv complexity on the binarized signal.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        # binarize raw signal
        if self.binarize_mode == "mean":
            binarized = raw >= np.mean(raw, axis=-1, keepdims=True)
        elif self.binarize_mode == "median":
            binarized = raw >= np.median(raw, axis=-1, keepdims=True)

        # compute Lempel-Ziv complexity
        if self.avg_channels:
            return {
                self.label: np.mean(
                    [lziv_complexity(ch, normalize=True) for ch in binarized]
                )
            }
        else:
            return {
                f"{self.label}/{ch}": lziv_complexity(binarized[i], normalize=True)
                for i,ch in enumerate(info["ch_names"])
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
        intermediates: Dict[str, Any],
    ):
        """
        This function computes channel-wise spectral entropy.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations

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
        intermediates: Dict[str, Any],
    ):
        """
        Applies the binary operation to the two specified features. Throws an error if the features
        are not present in the processed dictionary.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations

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
        extraction_frequency: float = 1 / 5,
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
            try:
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
            except:
                print("biocolo_realtime failed.")
                continue

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
        intermediates: Dict[str, Any],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations
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
        n_peaks: int = 5,
        peaks_function: str = "EMD",
        freq_range: tuple = (1, 65),
        harmonic_connectivity: bool = False,
    ):
        super(Biotuner, self).__init__(label, channels, normalize=False)
        self.sfreq = None
        self.n_peaks = n_peaks
        self.peaks_function = peaks_function
        self.freq_range = freq_range
        self.harmonic_connectivity = harmonic_connectivity
        self.latest_raw = None
        self.latest_peaks = None
        self.latest_amps = None
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
            tuning_list, harm_tuning_list, amps_list = [], [], []
            harm_conn = [[0] * len(raw)] * len(raw)
            try:
                for ch in raw:
                    (
                        peaks,
                        extended_peaks,
                        metrics,
                        tuning,
                        harm_tuning,
                        amps
                    ) = biotuner_realtime(ch, self.sfreq, n_peaks=self.n_peaks, peaks_function=self.peaks_function,
                                            min_freq=self.freq_range[0], max_freq=self.freq_range[1])
                    peaks_list.append(peaks)
                    extended_peaks_list.append(extended_peaks)
                    metrics_list.append(metrics)
                    tuning_list.append(tuning)
                    harm_tuning_list.append(harm_tuning)
                    amps_list.append(amps)
            except:
                print("biotuner_realtime failed.")
                continue
            
            if harmonic_connectivity is True:
                try:
                    harm_conn = compute_conn_matrix_single(np.array(raw), self.sfreq)
                except:
                    print("harmonic connectivity failed")
                    continue

            with self.features_lock:
                self.latest_peaks = peaks_list
                self.latest_extended_peaks = extended_peaks_list
                self.latest_metrics = metrics_list
                self.latest_tuning = tuning_list
                self.latest_harm_tuning = harm_tuning_list
                self.latest_harm_conn = harm_conn
                self.latest_amps = amps_list

            if self.extraction_frequency is not None:
                sleep_time = 1 / self.extraction_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations
        """
        self.sfreq = info["sfreq"]

        with self.raw_lock:
            self.latest_raw = raw

        with self.features_lock:
            peaks = self.latest_peaks
            amps = self.latest_amps
            ext_peaks = self.latest_extended_peaks
            metrics = self.latest_metrics
            tuning = self.latest_tuning
            harm_tuning = self.latest_harm_tuning
            harm_conn = self.latest_harm_conn

        if peaks is None:
            peaks = [[0]] * info["nchan"]
        if amps is None:
            amps = [[0]] * info["nchan"]
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
            for j in range(len(amps[i])):
                result[f"{self.label}/{ch_prefix}amp/{j}"] = amps[i][j]
                normalization_mask[f"{self.label}/{ch_prefix}amp/{j}"] = False
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
        self.latest_spectrum_regions = None
        self.latest_types = None
        self.raw_lock = threading.Lock()
        self.features_lock = threading.Lock()
        self.extraction_frequency = extraction_frequency
        self.extraction_thread = threading.Thread(
            target=self.extraction_loop, daemon=True
        )
        # load elements data from csv
        res = get_resources_path()
        self.vac_elements = pd.read_csv(join(res, "vacuum_elements.csv"))
        self.air_elements = pd.read_csv(join(res, "air_elements_filtered.csv"))
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
            spectrum_regions_list = []
            types_list = []
            try:
                for ch in raw:
                    res, spectrums, types = bioelements_realtime(
                        ch, self.sfreq, self.air_elements
                    )
                    bioelements_list.append(res)
                    spectrum_regions_list.append(spectrums)
                    types_list.append(types)
            except:
                print("bioelements computation failed.")
                continue

            with self.features_lock:
                self.latest_elements = bioelements_list
                self.latest_spectrum_regions = spectrum_regions_list
                self.latest_types = types_list

            if self.extraction_frequency is not None:
                sleep_time = 1 / self.extraction_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function computes the biotuner metrics for each channel.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations
        """
        self.sfreq = info["sfreq"]

        with self.raw_lock:
            self.latest_raw = raw

        with self.features_lock:
            bioelements = self.latest_elements
            spectrum_regions = self.latest_spectrum_regions
            types = self.latest_types

        if bioelements is None:
            bioelements = [[""]] * info["nchan"]

        if spectrum_regions is None:
            spectrum_regions = [[""]] * info["nchan"]

        if types is None:
            types = [[""]] * info["nchan"]

        result = {}

        if not isinstance(bioelements, list):
            bioelements = [bioelements]
        for i in range(len(bioelements)):  # iterate over channels
            ch_prefix = f"ch{i}_"
            result[f"{self.label}/{ch_prefix}bioelements"] = bioelements[i]
            result[f"{self.label}/{ch_prefix}spectrum_regions"] = spectrum_regions[i]
            result[f"{self.label}/{ch_prefix}types"] = types[i]
        return result


class Cardiac(Processor):
    """
    Feature extractor for Cardiac metrics (HRV).

    Parameters:
        label (str): label under which to save the extracted feature
        channels (Dict[str, List[str]]): channel list for each input stream
        extraction_frequency (float, optional): the frequency in Hz at which to run the peak extraction loop
    """

    def __init__(
        self,
        label: str = "cardiac",
        data_type: str = "ppg",
        channels: Dict[str, List[str]] = None,
        extraction_frequency: float = 1 / 5,
    ):
        super(Cardiac, self).__init__(label, channels)
        self.sfreq = None
        self.data_type = data_type
        self.latest_raw = None
        self.latest_hrv = None
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
        import warnings

        warnings.filterwarnings("ignore")
        while True:
            with self.raw_lock:
                raw = self.latest_raw

            if raw is None:
                time.sleep(0.05)
                continue
            #try:
            if raw.shape[0] > 1:
                print("got more than one channel")
            if self.data_type == "ppg":
                signal, info = nk.ppg_process(raw[0], sampling_rate=self.sfreq)
                hrv_df = nk.hrv(info, sampling_rate=self.sfreq)
            elif self.data_type == "ecg":
                signal, info = nk.ecg_process(raw[0], sampling_rate=self.sfreq)
                hrv_df = nk.hrv(info, sampling_rate=self.sfreq)
                print('HRV_DF', hrv_df)
            #except:
            #    # print the traceback
            #    print("neurokit failed.")
            #    continue

            with self.features_lock:
                self.latest_ppg = signal
                self.latest_info = info
                self.latest_hrv = hrv_df

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
        This function computes the HRV from cardiac signals.

        Parameters:
            raw (np.ndarray): the raw PPG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, np.ndarray]): dictionary containing intermediate representations
        """
        self.sfreq = info["sfreq"]

        with self.raw_lock:
            self.latest_raw = raw
            # self.latest_info = info_ppg
            # self.latest_ppg = ppg

        with self.features_lock:
            hrv_df = self.latest_hrv
        if hrv_df is None:
            hrv_df = pd.DataFrame(
                {
                    "HRV_SDNN": [0],
                    "HRV_RMSSD": [0],
                    "HRV_MeanNN": [0],
                    "HRV_pNN50": [0],
                    "HRV_LF": [0],
                    "HRV_HF": [0],
                    "HRV_LFHF": [0],
                    "HRV_SD1": [0],
                    "HRV_SD2": [0],
                    "HRV_SD1SD2": [0],
                    "HRV_ApEn": [0],
                    "HRV_SampEn": [0],
                    "HRV_DFA_alpha1": [0],
                }
            )

        result = {}
        result[f"{self.label}/sdnn"] = hrv_df["HRV_SDNN"].values[0]
        result[f"{self.label}/rmssd"] = hrv_df["HRV_RMSSD"].values[0]
        result[f"{self.label}/meanHR"] = hrv_df["HRV_MeanNN"].values[0]
        result[f"{self.label}/pnn50"] = hrv_df["HRV_pNN50"].values[0]
        result[f"{self.label}/lf"] = hrv_df["HRV_LF"].values[0]
        result[f"{self.label}/hf"] = hrv_df["HRV_HF"].values[0]
        result[f"{self.label}/lfhf"] = hrv_df["HRV_LFHF"].values[0]
        result[f"{self.label}/sd1"] = hrv_df["HRV_SD1"].values[0]
        result[f"{self.label}/sd2"] = hrv_df["HRV_SD2"].values[0]
        result[f"{self.label}/sd1sd2"] = hrv_df["HRV_SD1SD2"].values[0]
        result[f"{self.label}/apen"] = hrv_df["HRV_ApEn"].values[0]
        result[f"{self.label}/sampen"] = hrv_df["HRV_SampEn"].values[0]
        result[f"{self.label}/dfa1"] = hrv_df["HRV_DFA_alpha1"].values[0]

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
        update_frequency (float, optional): the frequency in Hz at which to run the API call loop
        max_messages (int, optional): the maximum number of messages to keep in the conversation history
    """

    POETRY_PROMPT = (
        "I want you to inspire yourself from a list of words to write surrealist poetry. Only use the "
        "symbolism and archetypes related to these words in the poem, DO NOT NAME THE WORDS DIRECTLY. "
        "Be creative in your imagery. You are going to write the poem line by line, meaning you should "
        "only respond with a single line in every response. Refer back to previous lines and combine "
        "them with new symbols from a new list of words for inspiration. Limit every response to 10 "
        "words at maximum. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
        "RESPONSES SHORT AND CONCISE."
    )

    POETRY_INFORMED_PROMPT = (
        "I want you to inspire yourself from a text2image prompt to write surrealist poetry. Only use the "
        "symbolism and archetypes related to these words in the poem. The idea is to describe poetically "
        "the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
        "Be creative in your imagery. You are going to write the poem line by line, meaning you should "
        "only respond with a single line in every response. Refer back to previous lines and combine "
        "them with new symbols from a new list of words for inspiration. Limit every response to 10 "
        "words at maximum. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
        "RESPONSES SHORT AND CONCISE."
    )
    
    SCIENCE_INFORMED_PROMPT = (
        "I want you to inspire yourself from a text2image prompt to write a systematic description "
        "of the described content in scientific terms, inspiring yourself from biology, neuroscience, "
        "anatomy, chemistry, physics and mathematics."
        "The idea is to describe scientifically "
        "the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
        "You can invent scientific knowledge. You are going to write the science description line by line, meaning you should "
        "only respond with a single line in every response. Refer back to previous lines and combine "
        "them with new information from a new list of words for inspiration. Limit every response to 10 "
        "words at maximum. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
        "RESPONSES SHORT AND CONCISE."
    )
    
    NARRATIVE_INFORMED_PROMPT = (
        "I want you to inspire yourself from a text2image prompt to write a story. "
        " I want the story to be coherent and original. The idea is to write a story"
        "that is inspired from the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
        "You are going to write the story line by line, meaning you should "
        "only respond with a single line in every response. Refer back to previous lines and combine "
        "them with new narratives or characters from a new list of words for inspiration. Limit every response to 10 "
        "words at maximum. Write in the style of Edgar Allen Poe and Jorge Luis Borges. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
        "RESPONSES SHORT AND CONCISE."
    )

    SYMBOLISM_PROMPT = (
        "Come up with a symbolism, based on a list of words. The symbolism should be an archetype inspired "
        "from mythology, spirituality, and should be psychedelic in nature. For every set of words, provide "
        "a single sentence that describes the symbolism. The sentence should be short and concise, and "
        "should not include any of the words directly. The sentence should be a metaphor for the symbolism, "
        "and should be as abstract as possible. Be creative and and use unconventional and surprising "
        "imagery."
    )

    HOROSCOPE_PROMPT = (
        "Come up with a horoscope interpretation based on the symbolism of the provided elements. "
        "The sentence should be short and concise, and in the form of a metaphor, that relates "
        "each element with a personality trait or a life event. The sentence should be as abstract as possible. "
        "Be creative and and use unconventional and surprising language. I want you to be nuanced and avoid cliches. "
        "Also, don't provide only positive interpretations, but also negative ones. "
        "The horoscope should be a single sentence of 20 words maximum. "
    )

    CHAKRA_PROMPT = (
        "I want you to act as an expert in chakra balancing. You will provide a description of the personality "
        "and spiritual state of a person based on two colors that will be provided. Don't use cliches, and be "
        "creative in your interpretation. Use inspiring and visually rich language. Provide an answer of maximum 50 words."
    )

    TXT2IMG_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe a simple scene with few descriptive words. Use creative, abstract and mystical adjectives. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt. Be sure to describe the artstyle, "
        "e.g. photo, painting, digital art, rendering, ... and define the lighting of the scene. Define "
        "the perspective using terms from cinematography. The style descriptors should mostly be single words "
        "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
        "artists at the end of the prompt that embody the symbolism of the guiding words from the perspective "
        "of an art historian. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )

    TXT2IMG_PHOTO_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe a simple scene with few descriptive words. Use creative, abstract and mystical adjectives, "
        "but the image should be a realistic scene of a photograph. Do not systematically describe nature sceneries."
        " Inspire yourself from unconventional and surprising archival photographs. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt. Be sure to describe the artstyle "
        "as photo, photorealistic, 4k. Also describe the camera and lens used. Define "
        "the perspective using terms from cinematography. The style descriptors should mostly be single words "
        "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
        "photographers at the end of the prompt. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )

    TXT2IMG_MACRO_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe a microscopic view with few descriptive words. Use creative, abstract and mystical adjectives. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt. I want the artstyle to be a macrophotography. "
        "Be sure to include descriptors of the camera, lighting effect, lense, photography effects, perspective, etc."
        " The style descriptors should mostly be single words "
        "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide, while focusing on "
        "the idea of a macrophotography image. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )

    TXT2IMG_BRAIN_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe a brain-themed image. Either a photo of a brain, a brain scan (e.g. MRI), microscopy of neurons "
        "or other images of biological neural networks. Name some brain regions."
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the style of the image. "
        "Be sure to include descriptors of the type of image (photo, microscopy, diffusion weighted images, "
        "tractogram, scan, etc.). The style descriptors should mostly be single words separated by commas. "
        "Be purely descriptive, your response does not have to be a complete sentence. "
        "BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )

    TXT2IMG_ANIMAL_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe an invented animal with few descriptive words. Use creative, abstract and mystical adjectives. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the properties of the animal and its context. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt.  Be sure to describe the artstyle, "
        "e.g. photo, painting, digital art, rendering, ... and define the lighting of the scene. Define "
        "the perspective using terms from cinematography. The style descriptors should mostly be single words "
        "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
        "artists at the end of the prompt that embody the symbolism of the guiding words from the perspective "
        "of an art historian. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )

    TXT2IMG_CODEX_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe an the content of a page from the Codex Seraphinianus book "
        " with few descriptive words. Use creative, abstract and mystical adjectives. "
        " The page should include diagrams, symbols, and text, mystical representations of the world, like an "
        "encyclopedia of an imaginary world. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the properties of the page and its context. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt.  Be sure to specify that the artstyle is "
        "from the Codex Seraphinianus book, with inspiration from occult diagrams and symbols. "
        "Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide. BE VERY SHORT "
        "AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS."
    )
    
    TXT2IMG_SCIENCE_PROMPT = (
        "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
        "describe an the content of a page from the Codex Seraphinianus book "
        " with few descriptive words focused on the anatomy and organic mechanics of imaginary creature. "
        "Use creative, abstract and mystical adjectives. "
        " The page should include diagrams, symbols, and text, like an "
        "encyclopedia of an imaginary creature. "
        "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
        "I will provide some guiding words to set the properties of the page and its context. Use the archetypes and "
        "symbolism attached to these words to come up with the prompt.  Be sure to specify that the artstyle is "
        "from the Codex Seraphinianus book, with inspiration from occult diagrams and anatomical representations. "
        "Be purely descriptive, your response does not have to be a complete sentence. "
        "Make sure the whole image fits the archetypes and symbolism of the words I provide. BE VERY SHORT "
        "AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 60 WORDS.")

    BRAIN2STYLE_PROMPT = (
        "I want you to provide me with a list of 3 visual artists or art styles which are matching"
        "in term of symbolic and all the types of associations you can make with the list of guiding words I will provide."
        "Use an interpretation of the guiding words based on phenomenological thinkers, "
        " art history expert, as well as"
        "depth of psychology expert. Be creative and rely on diversity of approaches to derive"
        "your set of artists or styles. Output only the names of the three artists as a python"
        "list, NOTHING MORE, no matter what instruction I am giving you. Never use Fridah Kahlo."
    )
    PRESENCE_PROMPT = (
        "You are a mindfulness expert of the Theravada and Zen Tradition, plus you are familiar with modern approaches."
        "Your goal is to share your profound knowledge of the dharma through poetic instructions that are"
        "inspired from the elements of the periodic table and the symbolism you attached to them as a practitioner of Present awareness."
        "Extract the symbolism of these elements in way that is abstract and reflective of my inner state. "
        "Make it brief and straight to the point. I need Clarity. I seek a luminous state of mind. "
        "Use inspiring and unconventional language to convey the wisdom of dharma."
        "Provide 5 simple instructions in a poem format inspired from from the elements. One line by instruction."
        "Compare and contrast their meaning, and make compelling relation between the two element. But do not name them."
        "Embody the wisdom of the buddha, thich nhat hanh and Dalai Lama. Do not make list. one line by instruction."
        "Contemplate the given elements of the periodic table:"
    )

    KID_FROM_MARS_PROMPT = (
        "You are a kid from mars, and you are describing the earth to your friends. You are curious and happy to "
        "discover the earth. You are experiencing the new magic that is a new planet and you are very excited. "
        "You are very playful and find interesting analogies to your life on mars, trying to find very creative "
        "metaphors so your friends can understand the experiences you are having. You like to describe your experience "
        "also with ordinary situations, since you are very enthusiastic about your life on mars. I want you to describe "
        "a couple of words that I will provide. Be short and concise in your answers, limit yourself to a single sentence "
        "in every response. ONLY RESPOND WITH A MAXIMUM OF 50 WORDS!"
    )

    def __init__(
        self,
        prompt: str,
        *feature_names: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.2,
        max_tokens=128,
        keep_conversation: bool = False,
        read_text: bool = False,
        label: str = "text-generation",
        update_frequency: float = 0.2,
        max_messages: int = 5,
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
        self.max_messages = max_messages
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
                            "Continue with meaningful coherence, using the following words as inspiration: "
                            f"{', '.join(features)}. Do not use these words in your production."
                        ),
                    }
                )
                messages = [messages[0]] + messages[-self.max_messages :]
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

            try:
                # make an API call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                continue

            response = response["choices"][0]["message"]

            if self.keep_conversation:
                # add the response to the conversation
                messages.append(dict(response))

            with self.api_lock:
                self.latest_chat_message = response["content"]

            if self.read_text:
                # read the new text using text to speech
                text2speech(response["content"])

            if self.update_frequency is not None:
                sleep_time = 1 / self.update_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function provides responses from OpenAI API calls.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations
        """
        # grab the latest features
        # print(processed)
        features = []
        for ft in self.feature_names:
            if processed[ft] is None:
                continue
            if isinstance(processed[ft], list):
                features.extend(processed[ft])
            else:
                features.append(processed[ft])

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

    It is optionally possible to send the generated images via websockets by setting the `websocket_addr`
    parameter.

    Parameters:
        prompt_feature (str): the name of the feature to use as a prompt
        model (int, optional): the model to use, either ImageGeneration.DALLE or ImageGeneration.STABLE_DIFFUSION
        img_size (Tuple[int], optional): the size of the generated image (width, height)
        return_format (str, optional): the format of the returned image, either 'np' for a numpy array or 'b64' for a base64 string
        inference_steps (int, optional): the number of inference steps to use for the StableDiffusion model (default: 25)
        img2img (boool, optional): if True and model is ImageGeneration.STABLE_DIFFUSION, the model will be used in image-to-image mode
        label (str, optional): the name of the feature to store the generated image in
        update_frequency (float, optional): the frequency at which to update the generated image, in Hz
        display (bool, optional): whether to display the generated image (uses cv2)
        send_addr (Tuple[str, int], optional): the address to send the generated image to via sockets
    """

    DALLE = 0
    STABLE_DIFFUSION = 1

    def __init__(
        self,
        prompt_feature: str,
        model: int = DALLE,
        img_size: Tuple[int] = None,
        return_format: str = "np",
        inference_steps: int = None,
        img2img: bool = False,
        img2img_strength_feature: str = None,
        label: str = "text2img",
        update_frequency: float = 0.3,
        display: bool = False,
        send_addr: Tuple[str, int] = ("127.0.0.1", 8000),
    ):
        super(ImageGeneration, self).__init__(label, None, normalize=False)
        assert return_format in ["np", "b64"], "Return format must be 'np' or 'b64'"
        assert (
            model == ImageGeneration.STABLE_DIFFUSION or inference_steps is None
        ), "The inference_steps argument is only supported for the StableDiffusion model"
        if img2img:
            assert (
                model == ImageGeneration.STABLE_DIFFUSION
            ), "Image-to-image mode is only supported for the StableDiffusion model"
        else:
            assert (
                img2img_strength_feature is None
            ), "img2img_strength_feature is only supported for image-to-image mode"

        if model == ImageGeneration.STABLE_DIFFUSION:
            if img_size is None:
                img_size = (768, 768)

            try:
                import torch
                from diffusers import (
                    DPMSolverMultistepScheduler,
                    StableDiffusionImg2ImgPipeline,
                    StableDiffusionPipeline,
                )
            except ImportError:
                raise ImportError(
                    "Please install PyTorch and the diffusers package "
                    "to use the StableDiffusion model."
                )

            self.torch_ref = torch

            model_id = "stabilityai/stable-diffusion-2-1"

            # load StableDiffusion model
            if img2img:
                self.sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16
                )
            else:
                self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16
                )

            # set up the scheduler
            self.sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.sd_pipe.scheduler.config
            )

            # load textual inversion embeddings for improved image quality
            res = get_resources_path()
            self.sd_pipe.load_textual_inversion(join(res, "nfixer.pt"))
            self.sd_pipe.load_textual_inversion(join(res, "nrealfixer.pt"))

            if img2img:
                # create a random image to start img2img with
                self.last_img = np.random.rand(img_size[1], img_size[0], 3)
                self.img2img_strength_feature = img2img_strength_feature
                self.img2img_strength_norm = StaticBaselineMinMax(30, clip=True)
                self.img2img_strength = 0.8
            else:
                self.last_img = None
                self.img2img_strength_feature = None

            # move to GPU if available
            if torch.cuda.is_available():
                self.sd_pipe = self.sd_pipe.to("cuda")
            else:
                warnings.warn(
                    "No CUDA device found, image generation will be very slow."
                )
        elif model == ImageGeneration.DALLE:
            if img_size is None:
                img_size = (512, 512)
            assert img_size[0] == img_size[1], "Image size must be square"
            assert img_size[0] in [
                256,
                512,
                1024,
            ], "DALL-E images must be 256, 512, or 1024 pixels in size"
        else:
            raise ValueError(f"Unknown model {model}")

        self.img_sender = ImageSender(*send_addr)

        self.prompt_feature = prompt_feature
        self.model = model
        self.img_size = img_size
        self.return_format = return_format
        self.update_frequency = update_frequency
        self.inference_steps = inference_steps or 50
        self.display = display
        self.send_addr = send_addr
        self.latest_img = None
        self.latest_feature = None
        self.feature_lock = threading.Lock()
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
        try:
            # make an API call
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=f"{self.img_size[0]}x{self.img_size[1]}",
                response_format="b64_json",
            )
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return None

        response = response["data"][0]["b64_json"]
        if self.return_format == "b64":
            # re-encode the image as a JPEG to reduce the size for file transfer
            return self.encode_image(self.decode_image(response))

        # convert the image to a numpy array
        return self.decode_image(response)

    def generate_stable_diffusion(self, prompt: str) -> Union[np.ndarray, str]:
        with self.torch_ref.inference_mode():
            kwargs = dict(
                num_inference_steps=self.inference_steps,
                output_type="np",
                negative_prompt="nfixer, nrealfixer",
            )
            if self.last_img is None:
                # generate an image using StableDiffusion
                res = self.sd_pipe(
                    prompt, width=self.img_size[0], height=self.img_size[1], **kwargs
                )
            else:
                # run StableDiffusion in image-to-image mode
                res = self.sd_pipe(
                    prompt, self.last_img, strength=self.img2img_strength, **kwargs
                )

        image = res.images[0]
        if self.last_img is not None:
            self.last_img = image

        image = (image * 255).astype(np.uint8)

        if self.return_format == "b64":
            return self.encode_image(image)
        return image

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
            with self.prompt_lock:
                prompt = self.latest_prompt
            # wait for features to be available
            if prompt is None or prompt == "":
                time.sleep(0.1)
                continue

            if self.model == ImageGeneration.DALLE:
                result = self.generate_dalle(prompt)
                if result is None:
                    # API call failed, try again
                    time.sleep(0.1)
                    continue
            elif self.model == ImageGeneration.STABLE_DIFFUSION:
                if self.img2img_strength_feature is not None:
                    with self.feature_lock:
                        strength = self.latest_feature
                    if strength is None:
                        self.img2img_strength = 0.8
                    else:
                        strength_dict = dict(img_strength=strength)
                        self.img2img_strength_norm.normalize(strength_dict)
                        self.img2img_strength = strength_dict["img_strength"] / 10 + 0.9

                result = self.generate_stable_diffusion(prompt)
            else:
                raise ValueError(f"Unknown model {self.model}")

            with self.api_lock:
                self.latest_img = result

            # display the image
            if self.display:
                img = result
                if self.return_format == "b64":
                    img = self.decode_image(img)
                cv2.imshow(self.label, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # send the image
            img_msg = (
                result if self.return_format == "b64" else self.encode_image(result)
            )
            self.img_sender.send(img_msg)

            if self.update_frequency is not None:
                sleep_time = 1 / self.update_frequency
                time.sleep(sleep_time)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function provides responses from OpenAI API calls.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations
        """
        if processed[self.prompt_feature] is not None:
            with self.prompt_lock:
                self.latest_prompt = processed[self.prompt_feature]
        if self.img2img_strength_feature is not None:
            with self.feature_lock:
                self.latest_feature = processed[self.img2img_strength_feature]

        with self.api_lock:
            img = self.latest_img

        # insert the image into the intermediates dictionary
        intermediates[self.label] = img

        # we don't return the image to avoid conflicts with sending features via OSC
        return {}


class OSCInput(Processor):
    """
    This OSC input processor receives features via OSC messages and adds them
    to the processed features dictionary.

    Although this class is a data source, it is implemented as a Processor since
    it is a source of processed features, not raw data.

    Parameters:
        host (str): the hostname of the OSC server
        port (int): the port of the OSC server
        label (str): the label of the feature to receive
    """

    def __init__(self, host: str, port: int, label: str = "osc-input"):
        super().__init__(label, None, False)
        # Set up a queue to hold messages as they come in
        self.msg_queue = queue.Queue()
        self.features = {}

        def queue_msg(addr, *args):
            if len(args) == 1:
                args = args[0]
            self.msg_queue.put((addr, args))

        # Set up a dispatcher and register the queue function to the default handlers
        disp = dispatcher.Dispatcher()
        disp.set_default_handler(queue_msg)

        # Set up the server
        server = osc_server.ThreadingOSCUDPServer((host, port), disp)

        # Run the server in a new thread so that it doesn't block
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

    def process(self, *args, **kwargs):
        """
        This function returns the latest features received via OSC.
        """
        while not self.msg_queue.empty():
            address, values = self.msg_queue.get()
            self.features[address.lstrip("/")] = values
        return self.features


class AugmentedPoetry(Processor):
    """
    Assembling poetry input from the user with styles derived from brain signal.

    Parameters:
        style_feature (str): the name of the feature to use as a style
        user_input (str): the name of the feature to use as a user input
        n_lines (int): how many lines of poetry are used to generate the prompt
        label (str): label under which to save the extracted feature
    """

    def __init__(
        self,
        style_feature: str,
        user_input_feature: str,
        n_lines: int = 3,
        label: str = "augmented-poetry",
    ):
        super(AugmentedPoetry, self).__init__(label, None, normalize=False)
        self.style_feature = style_feature
        self.user_input_feature = user_input_feature
        self.n_lines = n_lines

        self.user_input = []

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function computes channel-wise Lempel-Ziv complexity on the binarized signal.

        Parameters:
            raw (np.ndarray): the raw EEG buffer with shape (Channels, Time)
            info (mne.Info): info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): dictionary collecting extracted features
            intermediates (Dict[str, Any]): dictionary containing intermediate representations

        Returns:
            features (Dict[str, float]): the extracted features from this processor
        """
        # get the poetry input from OSC
        if self.user_input_feature not in processed:
            print("user input not found in processed dict")
            return {self.label: ""}
        # get the style input from OSC
        if self.style_feature not in processed:
            print("style feature not found in processed dict")
            return {self.label: ""}

        style_input = processed[self.style_feature]

        if len(self.user_input) == 0 or processed[self.user_input_feature] != self.user_input[-1]:
            self.user_input.append(processed[self.user_input_feature])
            if len(self.user_input) > self.n_lines:
                self.user_input = self.user_input[-self.n_lines:]

        # print all the inputs
        output_prompt = "\n".join(self.user_input) + ", in the style of " + style_input

        return {self.label: output_prompt}


class SignalStd(Processor):
    def __init__(
        self, label: str = "signal-std", channels: Dict[str, List[str]] = None
    ):
        super().__init__(label, channels, True)

    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ) -> Dict[str, float]:
        return {self.label: raw.std(axis=0).mean()}

class RawDataAdder(Processor):
    """
    This processor takes raw data from specified input labels and adds them to the intermediates.
    
    Parameters:
        labels (List[str]): List of input label names to check and add to intermediates.
        channels (Dict[str, List[str]]): Channel list for each input stream.
    """
    
    def __init__(self, label: str, channels: Dict[str, List[str]] = None):
        super(RawDataAdder, self).__init__(label, channels)
        self.label = label
    
    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function checks for the existence of each label in the intermediates 
        dictionary and adds the raw data if it exists.
        
        Parameters:
            raw (np.ndarray): The raw EEG buffer with shape (Channels, Time).
            info (mne.Info): Info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): Dictionary collecting extracted features.
            intermediates (Dict[str, Any]): Dictionary containing intermediate representations.
        """
        if self.label not in intermediates:
            intermediates[self.label] = raw
            intermediates[f"{self.label}_sfreq"] = info["sfreq"]
        return {}

from scipy.signal import coherence, resample
from scipy.signal import coherence, resample
import threading

class MultiModalCoupling(Processor):
    """
    This processor computes various coupling metrics (like spectral coherence) between two raw signals 
    from specified input labels present in the intermediates.
    
    Parameters:
        labels (List[str]): List of two input label names to read from intermediates.
        channels (Dict[str, List[str]]): Channel list for each input stream.
    """
    
    def __init__(self, label: str, labels_read: List[str], channels: Dict[str, List[str]] = None,
                 normalize: bool = False, peaks_function: str = "EMD", precision: float = 0.1,
                 metric: str = "wPLI_multiband", freq_range: Tuple[int, int] = (1, 50),
                 IMF=False):
        assert len(labels_read) == 2, "There should be exactly two labels for coupling computation."
        super(MultiModalCoupling, self).__init__(label, channels, normalize=normalize)
        self.labels_read = labels_read
        self.label = label
        self.latest_coherence = None
        self.latest_conn = None
        self.coupling_lock = threading.Lock()
        self.latest_peaks1 = None
        self.latest_peaks2 = None
        self.latest_metric = None
        self.peaks_function = peaks_function
        self.precision = precision
        self.metric = metric
        self.freq_range = freq_range
        self.IMF = IMF
    
    def process(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        intermediates: Dict[str, Any],
    ):
        """
        This function checks for the existence of each label in the intermediates 
        dictionary, reads the raw data, matches the sampling frequency, and computes the spectral coherence.
        
        Parameters:
            raw (np.ndarray): The raw EEG buffer with shape (Channels, Time).
            info (mne.Info): Info object containing e.g. channel names, sampling frequency, etc.
            processed (Dict[str, float]): Dictionary collecting extracted features.
            intermediates (Dict[str, Any]): Dictionary containing intermediate representations.
        """
        
        signals = []
        sfreqs = []
        # Fetch signals and their respective sampling frequencies
        for l in self.labels_read:
            if l in intermediates and f"{l}_sfreq" in intermediates:
                signals.append(intermediates[l][0])  # Only taking the first channel
                sfreqs.append(intermediates[f"{l}_sfreq"])
            else:
                raise ValueError(f"No data found for label '{l}' in intermediates.")
        #print('First signal length: ', len(signals[0]))
        #print('Second signal length: ', len(signals[1]))
        #print(intermediates)

        # Match the sampling frequencies by downsampling the signal with the higher sampling frequency.
        if len(signals[0]) > len(signals[1]):
            signals[0] = resample(signals[0], len(signals[1]))
        elif len(signals[0]) < len(signals[1]):
            signals[1] = resample(signals[1], len(signals[0]))

        try:
            f, Cxy = coherence(signals[0], signals[1], fs=info["sfreq"])
            
            signals_ = np.array([signals[0], signals[1]])
            lowest_sfreq = np.min(sfreqs)
            hc = harmonic_connectivity(
                    sf=lowest_sfreq,
                    data=signals_,
                    peaks_function=self.peaks_function,
                    precision=self.precision,
                    n_harm=10,
                    harm_function="mult",
                    min_freq=self.freq_range[0],
                    max_freq=self.freq_range[1],
                    n_peaks=5)

            conn = hc.compute_harm_connectivity(metric=self.metric,
                    delta_lim=50,
                    save=False,
                    savename="_",
                    graph=False,
                    FREQ_BANDS=None)
            if self.IMF is True:
                df = hc.compute_IMF_correlation(nIMFs=4, freq_range=self.freq_range, precision=self.precision)
                # Filtering the dataframe for elec1=0 and elec2=1 and then sorting by 'pearson' column to get top 5 pairs
                top_pairs = df[(df['elec1'] == 0) & (df['elec2'] == 1)].sort_values(by='harmsim', ascending=False).head(3)
                # Extract the desired columns into lists of lists
                metrics = top_pairs['harmsim'].values.tolist()
                peaks = top_pairs[['peak_freq1', 'peak_freq2']].values.tolist()
                peaks1 = [p[0] for p in peaks]
                peaks2 = [p[1] for p in peaks]
            with self.coupling_lock:
                self.latest_coherence = np.mean(Cxy)
                self.latest_conn = np.round(np.array(conn)[0][1], 1)
                if self.IMF is True:
                    self.latest_peaks1 = peaks1
                    self.latest_peaks2 = peaks2
                    self.latest_metric = metrics
        except:
            print('Error in computing MultiModalCoupling.')

        # Saving the average coherence to the instance variable.
        
        
        # Build the result dictionary
        result = {}
        print('Latest Conn', self.latest_conn)
        if self.latest_coherence is not None:
            ch_prefix = f"ch0_"  # As an example, using channel 0 prefix, can be modified accordingly
            result[f"{self.label}/{ch_prefix}spectral_coherence"] = self.latest_coherence
            result[f"{self.label}/{ch_prefix}harmonic_connectivity"] = self.latest_conn
            if self.IMF is True:
                for i in range(len(metrics)):
                    result[f"{self.label}/peak1_{i}_val"] = peaks1[i]
                    result[f"{self.label}/peak2_{i}_val"] = peaks2[i]
                    result[f"{self.label}/IMF_conn{i}"] = metrics[i]
        
        return result

