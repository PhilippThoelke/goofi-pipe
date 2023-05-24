import time
from typing import Dict

import numpy as np
from utils import Normalization


class WelfordsZTransform(Normalization):
    """
    Applies a z-transform to the extracted features. Each step, mean and standard deviation
    are updated according to Welford's algorithm to estimate variance.

    Parameters:
        biased_std (bool): if True, use biased standard deviation 1 / n instead of 1 / (n-1)
        outlier_stds (float): reject values outside outlier_stds standard deviations from the mean
    """

    INIT_STEPS = 50

    def __init__(self, biased_std: bool = False, outlier_stds: float = 4):
        super(WelfordsZTransform, self).__init__()
        self.biased = biased_std
        self.outlier_stds = outlier_stds
        self.count = 0
        self.mean = {}
        self.m2 = {}

    def transform(self, key, val):
        if self.count < 2:
            return 0
        n = self.count if self.biased else self.count - 1
        return (val - self.mean[key]) / (np.sqrt(self.m2[key] / n) + 1e-8)

    def normalize(self, processed: Dict[str, float]):
        """
        Updates mean and M2 according to Welford's algorithm and applies a z-transform to
        the extracted features.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        self.count += 1

        for key, val in processed.items():
            # initialize running stats
            if key not in self.mean:
                self.mean[key] = val
            if key not in self.m2:
                self.m2[key] = 0

            # update running stats according to Welford's algorithm
            if (
                abs(self.transform(key, val)) < self.outlier_stds
                or self.count < self.INIT_STEPS
            ):
                delta = val - self.mean[key]
                self.mean[key] += delta / self.count
                delta2 = val - self.mean[key]
                self.m2[key] += delta * delta2

            # normalize current feature
            processed[key] = self.transform(key, val)

    def reset(self):
        self.count = 0
        self.mean = {}
        self.m2 = {}


class StaticBaselineMinMax(Normalization):
    """
    Records a static baseline for all features (min and max value) and returns features normalized
    to be between 0 and 1 afterwards.

    Parameters:
        duration (float): duration of the baseline in seconds
        clip (bool): if True, clips values to the range [0, 1] after establishing the baseline
    """

    def __init__(self, duration: float, clip: bool = False):
        super(StaticBaselineMinMax, self).__init__()
        self.duration = duration
        self.start_time = None
        self.clip = clip
        self.vmin = {}
        self.vmax = {}

    def normalize(self, processed: Dict[str, float]):
        """
        Updates min and max estimates if currently recording the baseline, otherwise
        normalizing the incoming features using min and max.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        if self.start_time is None:
            print("Recording baseline...", end="")
            self.start_time = time.time()

        for key, val in processed.items():
            if time.time() - self.start_time < self.duration:
                # initialize min and max estimates
                if key not in self.vmin:
                    self.vmin[key] = val
                if key not in self.vmax:
                    self.vmax[key] = val

                # update min and max estimates
                if val < self.vmin[key]:
                    self.vmin[key] = val
                if val > self.vmax[key]:
                    self.vmax[key] = val
            elif self.start_time > 0:
                print("done")
                self.start_time = -1

            # normalize current feature
            if self.vmin[key] < self.vmax[key]:
                val = (val - self.vmin[key]) / (self.vmax[key] - self.vmin[key])
            if self.clip:
                val = np.clip(val, 0, 1)

            processed[key] = val

    def reset(self):
        """
        Resets the min and max values for all features and acquires a new baseline.
        """
        self.start_time = None
        self.vmin = {}
        self.vmax = {}


class StaticBaselineNormal(Normalization):
    """
    Records a static baseline for all features (mean and standard deviation) and returns features
    normalized to be have zero mean and unit standard deviation afterwards.

    Parameters:
        duration (float): duration of the baseline in seconds
    """

    def __init__(self, duration: float):
        super(StaticBaselineNormal, self).__init__()
        self.duration = duration
        self.start_time = None
        self.vals = {}
        self.means = None
        self.stds = None

    def normalize(self, processed: Dict[str, float]):
        """
        If currently recording the baseline, store feature values to compute mean and standard
        deviation, after recording simply normalize the features to have zero mean and unit
        standard deviation.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        if self.start_time is None:
            print("Recording baseline...", end="")
            self.start_time = time.time()

        for key, val in processed.items():
            if time.time() - self.start_time < self.duration:
                # initialize feature lists
                if key not in self.vals:
                    self.vals[key] = []

                # add current feature to list
                self.vals[key].append(val)
            elif self.start_time > 0:
                print("done")
                self.start_time = -1

                # compute mean and std
                self.means = {k: np.mean(self.vals[k]) for k in processed.keys()}
                self.stds = {k: np.std(self.vals[k]) for k in processed.keys()}
                self.vals = None

            # normalize current feature
            if self.means is None:
                val = (val - np.mean(self.vals[key])) / (np.std(self.vals[key]) + 1e-8)
            else:
                val = (val - self.means[key]) / self.stds[key]

            processed[key] = val

    def reset(self):
        """
        Resets the mean and stds for all features and acquires a new baseline.
        """
        self.start_time = None
        self.vals = {}
        self.means = None
        self.stds = None
