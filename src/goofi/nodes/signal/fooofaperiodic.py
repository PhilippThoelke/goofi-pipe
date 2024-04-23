import numpy as np
from fooof import FOOOF

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class FOOOFaperiodic(Node):
    def config_input_slots():
        return {"psd_data": DataType.ARRAY}

    def config_output_slots():
        return {
            "offset": DataType.ARRAY,
            "exponent": DataType.ARRAY,
            "cf_peaks": DataType.ARRAY,
            "cleaned_psd": DataType.ARRAY,
        }

    def config_params():
        return {
            "fooof": {
                "max_n_peaks": IntParam(5, 1, 20, doc="The maximum number of peaks to fit."),
            }
        }

    def process(self, psd_data: Data):
        if psd_data is None or psd_data.data is None:
            return None

        # Create FOOOF object & set its parameters
        fm = FOOOF(max_n_peaks=self.params["fooof"]["max_n_peaks"].value)

        # Extract the PSD and freqs from the Data object
        psd = psd_data.data

        try:
            if psd_data.data.ndim == 1:
                freqs = np.array(psd_data.meta["channels"]["dim0"])
            else:  # if 2D
                freqs = np.array(psd_data.meta["channels"]["dim1"])
        except KeyError:
            raise ValueError("No frequency information. Make sure to pass a power spectrum with frequency information.")

        # Fit FOOOF model
        fm.fit(freqs, psd)

        # Extract aperiodic component and peak parameters
        aperiodic_params = fm.get_params("aperiodic_params")
        peak_params = fm.get_params("peak_params")
        cleaned_psd = fm._spectrum_peak_rm

        # Extracting specific parameters
        offset = aperiodic_params[0]  # Extracting the offset
        exponent = aperiodic_params[-1]  # Extracting the exponent (if there's no knee value)

        # Getting all center frequencies of detected peaks
        cf_peaks = peak_params[:, 0] if peak_params.size else np.array([])

        return {
            "offset": (offset, {}),
            "exponent": (exponent, {}),
            "cf_peaks": (cf_peaks, {}),
            "cleaned_psd": (cleaned_psd, {}),
        }
