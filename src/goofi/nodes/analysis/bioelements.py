import numpy as np
import pandas as pd
from os.path import join, dirname
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Bioelements(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "elements": DataType.TABLE,
            "spec_regions": DataType.TABLE,
            "types": DataType.TABLE,
        }

    def config_params():
        return {
            "bioelements": {
                "tolerance": FloatParam(0.5, 0.01, 5),
            }
        }

    def setup(self):
        # load the dataframe here to avoid loading it on startup
        self.air_elements = pd.read_csv(join(self.asset_path, "air_elements_filtered.csv"))
    
    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        tolerance = self.params["bioelements"]["tolerance"].value
        elements, spec_regions, types = bioelements_realtime(
                                                            data.data,
                                                            self.air_elements,
                                                            tolerance
                                                        )

        return {
            "elements": (elements, data.meta),
            "spec_regions": (spec_regions, data.meta),
            "types": (types, data.meta),
        }


hertz_to_nm_fn, find_matching_spectral_lines_fn = None, None


def bioelements_realtime(data, df, tolerance):
    # import the biotuner function here to avoid loading it on startup
    global compute_biotuner_fn, harmonic_tuning_fn
    if compute_biotuner_fn is None or harmonic_tuning_fn is None:
        from biotuner.bioelements import find_matching_spectral_lines, hertz_to_nm

        hertz_to_nm_fn = hertz_to_nm
        find_matching_spectral_lines_fn = find_matching_spectral_lines

    # run biotuner peak extraction
    peaks_ang = [hertz_to_nm_fn(x) * 10 for x in data]  # convert to Angstrom
    res = find_matching_spectral_lines_fn(df, peaks_ang, tolerance=tolerance)
    elements_count = res["element"].value_counts()
    elements_final = elements_count.index.tolist()
    # take the three most common elements
    elements_final = elements_final[:3]

    # filter the res DataFrame for these three elements
    res_filtered = res[res["element"].isin(elements_final)]

    # Get unique spectrum regions and types for these elements
    spectrum_regions = res_filtered["spectrum_region"].unique().tolist()
    types = res_filtered["type"].unique().tolist()
    return elements_final, spectrum_regions, types
