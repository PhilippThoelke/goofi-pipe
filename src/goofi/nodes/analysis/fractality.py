import neurokit2 as nk
from goofi.params import StringParam
from goofi.data import Data, DataType
from goofi.node import Node

class Fractality(Node):
    
    def config_input_slots():
        return {"data_input": DataType.ARRAY}

    def config_output_slots():
        return {"fractal_dimension": DataType.ARRAY}

    def config_params():
        return {
            "method": {
                "name": StringParam("fractal_higuchi", options=["fractal_katz", "fractal_petrosian", "fractal_linelength", "fractal_psdslope", "fractal_nld", "fractal_higuchi"])
            }
        }

    def process(self, data_input: Data):
        print('HELLO')
        if data_input or data_input.data is None:
            return None

        method = self.params["method"]["name"].value
        result = None
        print(data_input.data.shape)
        # For methods in neurokit2
        if method == "fracal_katz":
            result = nk.fractal_katz(data_input.data)
        elif method == "fractal_petrosian":
            result = nk.fractal_petrosian(data_input.data)
        elif method == "fractal_linelength":
            result = nk.fractal_linelength(data_input.data)
        elif method == "fractal_psdslope":
            result = nk.fractal_psdslope(data_input.data)
        elif method == "fractal_nld":
            result = nk.fractal_nld(data_input.data)
        elif method == "fractal_higuchi":
            result = nk.fractal_higuchi(data_input.data)
        print(result)
        return {"fractal_dimension": (result, {})}
