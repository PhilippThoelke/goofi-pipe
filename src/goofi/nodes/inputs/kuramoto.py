import numpy as np
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam, IntParam


class Kuramoto(Node):
    def config_input_slots():
        return {"initial_phases": DataType.ARRAY}

    def config_output_slots():
        return {
            "phases": DataType.ARRAY,
            "coupling": DataType.ARRAY,
            "order_parameter": DataType.ARRAY,
            "waveforms": DataType.ARRAY,
        }

    def config_params():
        return {
            "kuramoto": {
                "coupling_strength": FloatParam(0.5, 0.0, 10.0),
                "natural_frequencies": StringParam("1, 1.5, 2", doc="Comma-separated list of natural frequencies"),
                "timesteps": IntParam(1000, 0, 10000, doc="Number of timesteps to integrate"),
            }
        }

    def process(self, initial_phases: Data):
        # Parse the string to get the natural frequencies
        omega_str = self.params["kuramoto"]["natural_frequencies"].value
        omega = np.array([float(freq.strip()) for freq in omega_str.split(",")])

        N = len(omega)  # Determine the number of oscillators from the length of omega
        K = self.params["kuramoto"]["coupling_strength"].value
        dt = 0.01

        # Number of timesteps to integrate
        timesteps = self.params["kuramoto"]["timesteps"].value

        if initial_phases is None:
            theta = 2 * np.pi * np.random.rand(N)
        else:
            theta = initial_phases.data

        # Store phases at each timestep
        theta_history = np.zeros((N, timesteps))
        time_points = np.linspace(0, dt * timesteps, timesteps)

        # Integrate over the specified number of timesteps
        for i in range(timesteps):
            coupling_term = np.sum(np.sin(theta - theta[:, np.newaxis]), axis=1) / N
            dtheta = dt * (omega + K * coupling_term)
            theta += dtheta
            theta = np.mod(theta, 2 * np.pi)
            theta_history[:, i] = theta

        # Compute actual waveforms from the phase history
        waveforms = np.sin(np.outer(omega, time_points) + theta_history)

        # Calculate the order parameter
        R = np.abs(np.mean(np.exp(1j * theta)))

        channels = {"dim0": [f"oscillator_{i}" for i in range(N)]}
        meta = {"channels": channels}

        return {
            "phases": (theta.reshape(-1, 1), meta),
            "coupling": (coupling_term.reshape(-1, 1), meta),
            "order_parameter": (np.array(R), {}),
            "waveforms": (waveforms, meta),
        }
