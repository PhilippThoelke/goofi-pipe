<<<<<<< HEAD
from neurofeedback import manager, data_in, data_out, normalization, processors
=======
from goofi import data_in, data_out, manager, normalization, processors
>>>>>>> a09faeb69c954d38cd684840c74e5aa717e1dfe3

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            # "muse": data_in.EEGRecording.make_eegbci(),
            # "muse": data_in.EEGStream("Muse00:55:DA:B0:49:D3")
            "plant": data_in.SerialStream(sfreq=256),
        },
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Ratio("/plant/alpha", "/plant/theta", "alpha/theta"),
            processors.Biotuner(channels={"plant": ["serial"]}),
            processors.Biocolor(
                channels={"plant": ["serial"]}, extraction_frequency=0.1
            ),
            processors.Bioelements(
                channels={"plant": ["serial"]}, extraction_frequency=0.1
            ),
            processors.TextGeneration(
                processors.TextGeneration.HOROSCOPE_PROMPT,
                # "/plant/biocolor/ch0_peak0_name",
                "/plant/bioelements/ch0_bioelements",
                keep_conversation=False,
                read_text=False,
                label="horoscope",
                update_frequency=0.05,
            ),
            processors.TextGeneration(
                processors.TextGeneration.POETRY_PROMPT,
                "/plant/biocolor/ch0_peak0_name",
                "/plant/bioelements/ch0_bioelements",
                keep_conversation=False,
                read_text=False,
                label="poetry",
                update_frequency=0.05,
            ),
        ],
        normalization=normalization.StaticBaselineNormal(duration=30),
        data_out=[
            data_out.OSCStream("10.0.0.255", 5070),
            data_out.PlotRaw("plant"),
            # data_out.PlotProcessed(),
        ],
        frequency=5,
    )

    mngr.run()
