from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            # "muse": data_in.EEGRecording.make_eegbci(),
            # "muse": data_in.EEGStream("Muse00:55:DA:B0:49:D3")
            "plant": data_in.SerialStream(sfreq=1000, buffer_seconds=5, auto_select=False, port='COM4'),
        },
        processors=[
            #processors.PSD(label="delta"),
            #processors.PSD(label="theta"),
            #processors.PSD(label="alpha"),
            #processors.PSD(label="beta"),
            #processors.PSD(label="gamma"),
            #processors.LempelZiv(),
            #processors.Cardiac(data_type='ppg', extraction_frequency=1),
            #processors.Ratio("/plant/alpha", "/plant/theta", "alpha/theta"),
            processors.Biotuner(channels={"plant": ["serial"]}, n_peaks=5, extraction_frequency=0.1, peaks_function='EMD',
                                harmonic_connectivity=False),
            # processors.Biocolor(
            #     channels={"plant": ["serial"]}, extraction_frequency=0.1
            # ),
            # processors.Bioelements(
            #     channels={"plant": ["serial"]}, extraction_frequency=0.1
            # ),
            # processors.TextGeneration(
            #     processors.TextGeneration.HOROSCOPE_PROMPT,
            #     # "/plant/biocolor/ch0_peak0_name",
            #     "/plant/bioelements/ch0_bioelements",
            #     keep_conversation=False,
            #     read_text=False,
            #     label="horoscope",
            #     update_frequency=0.05,
            # ),
            # processors.TextGeneration(
            #     processors.TextGeneration.POETRY_PROMPT,
            #     "/plant/biocolor/ch0_peak0_name",
            #     "/plant/bioelements/ch0_bioelements",
            #     keep_conversation=False,
            #     read_text=False,
            #     label="poetry",
            #     update_frequency=0.05,
            # ),
        ],
        normalization=normalization.StaticBaselineNormal(duration=5),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5070),
            data_out.PlotRaw("plant"),
            ##data_out.ProcessedToFile("plant_processed_test", overwrite=True),
            #data_out.RawToFile("philodandron_ground_leaf_02.csv", data_in_name='plant', overwrite=False),
            #data_out.PlotProcessed(),
        ],
        frequency=1,
    )

    mngr.run()
