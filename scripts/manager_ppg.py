from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            # "muse": data_in.EEGRecording.make_eegbci(),
            # "muse": data_in.EEGStream("Muse00:55:DA:B0:49:D3")
            "heart": data_in.SerialStream(sfreq=1000, buffer_seconds=30, auto_select=False, port='COM4'),
        },
        processors=[
            #processors.Cardiac(channels={"heart": ["serial"]}, data_type='ppg', extraction_frequency=1),
            #processors.Ratio("/plant/alpha", "/plant/theta", "alpha/theta"),
            processors.Biotuner(channels={"heart": ["serial"]}, n_peaks=5, extraction_frequency=0.2, peaks_function='harmonic_recurrence'),
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
        normalization=normalization.StaticBaselineNormal(duration=20),
        data_out=[
            data_out.OSCStream("127.0.0.1", 6000),
            data_out.PlotRaw("heart"),
            ##data_out.ProcessedToFile("plant_processed_test", overwrite=True),
            # data_out.RawToFile("ppg_raw_test.csv", data_in_name='plant', overwrite=True),
            data_out.PlotProcessed(),
        ],
        frequency=5,
    )

    mngr.run()
