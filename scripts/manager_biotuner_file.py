from goofi import manager, data_in, data_out, normalization, processors

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            # "muse": data_in.EEGRecording.make_eegbci(),
            # "muse": data_in.EEGStream("Muse00:55:DA:B0:49:D3")
            "file": data_in.EEGRecording.make_eegbci(),
        },
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Ratio("/file/alpha", "/file/theta", "alpha/theta"),
            processors.Biotuner(channels={"file": ["O1", "O2"]}),
            processors.Biocolor(channels={"file": ["O1", "O2"]}),
        ],
        normalization=normalization.StaticBaselineNormal(duration=30),
        data_out=[
            data_out.OSCStream("10.0.0.255", 5070),
            # data_out.PlotRaw("file"),
            data_out.PlotProcessed(),
        ],
        frequency=5,
    )

    mngr.run()
