from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            "muscle": data_in.SerialStream(
                sfreq=350, buffer_seconds=5, auto_select=False, port="COM4"
            ),
        },
        processors=[
            processors.PSD(label="gamma"),
            processors.PSD(label="muscle_low"),
            processors.PSD(label="muscle_mid"),
            processors.PSD(label="muscle_high"),
            processors.PSD(label="muscle_global"),
            processors.LempelZiv(),
            processors.SpectralEntropy(),
        ],
        normalization=normalization.StaticBaselineNormal(duration=10),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5000),
            data_out.PlotRaw("muscle"),
            ##data_out.ProcessedToFile("plant_processed_test", overwrite=True),
            # data_out.RawToFile("plant_raw_test.csv")
            data_out.PlotProcessed(),
        ],
        frequency=5,
    )

    mngr.run()
