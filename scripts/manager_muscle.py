from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    mngr = manager.Manager(
        data_in={
            "muscle": data_in.SerialStream(
                sfreq=350, buffer_seconds=5, auto_select=False, port="COM4"
            ),
        },
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
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
            data_out.OSCStream("192.168.0.255", 7008),
            #data_out.PlotRaw("muscle"),
            ##data_out.ProcessedToFile("plant_processed_test", overwrite=True),
            #data_out.RawToFile("guitar_raw_test.csv", data_in_name='muscle', overwrite=True),
            data_out.PlotProcessed(),
        ],
        frequency=5,
    )

    mngr.run()
