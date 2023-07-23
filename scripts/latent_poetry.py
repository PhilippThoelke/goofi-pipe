from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    # configure the pipeline through the Manager class
    mngr = manager.Manager(
        data_in={
            "muse": data_in.EEGRecording.make_eegbci()
            # "muse": data_in.EEGStream("Muse00:55:DA:B5:AB:F7")
        },
        processors=[
            processors.LempelZiv(),
            processors.Biocolor(channels={"muse": ["AF7"]}),
            processors.Bioelements(channels={"muse": ["AF7"]}),
            processors.TextGeneration(
                processors.TextGeneration.BRAIN2STYLE_PROMPT,
                "/muse/biocolor/ch0_peak0_name",
                "/muse/bioelements/ch0_bioelements",
                "/muse/bioelements/ch0_types",
                temperature=1.3,
            ),
            processors.OSCInput("127.0.0.1", 4976),
            processors.AugmentedPoetry(
                style_feature="/muse/text-generation",
                user_input_feature="/muse/user-message",
            ),
            processors.ImageGeneration(
                "/muse/augmented-poetry",
                model=processors.ImageGeneration.STABLE_DIFFUSION,
                img2img=False,
                # img2img_strength_feature="/muse/lempel-ziv",
                img_size=(1280, 720),
                inference_steps=25,
                update_frequency=1 / 10,
            ),
        ],
        normalization=normalization.StaticBaselineNormal(30),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5005),
            # data_out.PlotProcessed(),
        ],
    )

    # start the pipeline
    mngr.run()
