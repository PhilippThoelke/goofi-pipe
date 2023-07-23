from goofi import data_in, data_out, manager, normalization, processors

if __name__ == "__main__":
    # configure the pipeline through the Manager class
    mngr = manager.Manager(
        data_in={
            "file": data_in.EEGRecording.make_eegbci()
            # "file": data_in.EEGStream("Muse00:55:DA:B5:AB:F7")
        },
        processors=[
            processors.PSD(label="delta"),
            processors.PSD(label="theta"),
            processors.PSD(label="alpha"),
            processors.PSD(label="beta"),
            processors.PSD(label="gamma"),
            processors.LempelZiv(),
            processors.Biotuner(channels={"file": ["AF7"]}),
            processors.Biocolor(channels={"file": ["AF7"]}),
            processors.Bioelements(channels={"file": ["AF7"]}),
            processors.TextGeneration(
                processors.TextGeneration.TXT2IMG_ANIMAL_PROMPT,
                "/file/biocolor/ch0_peak0_name",
                "/file/bioelements/ch0_bioelements",
                "/file/bioelements/ch0_types",
            ),
            processors.TextGeneration(
                processors.TextGeneration.NARRATIVE_INFORMED_PROMPT,
                #"/file/text-generation"
                "/file/bioelements/ch0_bioelements",
                model="gpt-4",
                keep_conversation=True,
                read_text=True,
                label="poetry",
                update_frequency=1 / 20,
            ),
            processors.ImageGeneration(
                "/file/text-generation",
                model=processors.ImageGeneration.STABLE_DIFFUSION,
                img2img=True,
                img2img_strength_feature="/file/lempel-ziv",
                img_size=(1280, 720),
                inference_steps=15,
                update_frequency=1 / 10,
            ),
        ],
        normalization=normalization.StaticBaselineNormal(30),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5070),
            # data_out.PlotProcessed(),
        ],
    )

    # start the pipeline
    mngr.run()
