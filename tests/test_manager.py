from neurofeedback import data_in, data_out, manager, normalization, processors


def test_manager():
    mngr = manager.Manager(
        {"dummy": data_in.DummyStream()},
        [processors.LempelZiv()],
        normalization.WelfordsZTransform(),
        [data_out.OSCStream("127.0.0.1", 5005)],
    )
    mngr.run(n_iterations=10)
