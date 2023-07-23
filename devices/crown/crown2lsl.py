from os.path import dirname, exists, join

from neurosity import NeurositySDK

cred_file = join(dirname(__file__), "neurosity.cred")
if not exists(cred_file):
    raise FileNotFoundError(
        "Please create a file called neurosity.cred in the goofi-pipe/devices/crown "
        "directory, containing Neurosity email, password and device ID in separate "
        "lines."
    )

with open(cred_file, "r") as f:
    email, password, device_id = map(str.strip, f.readlines())

neurosity = NeurositySDK({"device_id": device_id})

neurosity.login({"email": email, "password": password})


def callback(data):
    print("data", data)


unsubscribe = neurosity.brainwaves_raw(callback)
