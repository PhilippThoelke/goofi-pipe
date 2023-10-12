from multiprocessing import shared_memory

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class SharedMemOut(Node):
    def config_params():
        return {"shared_memory": {"name": "goofi-pipe-memory"}}

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        self.shm = None

    def process(self, data: Data):
        if data is None:
            return

        # float32 = 4 bytes
        size = np.prod(data.data.shape) * 4

        if self.shm is None or self.shm.size != size or self.shm.buf is None:
            # create the shared memory
            self.create_shared_memory(size)

        if self.shm is None:
            raise RuntimeError("Shared memory could not be created.")

        # copy the data to the shared memory
        data = data.data.astype(np.float32)
        self.shm.buf[:] = data.tobytes()

    def create_shared_memory(self, size: int):
        if self.shm is not None:
            # close and unlink the existing shared memory
            self.shm.close()
            self.shm.unlink()

        name = self.params.shared_memory.name.value

        try:
            # try to create the shared memory
            self.shm = shared_memory.SharedMemory(name, create=True, size=int(size))
        except FileExistsError:
            # try to close and unlink the existing shared memory
            self.shm = shared_memory.SharedMemory(name, create=False)
            self.shm.close()
            self.shm.unlink()
            # try to create the shared memory again
            self.shm = shared_memory.SharedMemory(name, create=True, size=int(size))

    def shared_memory_name_changed(self, value):
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
