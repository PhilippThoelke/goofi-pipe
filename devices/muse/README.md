# Muse EEG headset
Simply install `muselsl` in your Python environment:

```bash
pip install muselsl
```

and stream data from the Muse headset to LSL:

```bash
muselsl stream
```

Then you can reference it in goofi-pipe's Manager as
    
```python
from goofi import manager, data_in

manager.Manager(
    data_in={"muse": data_in.EEGStream("name-of-LSL-stream")},
    ...
)
```