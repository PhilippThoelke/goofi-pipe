<p align="center">
<img src=https://github.com/PhilippThoelke/goofi-pipe/assets/36135990/60fb2ba9-4124-4ca4-96e2-ae450d55596d width="150">
</p>

<h1 align="center">goofi-pipe</h1>
<h3 align="center">Generative Organic Oscillation Feedback Isomorphism Pipeline</h3>

# Installation
## Running the pipeline
In your terminal, make sure you activated the desired Python environment with Python>=3.8. Then, run the following commands:
```bash
pip install git+https://github.com/PhilippThoelke/goofi-pipe@goofi2 # install goofi-pipe, along with its dependencies
goofi-pipe # start the application
```

> **Note**:
> If you want the option to edit goofi-pipe's code after the installation, please follow the _Development_ instructions below.

## Development
In your terminal, make sure you activated the desired Python environment with Python>=3.8, and that you are in the directory where you want to install goofi-pipe. Then, run the following commands:
```bash
git clone git@github.com:PhilippThoelke/goofi-pipe.git # download the repository
cd goofi-pipe # navigate into the repository
git checkout goofi2 # switch to the goofi-pipe 2.0 development branch
pip install -e . # install goofi-pipe in development mode
goofi-pipe # start the application to make sure the installation was successful
```
