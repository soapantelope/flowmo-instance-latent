## Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization

This repo contains the code for our Neurok2D experiments

<p align="center">
  <img src="demo.gif" alt="sample GIF" />
</p>

## Get the code
```
git clone https://github.com/kylesargent/FlowMo
cd FlowMo
```

## Install the requirements
```
conda create -n FlowMo python=3.13.2 pip
conda activate FlowMo
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Training
Example training command (config files are in flowmo/configs):
```
torchrun --nproc-per-node=1 -m flowmo.train --config-path path_to_config_file --experiment-name exp_name 2>&1 | tee logs/exp_name.log
```