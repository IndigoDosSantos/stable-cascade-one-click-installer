## Stable Cascade One-Click Installer

Easy setup for generating beautiful images with Stable Cascade.

### GUI 
1. Follow original installation instructions below (make sure to use a venv)
2. Use `run.bat` for gradio interface

### Troubleshooting
- If you get a torch error run `pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html`
- If you see a protobuf error run `pip install protobuf==3.20.0`
- If you have lots of missing modules open cdm in root folder and activate the venv `conda activate RPG`, copy the content from pip-install.txt and paste it into you cmd. (Yes you can place the whole lot in one copy/paste)

## Stable Cascade One-Click Installer

Easy setup for generating beautiful images with Stable Cascade.

### Updates
- Added seed and steps control
- Added Linux/OS X support (thank you @il-katta)

### Installation ( Windows )

1. **Download**: Get the installer by clicking this link: [Download ZIP](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/archive/refs/heads/main.zip)

2. **Extract**: Unzip the downloaded file to your preferred location.

3. **Install**: Double-click the `install.bat` file to automatically set up all dependencies.

4. **Generate**: Double-click `generate_images.bat` to open the image generation script and get creative!

### Installation ( Linux/OS X)

1. **Download**: Get the installer by clicking this link: [Download ZIP](https://github.com/EtienneDosSantos/stable-cascade-one-click-installer/archive/refs/heads/main.zip)

2. **Extract**: Unzip the downloaded file to your preferred location.

3. **Install**: Execute the `install.sh` script to automatically set up all dependencies.

4. **Generate**: Execute `generate_images.sh` script to start image generation


### Requirements

- This script requires Python to be installed on your system. Python is **not** included in the one-click installer. If you do not have Python installed, download and install it from the [official Python website](https://www.python.org/downloads/) before running the installer.
- Visual Studio build tools for C++ desktop development from the [official Visual Studio website](https://visualstudio.microsoft.com/downloads/)
- Hardware (only tested with 4060 Ti 16 GB VRAM): ![Screenshot of task manager during inference](https://raw.githubusercontent.com/EtienneDosSantos/stable-cascade-one-click-installer/main/hardware_requirements.jpg)

## Acknowledgments

This project makes use of code and models from Stability AI, licensed under the Stability AI Non-Commercial Research Community License Agreement. The full license agreement is available in this repository ([LICENSE-StabilityAI.txt](./LICENSE-StabilityAI.txt)). For more information about Stability AI and their work, visit [their website](https://stability.ai/).
