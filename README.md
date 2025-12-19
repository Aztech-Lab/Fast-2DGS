# Fast-2DGS
The original implementation of the Fast-2DGS paper 
(will update soon)

## Setup
1. Navigate to projekt folder, Create a new Python environment and install the Image-GS env (double check CUDA path).
    ```bash
    conda create -n 2dgs python=3.12
    conda activate 2dgs
    pip install -r requirements.txt
    git clone https://github.com/Aztech-Lab/gmod.git
    cd gmod
    pip install -e . --no-build-isolation
    cd ..
    ```
2. Clone the datasets from [2DGS_dataset](https://github.com/Aztech-Lab/2DGS_dataset), download [DIV2K_train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and organize the folder structure as follows:
     ```bash
    git clone https://github.com/Aztech-Lab/2DGS_dataset.git
    ```

    ```
    2DGS_dataset
    └── dataset
        ├── Kodak
        └── DIV2K
            └── DIV2K_train_HR (need to download)
            └── DIV2K_valid_HR
        └── ImageGS_anime
        └── ImageGS_textures

    ```





