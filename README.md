# st_analysis

This repository implements a patch-based structure tensor analysis workflow for 2D histology slides stored as OME-Zarr. It computes local orientation and anisotropy per patch, extracts dominant orientation peaks per patch, and reconstructs full-resolution stitched maps.

# Installation

 A Linux machine with pixi installed. Pixi will manage all Python dependencies and non-python dependencies (c3d, greedy, ANTS) through conda environments.

## Steps
 1. Install pixi (if not already installed):
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```
    
 2. Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/vmliu366/st_analysis.git
    cd st_analysis
    pixi install
    ```


# Usage

 1. Perform a dry run:
    ```bash
    pixi run snakemake -np
    ```
 2. Run the app using all cores:
    ```bash
    pixi run snakemake --cores all
    ```

# Contributing
 We welcome contributions! Please refer to the [contributing guidelines](CONTRIBUTING.md) for more details on how to contribute.

# License
 This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## TO-DOs
- [ ] Add in OD maps
- [ ] Add in 3D registration 
- [ ] Fix stitching artifacts (global normalization)
- [ ] Implement lazy loading per patch 
- [ ] Reorganize for clean structure 
