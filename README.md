# Lau et al. (2022) Implementation Uncertainty
This GitHub repository contains the code and data needed to recreate all figures and replicate the computation experiments for Lau et al. (2022). To cite this paper, please use the following citation:
```
This the the citation for the paper
```

To recreate all figures, run the Python scripts within the `figure_generation` directory. Data from Trindade et al. (2020) can be found in the `results` directory. 

To replicate the cmoputational experiment, follow the steps outlined below. Note that this experiment was run using high performance computing and cannot easily be replicated on a personal computer. To accurately replicate the experiment, please use parallel master-worker version of [Borg MOEA](http://borgmoea.org). You can request access to the source code [here](http://borgmoea.org/#contact). 

*Note: All filepaths in the code files provided should be modified to reflect current individual data and information locations.*

## Folders :file_folder:
1. `figure_generation` Contains all python code files used for generating figures
2. `Figures` Contains all figures included in the paper. Only Figures 5 to 13 are generated using the files in `process_output`.
3. `process_output` Contains all python code files used for post-processing the output of the DU Reevaluation
4. `src` Contains all the files necessary to build WaterPaths

## Setup :hammer:
### Download and compile WaterPaths
1. Clone this repository and unzip all files. In the command line, enter the directory where this repository is stored. 
2. Type `make gcc` into the command line to compile WaterPaths.
### Install Borg MOEA
1. Once access is obtained, clone and unzip the Borg MOEA repository. 
2. Copy the contents of the Borg MOEA repository into the `implementation-uncertainty` directory under a directory called `Borg`.
3. Download and install [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/). Please skip this step if you already have OpenMPI installed. 
4. Access the `Borg` directory via the command line. Enter `ls` into the command line and verify that the directory contains the `borg.exe` file. 
5. Run `make` in the command line. This will install Borg onto the machine you are using. 

## Implementation Uncertainty Analysis :mag:
The figure below illustrates implementation uncertainty sampling scheme.
<p align="center">
<img src="Figures/Fig04_sampling_IU.jpg" width="600">
</p>

### Generate the implementation uncertainty sampling range

### Bootstrap analysis

### Generate ROF tables for bootstrapped realizations

