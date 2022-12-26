# Lau et al. (2022) Implementation Uncertainty
This GitHub repository contains the code and data needed to recreate all figures and replicate the computation experiments for Lau et al. (2022). 

To recreate all figures, run the Python scripts within the `figure_generation` directory. Data from Trindade et al. (2020) can be found in the `results` directory. 

To replicate the cmoputational experiment, follow the steps outlined below. Note that this experiment was run using high performance computing and cannot easily be replicated on a personal computer. To accurately replicate the experiment, please use parallel master-worker version of [Borg MOEA](http://borgmoea.org). You can request access to the source code [here](http://borgmoea.org/#contact).

## Folders :file_folder:
1. process_output: Contains all python code files used for post-processing the output of the DU Reevaluation
2. figure_generation: Contains all python code files used for generating figures
3. Figures: Contains the PDFs of all figures generated using code in Folder 2

## Setup :hammer:


## DU Optimization and Re-Evaluation :dart:
The figure below illustrates (a) the DU Optimization and (b) DU Re-Evaluation sampling scheme.
![du_sampling](https://github.com/lbl59/implementation-uncertainty/blob/main/Figures/sampling_DU.jpg raw=True " "IU Sampling")

## Implementation Uncertainty Analysis :mag:
The figure below illustrates implementation uncertainty sampling scheme.
![iu_sampling](https://github.com/lbl59/implementation-uncertainty/blob/main/Figures/sampling_IU.jpg raw=True " "IU Sampling")
### 1. Generate the implementation uncertainty sampling range
### 2. Bootstrap analysis
### 3. Generate ROF tables for bootstrapped realizations
