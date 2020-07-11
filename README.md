# dog-dynamics
This repository contains all code used in the project *Estimation of Canine Dynamics from Monocular Video* - a fourth year Master's project. This project worked on estimating ground reaction forces and joint forces in dogs from oridinary video. 

## Layout
```
|
|--- dataset_production - Used for generating all datasets discussed in project/ECCV
|          |--- unity_management - Used for managing the Unity dog big pack described used in smal_fitter
|          |--- 3d_optimisation - Experimental scripts (not used in final report) on estimating 3D camera positions from Dataset 2
|
|--- data - Management and utilisation of produced datasets
|
|--- dynamics - Dynamic & Kinematic processing and calculations
|
|--- smal - Experimentation on SMAL Model
|
|--- vis - Visualisation scripts & tools
| 
|--- smal_fitter - Separate repo for SMAL/SMBLD fitting to target meshes
```

## smal-fitter

This repo was used to generate the Unity shape and pose priors discussed in the report.
