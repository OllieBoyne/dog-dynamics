# Estimation of Canine Dynamics from Monocular Video
This repository contains all code used in the project *Estimation of Canine Dynamics from Monocular Video* - a fourth year Master's project. This project worked on estimating ground reaction forces and joint forces in dogs from oridinary video. 

## Layout
```
|--- data - Management and utilisation of produced datasets
|
|--- dataset_production - Used for generating all datasets discussed in project/ECCV
|          |--- unity_management - Used for managing the Unity dog big pack described used in smal_fitter
|          |--- multicam_optimiser - Experimental scripts (not used in final report) on estimating 3D camera positions from Dataset 2
|
|--- dynamics - Dynamic & Kinematic processing and calculations
|
|--- smal - Experimentation on SMAL Model
|
|--- smal_fitter - Separate repo for SMAL/SMBLD fitting to target meshes
|
|--- vis - Visualisation scripts & tools
```

## smal-fitter

This repo was used to generate the Unity shape and pose priors discussed in the report.
