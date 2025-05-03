# Meningioma/preprocessing/tools/skull_stripping submodule

This submodule contains various implementations of the *skull stripping* or *brain extraction (BE)* accross different neuroimaging plataforms. This submodule server as proxy to adapt the input formats of the pipeline coded in this project to the expected input of these tools. Two skull-stripping method have been implemented for now:

1. `ants_bet.py`. ANTs provides a script-based approach for extracting the brain from a T1-weighted image by registering it to a standard template/atlas that has an associated brain probability map. In Nipype, this is encapsulated by the BrainExtraction interface. Key inputs include:
   - anatomical_image (your subject’s T1 or T2 volume).
   - brain_template (the reference template, typically a T1-weighted atlas).
   - brain_probability_mask (a probability map or mask for the same template that indicates likely brain vs. non-brain).
    
    ANTs uses these to do a registration, then transfer the template’s “brain prior” onto your image and refine the boundary. Typical Files for T1 Brain Extraction include     Template (e.g. T1.nii.gz from the [ANTs templates repository](https://github.com/ntustison/TemplateBuildingExample), like “OASIS” T1 template or “IXI” T1 template). Brain Probability Mask, which is the same template, but a separate file that indicates the probability of each voxel being inside the brain (range 0–1). Provided along with the template (e.g. T1_BrainProbabilityMask.nii.gz) in the same coordinate space.
2. `fsl_bet.py`. Provided by [S.M. Smith. Fast robust automated brain extraction. Human Brain Mapping, 17(3):143-155, November 2002.](https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/tutorial_packages/centos6/fsl_507/doc/wiki/BET.html). This is included in the nipype FSL tools wrapper. 