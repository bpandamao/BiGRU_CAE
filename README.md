# BiGRU_CAE
### A novel stacked hybrid autoencoder for imputing LISA data gaps

The Laser Interferometer Space Antenna (LISA) data stream will contain gaps with missing or unusable data due to antenna repointing, orbital corrections, instrument malfunctions, and unknown random processes.  We introduce a new deep learning model to impute data gaps in the LISA data stream.  The stacked hybrid autoencoder combines a denoising convolutional autoencoder (DCAE) with a bi-directional gated recurrent unit (BiGRU).  The DCAE is used to extract relevant features in the corrupted data, while the BiGRU captures the temporal dynamics of the gravitational-wave signals. 

![BGR-CAE model structure. The left dashed box represents the training of DCAE. The blue line represents the data flow for the training of the hybrid model with BiGRU layers in the decoder. The black dashed line represents the processing of the observed data stream with gaps in our purposed BGR-CAE model.](model_structure.jpg)


Here, we demonstrate the process of training BGR-CAE for the toy model signal with gaps.

We chose a toy model that describes a chirping waveform.

$h(t;a,f,\dot{f},\epsilon) = a \sin (2\pi t[f + 0.5\dot{f}t])$

| parameter | default_value | uniform_training_range|
|-----------|------------|--------------------|
| $\dot{f}$ | $10^{-8}$  | $10^{-12}$|
| $a$ | $5\cdot 10^{-21}$  |$10^{-21}$|
| $f$ | $10^{-3}$  | $10^{-6}$|

## Code structure
**Models**
Contains the method to generate data and gaps, as well as the setting for DCAE and Bi-GRU;

**Training**
Contains two notebooks to train BGR-CAE:
1. to train DCAE: Data are generated by [models/data_generation.py](https://github.com/bpandamao/BGR_CAE/blob/main/models/data_generation.py), make sure to import the correct dataset;
2. to train bi-GRU: Training data can be generated in [training/train_dcae.ipynb](https://github.com/bpandamao/BGR_CAE/blob/main/training/train_dcae.ipynb) or the notebook directly after importing the DCAE model.
   
[training/test_signal_experiment.ipynb](https://github.com/bpandamao/BGR_CAE/blob/main/training/test_signal_experiment.ipynb) gives an example of implementing the model on a signal with gaps.

**Evaluation**
Parameter estimation through MCMC is considered to test the performance of the recovery of the signal with gaps, see [evaluation/parameter_estimation_MCMC.ipynb](https://github.com/bpandamao/BGR_CAE/blob/main/evaluation/parameter_estimation_MCMC.ipynb)


## Get started
1. Install Anaconda if you do not have it.
2. Create a virtual environment using:
```
conda create -n BGR_CAE_trial -c conda-forge numpy scipy matplotlib jupyter torch scikit-learn random corner tqdm statsmodels
conda activate BGR_CAE_trial
```
