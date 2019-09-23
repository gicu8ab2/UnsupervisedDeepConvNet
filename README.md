# This codebase is a mix of Matlab and Python code used to study
techniques to make neural networks robust to adversarial attacks.  I
demonstrate how soft label coding and Bayesian Nonparametric
classification stages can be used to improve the performance of the
AllConvNet against fgsm attacks using the RadioML dataset The
unpublished paper that goes with this code is contained in the docs
folder (ICLR_submission_092718.pdf) of this repo.  I use the "All
Convolution model" described in

'Striving for Simplicity: The All Convolutional Net' The original paper can be found [here](https://arxiv.org/abs/1412.6806#).

I use the Bayesian nonparametric mixture of factor analyzers which can be found
(here)(http://people.ee.duke.edu/~lcarin/BCS.html) and is documented in the pdfs
Chen2010_CS_on_manifolds.pdf and MFA_code_doc.pdf in docs folder.



## Requirements

- Python2.X/3.X
- Matlab
- keras with Tensorflow backend (keras version 1.0.4 or later)
- h5py 
- numpy

## External data

The code can be applied to any dataset including image data, but in
this repo we are using the RadioML dataset.

You can Download the dataset from [here](https://www.deepsig.io/datasets)

or

from shared google drive folder 
[here](https://drive.google.com/drive/folders/1Cm_FUWedA0ewDSA3AZ2kdYu-KokA_-ua?usp=sharing)

## Usage

From a Matlab prompt run the following scripts (changing the
variable path_to_venv_python to point to the correct python path);

1) train_robust_DCNN.m

2) test_weak_DCNN_fgsm_attack.m

3) test_robust_DCNN_fgsm_attack.m

4) test_robust_DCNN_no_attack.m. 