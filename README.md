# Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness

Code for the paper [Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness](https://arxiv.org/abs/2310.02832). Preprint is on arXiv.

## Abstract

Effective OOD detection is crucial for reliable machine learning models, yet most current methods are limited in practical use due to requirements like access to training data or intervention in training. We present a novel method for detecting OOD data in deep neural networks based on transformation smoothness between intermediate layers of a network (BLOOD), which is applicable to pre-trained models without access to training data. BLOOD utilizes the tendency of between-layer representation transformations of in-distribution (ID) data to be smoother than the corresponding transformations of OOD data, a property that we also demonstrate empirically for Transformer networks. We evaluate BLOOD on several text classification tasks with Transformer networks and demonstrate that it outperforms methods with comparable resource requirements. Our analysis also suggests that when learning simpler tasks, OOD data transformations maintain their original sharpness, whereas sharpness increases with more complex tasks.

## Run the Experiments

### Get the Experimental Results

- The main OOD detection experiments are run by running <code>python3 run.py</code>.
- The OOD detection experiments using the energy method are run by running <code>python3 run_energy.py</code>.
- The semantic shift experiments are run by running <code>python3 run_shifted-ng.py</code>.
- The background shift experiments are run by running <code>python3 run_shifted-sst.py</code>.

### Analyzing result

After running experiments, the results can be verified and visualized using jupyter notebooks <code>results.ipynb</code> and <code>figures.ipynb</code> respectively.
