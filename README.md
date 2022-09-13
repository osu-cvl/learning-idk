# Learning When to Say "I Don't know"

This repo contains the official code for the paper ["Learning When to Say "I Don't Know""](https://arxiv.org/abs/2209.04944) by Nicholas Kashani Motlagh, [Jim Davis](http://web.cse.ohio-state.edu/~davis.1719/), Tim Anderson, and Jeremy Gwinnup, which was accepted to the International Symposium on Visual Computing (ISVC) 2022.

We propose a new Reject Option Classification technique to identify and remove regions of uncertainty in the decision space for a given neural classifier and dataset. Such existing formulations employ a learned rejection (remove)/selection (keep) function and require either a known cost for rejecting examples or strong constraints on the accuracy or coverage of the selected examples. We consider an alternative formulation by instead analyzing the complementary reject region and employing a validation set to learn per-class softmax thresholds. The goal is to maximize the accuracy of the selected examples subject to a natural randomness allowance on the rejected examples (rejecting more incorrect than correct predictions). This repo contains code used to compute per-class thresholds given precomputed validation logits and targets from a pretrained model.

## Overview 

The contents of this repo are organized as follows:
* [threshold.py](threshold.py): a sample script for determining per-class thresholds using the proposed approach.
* [synth_logits/](synth_logits/): a directory of logits extracted from small trained neural networks.
* [temperature_scaling.py](temperature_scaling.py): a class that implements temperature scaling (taken from [another repo](https://github.com/osu-cvl/calibration/tree/main/temperature_scaling)).

## Main Requirements
* Matplotlib
* NumPy
* PyTorch
* SciPy
* statsmodels

with specific versions given in [requirements.txt](requirements.txt). To reproduce the environment using conda run ```conda create -c conda-forge -c pytorch -n <environment-name> --file requirements.txt```.

## Learning Thresholds

An example command to run our thresholding algorithm is:

```
python threshold.py \
    --data_path <path to validation logits file> \
    --test_data_path <path to test logits (optional)> \
    --delta .05 \
    --thresh_func 'b_cdf'
```

where 

* ```data_path``` is the path to validation logits extracted from a pretrained network. These logits will be used to learn per-class thresholds.

The above command-line argument is the only required one to run our algorithm. The following argumenst are optional.

* ```delta``` is the user-defined significance level used in the BinomialCDF algorithm. The default is ```0.05```.
* ```thresh_func``` is the method used to check the viability of the reject region. It must be one of (b_cdf, wilson, wilson_cc, clopper-pearson, agresti_coull). The default is ```b_cdf```.
* ```test_data_path``` is the path to test logits extracted from a pretrained network. These logits will be used to evaluate per-class thresholds. The default is ```None```.
* ```threshold_path``` is the path to save the tensor of thresholds. The default is ```thresholds.pt```.
* ```synth``` is a boolean flag indicating that data_path contains synthetic data (formatted slightly differently). The default is ```False```.
* ```skip_ts``` is a boolean flag that specifies whether to skip temperature scaling before learning thresholds. The default is ```False```.

## Synthetic Data

You can run the algorithm on synthetic data using 

```
python threshold.py \
    --data_path synth_logits/val_logits_v<#>.pt \
    --test_data_path synth_logits/test_logits_v<#>.pt \
    --delta .05 \
    --thresh_func 'b_cdf'
    --synth
```

where <#> corresponds to the number in the paper (1-8). Remember to set the ```--synth``` flag.

## Citation

Please cite our paper "Learning When to Say 'I Don't Know'" with

```
@article{KashaniMotlagh2022,
  title={Learning When to Say "I Don't Know"},
  author={Kashani Motlagh, Nicholas, Davis, Jim, Anderson, Tim, and Gwinnup, Jeremy},
  journal={International Symposium on Visual Computing},
  year={2022}
}
```
