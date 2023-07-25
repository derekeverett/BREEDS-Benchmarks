# Creating BREEDS Sub-population shift Benchmarks

This repository contains the modified code for the paper:

**BREEDS: Benchmarks for Subpopulation Shift** <br>
*Shibani Santurkar\*, Dimitris Tsipras\*, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2008.04859 <br>

![](pipeline.png)

```bibtex
    @InProceedings{santurkar2020breeds,
        title={BREEDS: Benchmarks for Subpopulation Shift},
        author={Shibani Santurkar and Dimitris Tsipras and Aleksander Madry},
        year={2020},
        booktitle={ArXiv preprint arXiv:2008.04859}
    }
```

## Getting started

1.  Clone the repo

2.  Install dependencies:
    ```
    conda create -n breeds-benchmarks python=3.7 pip
    conda activate breeds-benchmarks
    pip install -r requirements.txt
    conda install pygraphviz
    ```
3.  Download the [ImageNet](http://www.image-net.org/) dataset.
4.  Use the `make_symlinks_datasets.py` script to create BREEDS datasets.
