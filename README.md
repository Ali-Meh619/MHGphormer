# Metapath-based Heterogeneous Graph-Transformer Network (MHGphormer)

This is the official Pytorch implementation of the Metapath-based Heterogeneous Graph-Transformer Network (MHGphormer), which is proposed in our paper ["Joint Spectrum, Precoding, and Phase Shifts Design for RIS-Aided Multiuser MIMO THz Systems"](https://people.ece.ubc.ca/vincentw/J/MW-TCOM-2024.pdf) accepted for publication in *IEEE Transactions on Communications*, 2024.

## Installation

First, check the requirements as follows:\
python\
numpy\
pytorch



Then clone the repository as follows:
```shell
git clone https://github.com/Ali-Meh619/MHGphormer.git
```

## Description

The file "System Setup and Generating Dataset" contains the code for simulation parameters, creating the system environment and generating the training and test datasets.\
The file "MHGphormer Architecture" contains the code for the MHGphormer learning algorithm.\
The file "Loss Function and Training" contains the code for designing the loss function based on the optimization problem and the training process.\
The file "Execution" contains the code for running the model in the training and test phases.


## Citation

If you find our paper and code useful, please kindly cite our paper as follows:
```bibtex
@article{b16,
  title={Joint spectrum, precoding, and phase shifts design for {RIS}-aided multiuser {MIMO} {TH}z systems},
  author={Mehrabian, Ali and Wong, Vincent W. S.},
  journal={IEEE Trans. Commun.},
 volume={72},
  number={8},
  pages={5087-5101},
month={Aug.},
  year={2024}
}
```

## Contact

Please feel free to contact us if you have any questions:
- Ali Mehrabian: alimehrabian619@{ece.ubc.ca, yahoo.com}

