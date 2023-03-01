# How gradient estimator variance and bias impact learning in neural networks
This repository provides scripts for our work presented in:

A. Ghosh, Y. H. Liu, G. Lajoie, K. Körding, B. A. Richards. _How gradient estimator variance and bias impact learning in neural networks_. International Conference on Learning Representations 2023. 

[Link to paper](https://openreview.net/forum?id=EBC60mxBwyw)

# Overview of Repository
- [Fig1](Fig1): Scripts to replicate Fig 1 in paper, i.e. train VGG-16 on CIFAR-10 with varying amounts of variance and bias in gradient estimates.

- [Student-Teacher](Student-Teacher): Scripts to replicate Fig 3-6 in paper, i.e. student-teacher setup experiments for both feedforward MLP and VGG-xx networks. 

- [Generalization](Generalization): Scripts to replicate Fig 7 in paper, i.e. verifying claims about generalization performance.

# Corresponding Authors

Please do not hesitate to contact us if you have any questions related to the use of these scripts:

[Arna Ghosh](mailto:arna.ghosh@mail.mcgill.ca)

McGill University & Mila-Quebec AI Institute, Montreal, QC, Canada


[Blake A. Richards](mailto:blake.richards@mcgill.ca)

McGill University, Mila-Quebec AI Institute & Montreal Neurological Institute, Montreal, QC, Canada

CIFAR Learning in Machines & Brains, Toronto, ON, Canada



# Citing
Please use the following bibtex item to cite our paper:

```
@inproceedings{ghosh2023gradient,
  title={How gradient estimator variance and bias impact learning in neural networks},
  author={Ghosh, Arna and Liu, Yuhan Helena and Lajoie, Guillaume and K{\"o}rding, Konrad and Richards, Blake Aaron},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

# License

This project is licensed under the [MIT License](LICENSE)

# Acknowledgement

This work was enabled by the material support of NVIDIA in the form of computational resources and support provided by [Mila](https://mila.quebec/en/).
