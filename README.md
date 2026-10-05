# Connectomic analysis of ORN, OviDN, and CA circuits in BANC

This repository accompanies the paper "Food perception alone stimulates oviposition through serotonin signaling in Drosophila". It includes all code necessary to reproduce the results reported in Fig. 5, Supplemental Figs. 2-4, and Supplemental Video.

## Contributors

* Christian Arntsen - all influence analyses, four-layer circuit reconstruction
* Ayush Shrivastava - shortest paths identification for neurotransmitter enrichment analysis, 3D rendering

## Installation

### This code

```
git clone https://github.com/ayush-shrivastava003/banc-analysis
```

### Dataset

We use the connectome archived on Dataverse by Bates et al. (version 3.0, 7/1/2026): 

- Edge list: [banc_888_edgelist_simple_v3.feather](https://dataverse.harvard.edu/file.xhtml?fileId=13918810&version=3.0)
- Neuron attributes: [banc_888_meta.feather](https://dataverse.harvard.edu/file.xhtml?fileId=14033740&version=3.0)

Dataverse DOI: https://doi.org/10.7910/DVN/7WTH1N

### Dependencies

All code was written in Python. Most required packages are installable via pip:

```
pip install pandas numpy scipy matplotlib navis
```

For influence metrics, we used the [Connectome Influence Calculator](https://github.com/DrugowitschLab/ConnectomeInfluenceCalculator) that accompanied Bates et al.

3D renderings will require installation of [Blender](https://www.blender.org/download/). We recommend running the relevant code on a computing cluster.