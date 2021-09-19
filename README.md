Canonical 3D Deformable Mapping
==========
Code for *Canonical 3D Deformable Mapping* paper: [arXiv](http://arxiv.org/abs/2008.12709), [web page](http://www.robots.ox.ac.uk/~david/c3dm/).

Installation
-----------
```
git clone https://github.com/facebookresearch/c3dm.git
cd c3dm
conda create -n c3dm python=3.8
```
If you want CUDA support, please follow [the instructions](https://pytorch.org/get-started/locally/) to install Pytorch.
We ran the experiments using the module `torch==1.5.1+cu101`.

All other dependencies can be installed by running `pip`:
```
pip install -e .
```

Dependencies:
- pytorch 1.5.1
- pytorch3d
- pyyaml
- numpy
- PIL
- matplotlib
- visdom
- plotly (visualisation only)
- trimesh (only for metrics)

Running the code
-----------
For evaluation, pass the config name for the dataset, e.g.:
```
cd c3dm
tar -xzf dataset_root.tar.gz
python ./experiment.py freicars.yaml --eval
```
The code should download the required data and pre-trained models.

For training from scratch, make sure there is no model in `c3dm/exp_out`,
otherwise training will continue from it. Then run e.g.
```
python ./experiment.py freicars.yaml
```

License
-----------
The code is released under the [MIT License](LICENSE).


Citation
-----------
David Novotny, Roman Shapovalov, Andrea Vedaldi. Canonical 3D Deformer Maps: Unifying parametric and non-parametric methods for dense weakly-supervised category reconstruction. *NeurIPS 2020.*

Bibtex:
```
@inproceedings{Novotny2020,
    author = {Novotny, David and Shapovalov, Roman and Vedaldi, Andrea},
    booktitle = {NeurIPS},
    title = {{Canonical 3D Deformer Maps: Unifying parametric and non-parametric methods for dense weakly-supervised category reconstruction}},
    url = {http://arxiv.org/abs/2008.12709},
    year = {2020}
}
```
