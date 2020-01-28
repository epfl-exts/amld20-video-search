
Video search with Face Recognition
---

Talk @ Applied Machine Learning days 2020 - [Google slides](https://docs.google.com/presentation/d/1Jg9rO_3dXwKzJyDOr2ley8Is5oWKE6D_aJJlJrpw0mw)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pacm/video-search) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pacm/video-search/master)

**About Face Recognition**

* dlib C++ ML toolkit with built-in Face Recognition - [GitHub](https://github.com/davisking/dlib)
* TensorFlow.js blazeface model - [online demo](https://storage.googleapis.com/tfjs-models/demos/blazeface/index.html) / [GitHub](https://github.com/tensorflow/tfjs-models/tree/master/blazeface)

**Data Visualization with Machine Learning**

* [Visualizing Data Using t-SNE](https://www.youtube.com/watch?v=RJVL80Gg3lA) presented by Laurens van der Maaten

**Troubleshooting**

If `dlib` fails to import, activate the environment and reinstall it via `conda-forge`

```bash
conda install -c conda-forge dlib
```

When running the notebook on your machine in Jupyter Lab, you will need to activate the `ipywidgets` plugin by running this command in the Conda environment

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**Credits**

* [Bolivian dancers](https://unsplash.com/photos/pLM-A2Wx_0o) by [Milica Spasojevic](https://unsplash.com/@milica_spasojevic)
* [Swiss cow](https://unsplash.com/photos/Vu402lSFOO0) by [Paul Hanaoka](https://unsplash.com/@paul_)
* Model `.dat` files from [dlib-models]( https://github.com/davisking/dlib-models) GitHub repository
