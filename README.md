# From Notebook to Kubeflow Pipeline using Fashion MNIST

This project aims to show how to convert the Fashion MNIST example notebook found on the [Tensorflow website](https://www.tensorflow.org/tutorials/keras/classification) into notebook that can be run using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/). Our hope is that this baseline workflow can be extended to apply to more complex scenarios. For a more detailed explanation of the different components of this notebook, check out the accompanying [blog post](https://www.manceps.com/articles/tutorial/kubeflow-pipelines-tutorial).

## Prerequisites

* We recommend deploying Kubeflow on a system with 16GB of RAM or more. Otherwise, spin-up a virtual machine instance somewhere with these resources (e.g. t2.xlarge EC2 instance).

* A basic understanding of [Tensorflow](https://www.tensorflow.org/overview/) and [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) is helpful but not strictly necessary.

## Installation

1. [Install Kubeflow](https://www.manceps.com/articles/tutorial/how-to-install-kubeflow-on-various-operating-systems) on your local machine.

2. Launch a [notebook server](https://www.kubeflow.org/docs/notebooks/setup/) from the Kubeflow Dashboard.

![alt text](/images/kf-demo-server.png "kf-demo-notebook-server")

3. Once in the Notebook server, launch a new terminal from the menu on the right (New > Terminal).

![alt text](/images/kf-demo-open-terminal.png "kf-demo-open-terminal")

4. In the terminal, download this Notebook from GitHub.

```
$ git clone https://github.com/manceps/manceps-canonical.git
```

5. From there click on *KF_Fashion_MNIST.ipynb* on the notebook server homepage to begin working through the notebook.

## Contributors

* Chris Thompson
* Rui Vasconcelos

## Acknowledgements

Thanks to the folks at Tensorflow for providing the notebook this tutorial is based off of. Also thanks to the folks who have put in the hard work to make Kubeflow Pipelines a reality. There are many more excellent Kubeflow examples available on the [Kubeflow repository](https://github.com/kubeflow/examples).




