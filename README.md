# Frugal Flows

This repository is the official implementation of *Marginal Causal Flows for Inference and Validation*, submitted to NeurIPS 2024.

# Set-up the environment

Environment requirements to run the paper experiments can be found in the ```environment.yaml``` file. This file can be used to set up an environment with any environment manager e.g., venv, Conda, Mamba, Micromamba. With Micromamba, you can create and activate the environment as follows:
```
micromamba create -f environment.yaml

micromamba activate <name-environment>
```
This will automatically install the Frugal Flows package together with its dependencies and all the other packages required to run the experiments in the paper.

# Install Frugal Flows

Alternatively, you can solely install the Frugal Flow package:

```
git clone <URL-repository>

cd deep-copula-frugal

pip install -e ./
 
```

The dependencies of ```frugal-flows``` can be found in the ```pyproject.toml``` file.

# Reproduce paper experiments

To reproduce Table 1:

To reproduce Figure 3:

To reproduce Figure 4:

To reproduce Table 3 in the Appendix:


# Acknowledgement

This repository is developed mainly based on the [FlowJAX](https://github.com/danielward27/flowjax/tree/main) repository. Many thanks to its contributors!
