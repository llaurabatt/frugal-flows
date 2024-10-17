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

# General Structure
* The main bulk of the frugal flow implementation can be found in [frugal_flows](./frugal_flows/).
* The script containing functions to generate the simulated data for the inference experiments can be found [here](./data/template_causl_simulations.py).
* The main class which allows the user to implement Frugal Flows at ease can be found in [benchmarking.py](./frugal_flows/benchmarking.py)

# Reproduce paper experiments

* To reproduce Table 1: [Continous_Frugal_Flows.ipynb](./validation/Continous_Frugal_Flows.ipynb)
* To reproduce Figure 3: [Lalonde_Data_Pipeline.ipynb](./validation/Lalonde_Data_Pipeline.ipynb)
* To reproduce Figure 4: [e401k_Data_Pipeline.ipynb](./validation/e401k_Data_Pipeline.ipynb)
* To reproduce Table 3 in the Appendix: [Logistic_Sampling.ipynb](./validation/Logistic_Sampling.ipynb)

## To reproduce comparisons to Causal Flows

To recover Causal Flows ATE values reported in Table X:
* Clone our [forked causal-flow repository](https://github.com/llaurabatt/causal-flows.git)
* Build your environment from the ```environment.yaml``` file
* Run ```run.sh``` to reproduce experiments
* Run ```ate_FF_loop.ipynb``` to produce ATE values

# Acknowledgement

This repository is developed mainly based on the [FlowJAX](https://github.com/danielward27/flowjax/tree/main) repository. Many thanks to its contributors!
