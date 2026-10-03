# Frugal Flows

This repository is the official implementation of *Marginal Causal Flows for Inference and Validation*, accepted at NeurIPS 2024.

# Set-up the environment

## Micromamba

Environment requirements to run the paper experiments can be found in the ```environment.yaml``` file. This file can be used to set up an environment with any environment manager e.g., venv, Conda, Mamba, Micromamba. With Micromamba, you can create and activate the environment as follows:
```
micromamba create -f environment.yaml

micromamba activate <name-environment>
```
This will automatically install the Frugal Flows package together with its dependencies and all the Python packages required to run the experiments in the paper. 

Some experiments require additional R packages. To install these, please refer to the section "Rerunning Experiments".

## Rerunning Experiments
Rerunning all the experiments presented in the paper requires the installation of `rpy2` and other relevant R libraries. To install these, please run:
```
python install_rpy2_libraries.py
``` 

## Pip install Frugal Flows

Alternatively, you can solely install the Frugal Flow package using `pip`:

```
git clone <URL-repository>

cd frugal-flows

pip install -e ./
 
```

The dependencies of ```frugal-flows``` can be found in the ```pyproject.toml``` file.



# Multivariate outcomes: the Gaussian-scale frugal flow

For multivariate outcomes (images, many columns), use the Gaussian-scale frugal flow (`frugal_flows.gaussian_scale`). It is the frugal flow written on a standard-normal scale:
- a causal margin p(y | do(t)), a spline flow conditioned on the treatment;
- a copula flow for the covariates given the outcome.

```python
import jax.random as jr
import frugal_flows as ff

# Y: (n, K) outcomes, Z: (n, d) covariates, T: (n,) binary treatment
flow, ot, info = ff.fit_gaussian_frugal_flow(jr.key(0), Y, Z, T)          # standardises Y, ranks Z, fits

s = ff.interventional_samples(jr.key(1), flow, 1, 5000, outcome_transform=ot, dim_y=Y.shape[1])
ate = s["ate"]                                                          # (K,) on the original Y scale

y_cf = ot.inverse(ff.counterfactual_gaussian(flow, ot.forward(Y), T, 1 - T))   # unit-level counterfactuals

ff.save_gaussian_flow("my_fit", flow, info["build_kwargs"], outcome_transform=ot)
flow, ot = ff.load_gaussian_flow("my_fit")
```

- **The model, its settings and its known limitation:** [docs/gaussian_scale/README.md](./docs/gaussian_scale/README.md). The limitation is a treatment-blind copula.
- **Pretrained example fits** of every model on one MorphoMNIST 8×8 dataset: [examples/morphomnist_8x8_n5000](./examples/morphomnist_8x8_n5000/).
- **The MorphoMNIST experiments:** [validation/morphomnist/README.md](./validation/morphomnist/README.md).

`pip install -e .` installs the core package. `pip install -e ".[validation,fid]"` adds what the MorphoMNIST runners and the realism (FID) scorer need. Python 3.11 or later; flowjax is pinned to 19.1.0.

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

To recover Causal Flows ATE values reported in Table 1:
* Clone our [forked causal-flow repository](https://github.com/llaurabatt/causal-flows.git)
* Build your environment from the ```environment.yaml``` file
* Run ```run.sh``` to reproduce experiments
* Run ```ate_FF_loop.ipynb``` to produce ATE values

# Acknowledgement

This repository is developed mainly based on the [FlowJAX](https://github.com/danielward27/flowjax/tree/main) repository. Many thanks to its contributors!
