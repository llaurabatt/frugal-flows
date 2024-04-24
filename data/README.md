# Data Processing
We outline the data processing and validationg steps used in the `Frugal Flows` paper.

## Simulated Data
The create_sim_data.py script, located in the data directory, is a Python script designed to generate simulated data for both continuous and discrete treatments and outcomes. The generated data is confounded, meaning that the treatment and outcome variables are influenced by common causes.

### Main Features
* Confounded Data Generation: The script generates confounded data by creating a correlation matrix that defines the relationships between the variables. This matrix is used to generate samples from a multivariate Gaussian distribution, which are then transformed using the standard Gaussian univariate CDF to create the confounded data.
* Continuous and Discrete Treatments/Outcomes: The script supports the generation of both continuous and discrete treatments and outcomes. This is controlled by the TREATMENT_TYPE and OUTCOME_TYPE variables, which can be set to 'C' for continuous or 'D' for discrete.
* Flexible Data Generation: The script uses a variety of distributions (Bernoulli, Poisson, Gamma, and Normal) for the generation of the confounding variables. The weights for the propensity score and the outcome can be adjusted to control the influence of the confounding variables on the treatment and outcome.

### Usage
To use the create_sim_data.py script, simply run it from the command line after installing the `frugal_flows` package. The script will generate four datasets, each representing a different combination of treatment and outcome types (continuous/continuous, continuous/discrete, discrete/continuous, and discrete/discrete), and save them as CSV files in the current directory. The names of the output files are data_xcyc.csv, data_xcyd.csv, data_xdyc.csv, and data_xdyd.csv, respectively.
```
>>> ls
README.md          create_sim_data.py 
>>> python create_sim_data.py
>>> ls
README.md          create_sim_data.py data_xcyc.csv      data_xcyd.csv      data_xdyc.csv      data_xdyd.csv
```


