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

# STAR Data
## STAR Dataset Schema
The `STAR` dataset consists of 11,598 observations and 47 variables detailing various aspects of student and teacher demographics, performance, and school characteristics from kindergarten through 3rd grade.

### Constant Variables Across All Grades
* `gender`: Factor indicating student's gender.
* `ethnicity`: Factor indicating student's ethnicity with levels "cauc" (Caucasian), "afam" (African-American), "asian" (Asian), "hispanic" (Hispanic), "amindian" (American-Indian) or "other".
* `birth`: Student's birth quarter (of class year).

### Grade-specific Variables
Each of the following sets of variables has an entry for kindergarten (`k`), 1st grade (`1`), 2nd grade (`2`), and 3rd grade (`3`).

#### STAR Class Type
* `stark`, `star1`, `star2`, `star3`: Factor indicating the STAR class type: regular, small, or regular-with-aide. NA indicates that no STAR class was attended.

#### Academic Performance Scores
* `readk`, `read1`, `read2`, `read3`: Total reading scaled score.
* `mathk`, `math1`, `math2`, `math3`: Total math scaled score.

#### Socio*economic and School Environment
* `lunchk`, `lunch1`, `lunch2`, `lunch3`: Factor indicating whether the student qualified for free lunch.
* `schoolk`, `school1`, `school2`, `school3`: Factor indicating school type: "inner-city", "suburban", "rural", or "urban".

#### Teacher Characteristics
* `degreek`, `degree1`, `degree2`, `degree3`: Factor indicating the highest degree of the teacher: "bachelor", "master", "specialist", or "phd".
* `ladderk`, `ladder1`, `ladder2`, `ladder3`: Factor indicating teacher's career ladder level: "level1", "level2", "level3", "apprentice", "probation" or "pending".
* `experiencek`, `experience1`, `experience2`, `experience3`: Years of teacher's total teaching experience.
* `tethnicityk`, `tethnicity1`, `tethnicity2`, `tethnicity3`: Factor indicating teacher's ethnicity with levels "cauc" (Caucasian), "afam" (African-American), or "asian" (Asian).

#### School and System Identification
* `systemk`, `system1`, `system2`, `system3`: Factor indicating school system ID.
* `schoolidk`, `schoolid1`, `schoolid2`, `schoolid3`: Factor indicating school ID.

## Lalonde Dataset
### Schema
The treatment assignment indicator is the first variable of the data frame: treatment (1 = treated; 0 = control). The subsequent features are:
* age, measured in years (continuous)
* education, measured in years (ordinal/continuous);
* black, indicating race (1 if black, 0 otherwise);
* hispanic, indicating race (1 if Hispanic, 0 otherwise);
* married, indicating marital status (1 if married, 0 otherwise);
* nodegree, indicating high school diploma (1 if no degree, 0 otherwise);

The outcomes are:
* re74, real earnings in 1974 (continuous)
* re75, real earnings in 1975 (continuous)
* re78, real earnings in 1978 (continuous)