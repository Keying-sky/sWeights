# S1 Coursework
### Keying Song

In this coursework, a package named *fit_methods* was created for the the comparison between the extended multi-dimensional likelihood fit and the weighted fit exploiting *sWeights*.

## Declaration
No auto-generation tools were used in this coursework except for generation of BibTeX references.

## Project Structure
The main structure of *fit_methods* is like:
```
.
├── fit_methods/
│   ├── bootstrap/         # bootstrap analysis implementation
│   ├── distribution/      # p.d.f.s' definition, verification and visualisation
│   ├── fitting/           # the fitter for EML fitting
│   ├── generation/        # the data generator
│   └── sweight/           # sWeights analysis implementation
├── pyproject.toml         
├── README.md            
└── usage_examples.ipynb   # the main file to answer the questions
```

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/s1_coursework/ks2146.git
```

2. Install:
```bash
pip install -e .
```

3. Use:
After installing, all the classes in package 's1_coursework' can be imported and used anywhere on your own machine.
```python
from fit_methods import PDF, Generator, Fitter, Bootstrap, Sweightor
```

## Usage

The main workflow is demonstrated in `usage_examples.ipynb`. The five sections in it address each of the five questions from (b) to (f) in the coursework.

Here is a short example of usages:
```python
from fit_methods import PDF, Generator, Fitter, Bootstrap, Sweightor

params = [3, 0.3, 1, 1.4, 0.6, 0.3, 0, 2.5]

# Generate data
generator = Generator(params)
x, y = generator.generate_sample(100000)

# EML fitting
fitter = Fitter(x, y)
result = fitter.fit()

# Bootstrapping
study = Bootstrap(params)
results = study.toy_study()
study.uncertainties(results)  

# sWeights
x, y = Generator(params).generate_sample(100000)
result = Sweightor(x, y).do_sWeight(params[5])
Sweightor(x, y).plot_results(result, params)

```

## Dependencies
- numpy
- scipy
- iminuit
- iminuit.cost
- matplotlib
- sweights

