# S1 Coursework
### Keying Song

In this coursework, a comparison between the extended multi-dimensional likelihood fit and the weighted fit exploiting *sWeights* is made step by step.

## Project Structure
The main structure of the packages created for this project is like:
```
.
├── src/
│   ├── bootstrap/         # bootstrap analysis implementation
│   ├── distribution/      # p.d.f.s' definition, verification and visualisation
│   ├── fitting/           # the fitter for EML fitting
│   ├── generation/        # the data generator
│   └── sweight/           # sWeights analysis implementation
├── pyproject.toml         # configuration
├── README.md              # readme file
└── usage_examples.ipynb   # the main file to answer the questions
```

## Features
- **Data Generation**: Generate mixed samples of signal and background events
  - Crystal Ball distribution for signal in X
  - Truncated exponential for signal in Y
  - Uniform distribution for background in X
  - Truncated normal for background in Y
- **Fitting Methods**:
  - Extended Maximum Likelihood fitting
  - Parameter estimation with uncertainties
  - Profile likelihood analysis
- **Statistical Analysis**:
  - Bootstrap analysis for parameter estimation
  - sWeights method for background subtraction
  - Uncertainty estimation
  - Performance benchmarking

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install:
```bash
pip install -e .
```

## Usage

The main workflow is demonstrated in `usage_examples.ipynb`. The six sections in it address each of the five questions from (b) to (f) in the coursework.

Here is a short example of usages:
```python
from src.generation import Generator
from src.fitting import Fitter
from src.sweight import Sweightor
from src.bootstrap import Bootstrap, BootstrapResult

# Generate data
generator = Generator(params)
x, y = generator.generate_sample(100000)

# EML fitting
fitter = Fitter(x, y)
result = fitter.fit()

# Bootstrapping
study = Bootstrap(true_params)
results = study.toy_study()
# 5 uncertaintis of lambda with different sample sizes
study.uncertainties(results)  
```

## Dependencies
- numpy
- scipy
- iminuit
- matplotlib

