# S1 Coursework
### Keying Song

In this coursework, a comparison between the extended multi-dimensional likelihood fit and the weighted fit exploiting *sWeights* is made step by step.

## Declaration
No auto-generation tools were used in this coursework except for generation of BibTeX references.

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
```bash
from src.distribution import PDF
from src.generation import Generator
from src.fitting import Fitter
from src.sweight import Sweightor
from src.bootstrap import Bootstrap
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

