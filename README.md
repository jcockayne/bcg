# BCG
BCG provides an implementation of the Bayesian Conjugate Gradient method from the paper by Cockayne et. al. which can be found [here](www.jcockayne.com/papers/linear_solvers).

## Installation

The package depends on the C++ linear algebra library `eigen3`. On Ubuntu systems this can be installed with `sudo apt install libeigen3-dev`, and on OSX with `brew install eigen3` (via [Homebrew](brew.sh)).

It requires Python 3 and the following libraries, all of which can be installed with `pip install`:
* `numpy`
* `scipy`
* `Cython`
* `eigency`

To install, ensure the above dependencies are installed. Then clone the repository and type ```make install```. This will compile the C++ code and install the library in the active virtualenv.

## Usage

Example usage can be found in the Jupyter notebook in `notebooks/Demo.ipynb`

## Credits
This library is written and maintained by [Jon Cockayne](www.joncockayne.com).
