[![CI](https://github.com//bimorph-mirror-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com//bimorph-mirror-analysis/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh//bimorph-mirror-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh//bimorph-mirror-analysis)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# bimorph_mirror_analysis

This is a python package used for callibrating bimorph mirrors. It requires the generation of data from the bluesky plan located at https://github.com/DiamondLightSource/dodal/blob/230-bimorph-optimisation-plan/src/dodal/plans/bimorph.py.


## Installation

The package can be installed using pip, the python package manager. It is recommended to install it into a python virtual environment. To do this, first create the virtual environment by entering the following command into the command line 
```
python -m venv </path/to/virtual/environment>
```

One this has been executed, activate the virtual environment with 
```
source </path/to/virtual/environment>/bin/activate
```

Once the environment has been activated, any python packages installed with pip will be installed into the virtual environment and not the base installation of python.

Now that we have activated the virtual environemnt we can install the bimorph-mirror-analysis package into it with
```
pip install git+https://github.com/DiamondLightSource/bimorph-mirror-analysis
```

Once this completes, the package will have been installed and can now be used.

When you have finished using the package, the virtual environment can be deactivated using the command 
```
deactivate
```

Any time you wish to use the package, you must reactivate the virtual environment again.
```
source </path/to/virtual/environment>/bin/activate
```


## Usage

Once the packaged has been installed, it can be used from the command line with `bimorph-mirror-analysis`

There are two commands available to be used, `calculate-voltages` and `generate-plots`.

### calculate-voltages

The `calculate-voltages` command is used to calculate the optimal voltages to be applied to the bimorph mirror. It will print the voltages out to the terminal and optioanlly save them in a csv file (when the `--output-path` flag is used). It accepts restrictions to the allowed voltages for the mirror, including a maximum and minimum allowed voltage and the maximum difference in voltage between adjacent actuators.
The default behaviour of the analysis is to treat the first pencil beam scan as the baseline (which has not had the voltage increment applied to any actuator). This can be changed with the `--baseline-voltage-scan` flag which accepts the index (starting from 0) of the baseline pencil beam scan.

It has the optional ability to only consider slit position within a certain range when performing the analysis, which can be accessed with the `--slit-range` flag.
A human readable file containing the pencil beam scan data can be output with the `--human-readable` flag. This produces a csv file where the first column contains the slit positions and the remaining columns each contain the detector data for a pencil beam scan.


### generate-plots

The `generate-plots` command is used to generate plots of the pencil beam scans, the actuator influence functions and a plot of the baseline mirror shape and the predicted mirror shape after the optimal voltages have been applied. It takes the same inputs as the `calculate-voltages` command as well as a required input of an output directory to save the files in.

Examples of each plot can be seen below:
![Image](https://github.com/user-attachments/assets/811a014d-5d0f-4632-94b2-b8177fde7af0)
![Image](https://github.com/user-attachments/assets/189d9681-b8d0-4968-8622-73a6d1d94956)
![Image](https://github.com/user-attachments/assets/be115552-a814-4712-94c2-aa946d758016)


## How are voltages calculated?

Two methods of calculating the optimal voltages are implemented in the package. The first is multiple linear regression (MLR), which does not respect the restrictions provided. The second is an iterative quadratic programming method which uses the SLSQP algorithm, which is able to respect restrictions when optimising the voltages. 

When using the `calculate-voltages` command, the analysis first uses the MLR method and checks whether the voltages respect the constraints supplied. If they do not, it then uses the iterative approach instead and outputs the voltages it returns. 

When using the `generate-plots` command, both approaches are run and the predicted mirror shape using each are plotted in the mirror surface plot. When the MLR method produces voltages that fit the constraits, the sets of voltages for the two methods are the exact same and thereforw the predicted mirror shape for each are too, so the traces overlap exactly.
