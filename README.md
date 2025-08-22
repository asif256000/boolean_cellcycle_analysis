# A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation

This repository contains the driving code for the computation part of the paper titled "A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation". This paper is currently under review in a peer-reviewed journal.

## Description

The primary goal of this paper is to analyze all perturbations of the boolean networks of cell cycles and identify the perturbations that perform better under certain conditions. The code in this repository implements the perturbation approach described in the paper and provides a framework for analyzing the results using graph scores (details in the paper) as a metric.

## Installation

1. Clone the repository: `git clone https://github.com/asif256000/boolean_cellcycle_analysis`
2. Install the required libraries: `pip install -r requirements.txt`

### GPU Acceleration

This project supports GPU acceleration using JAX for significantly faster performance with large state spaces:

1. Install JAX with GPU support: `pip install "jax[cuda12]"` (for NVIDIA GPUs)
2. For Google TPUs: `pip install "jax[tpu]"`
3. For AMD GPUs: `pip install "jax[rocm]"`
4. For CPU-only: `pip install jax`

Note: JAX GPU support requires compatible hardware and drivers. See the [JAX installation guide](https://github.com/google/jax#installation) for details.

#### Acceleration Features

- **Performance**: Significant speedup for large models
- **Memory optimization**: Efficiently handles very large state spaces
- **Dynamic batch sizing**: Optimizes based on model complexity
- **Fallback mechanism**: Gracefully falls back to CPU when needed

## Usage

### Text Usage

1. Open the terminal and navigate to the project directory.
2. Simulation parameters can be configured in `sim_input/simulation_params.yaml`. Many of these parameters can be overridden using command-line arguments. For advanced settings not exposed via command-line or YAML, you might need to inspect the script `src/async_perturb_test.py`.
3. Run the script from the project root directory: `python src/async_perturb_test.py [options]`.
   Example: `python src/async_perturb_test.py -r single double -o model02 -s 10 -d 5 -g1`
   Below are the available command-line arguments:
   - `--run_options` (`-r`): Specify the type of simulation(s) to run.
     Choices: `original`, `single`, `double`, `discovery`, `perturbation`.
     Can take multiple values (e.g., `-r single double`). Defaults to `original`.
   - `--organism` (`-o`): Select the model for the simulation.
     Choices: `model01`, `model02`, `model03`. Defaults to `model01`.
   - `--single_iter_cnt` (`-s`): Set the number of iterations for single perturbation analysis.
     Defaults to `4`.
   - `--double_iter_cnt` (`-d`): Set the number of iterations for double perturbation analysis.
     Defaults to `2`.
   - `--g1_only_start_states` (`-g1`): Use only G1 states as start states for the simulation.
     This is a flag; add it to enable. Cannot be used with `-c`.
   - `--custom_start_states` (`-c`): Use custom start states defined in the model inputs.
     This is a flag; add it to enable. Cannot be used with `-g1`.
   - `--input_file` (`-i`): Specify the YAML input file for simulation parameters, located in the `sim_input` directory (relative to the project root).
     Defaults to `simulation_params.yaml`.
   - `--discovery_depth` (`-depth`): Define the perturbation discovery depth for new nodes.
     Used with the `discovery` run option. Defaults to `2`.
     You can also run `python src/async_perturb_test.py -h` to see the full help message.
4. The output will be available as excel files in the folder `other_results` (created in the project root if it doesn't exist).

### GUI Usage

1. Open the terminal, navigate to the project directory, and then move to the `src` folder: `cd src`.
2. Simulation parameters can be configured in `../sim_input/simulation_params.yaml` (relative to the `src` directory). Some parameters might be configurable through the GUI popup. For advanced settings not exposed via GUI or YAML, you might need to inspect the script `async_perturb_test.py`.
3. Run the script: `python gui.py`. Edit the options in the GUI popup as needed.
4. The output will be available as excel files in the folder `../other_results` (i.e., in the `other_results` folder at the project root).

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/asif256000/boolean_cellcycle_analysis?utm_source=oss&utm_medium=github&utm_campaign=asif256000%2Fboolean_cellcycle_analysis&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
