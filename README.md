# A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation

This repository contains the driving code for the computation part of the paper titled "A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation". This paper is currently under review in a peer-reviewed journal.

## Description

The primary goal of this paper is to analyze all perturbations of the boolean networks of cell cycles and identify the perturbations that perform better under certain conditions. The code in this repository implements the perturbation approach described in the paper and provides a framework for analyzing the results using graph scores (details in the paper) as a metric.

## Installation

1. Clone the repository: `git clone https://github.com/asif256000/boolean_cellcycle_analysis`
2. Install the required libraries: `pip install -r requirements.txt`

## Usage

1. Open the terminal and navigate to the project directory.
2. Run the script: `python async_perturb_test.py <model_name> <filter_states> <custom_state> <single_it_count> <double_it_count>`
    `<model_name>` represents the model to use. Available models: model01, model02 and model03.
    `<filter_states>` Set True to use filter states.
    `<custom_state>` Set True to use custom states.
    `<single_it_count>` Enter the number of single iterations the program should run.
    `<double_it_count>` Enter the number of double iterations the program should run.
3. The output will be available as excel files in the folder `other_results`.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.
