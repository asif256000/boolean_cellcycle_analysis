# A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation

This repository contains the driving code for the computation part of the paper titled "A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation". This paper is currently under review in a peer-reviewed journal.

## Description

The primary goal of this paper is to analyze all perturbations of the boolean networks of cell cycles and identify the perturbations that perform better under certain conditions. The code in this repository implements the perturbation approach described in the paper and provides a framework for analyzing the results using graph scores (details in the paper) as a metric.

## Installation

1. Clone the repository: `git clone https://github.com/username/repo-name.git`
2. Install the required libraries: `pip install -r requirements.txt`

## Usage

1. Open the terminal and navigate to the project directory.
2. Open the file `async_perturb.py` and modify the input parameters `organism` and `calc_params` as per your requirement.
3. Run the script: `python async_perturb.py`
4. The output will be available as excel files in the folder `other_results`.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.
