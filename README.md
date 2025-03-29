# A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation

This repository contains the driving code for the computation part of the paper titled "A Perturbation Approach for Refining Mathematical Models of Cell Cycle Regulation". This paper is currently under review in a peer-reviewed journal.

## Description

The primary goal of this paper is to analyze all perturbations of the boolean networks of cell cycles and identify the perturbations that perform better under certain conditions. The code in this repository implements the perturbation approach described in the paper and provides a framework for analyzing the results using graph scores (details in the paper) as a metric.

## Installation

1. Clone the repository: `git clone https://github.com/asif256000/boolean_cellcycle_analysis`
2. Install the required libraries: `pip install -r requirements.txt`

## Usage

### Text Usage

1. Open the terminal and navigate to the project directory.
2. Open `async_perturb_test.py` and edit `calc_params` for advanced settings.
3. Run the script: `python async_perturb_test.py <model_name> <-f> <-c> <single_it_count> <double_it_count>`.  
   `<model_name>` represents the model to use. Available models: model01, model02 and model03.  
   `<-f>` Add this flag to use filter states.  
   `<-c>` Add this flag to use custom states.  
   `<single_it_count>` Enter the number of single iterations the program should run.  
   `<double_it_count>` Enter the number of double iterations the program should run.
   You can also run `python async_perturb_test.py -h` to check the help page.
4. The output will be available as excel files in the folder `other_results`.

### GUI Usage

1. Open the terminal and navigate to the project directory and move to the `src` folder.
2. Open `async_perturb_test.py` and edit `calc_params` for advanced settings.
3. Run the script: `python gui.py`. Edit the options in the popup.
4. The output will be available as excel files in the folder `other_results`.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/asif256000/boolean_cellcycle_analysis?utm_source=oss&utm_medium=github&utm_campaign=asif256000%2Fboolean_cellcycle_analysis&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
