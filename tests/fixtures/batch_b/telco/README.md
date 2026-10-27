# Telco Customer Churn Predictor

Contributors: 
- Adam Morphy
- Anupriya Srivastava
- Jordan Casoli
- Zihan Zhou

Data Analytics project performed by Group 12 for DSCI 522 (Data Science Workflow), a course in the Master of Data Science program at the University of British Columbia.

## About

For this project we are trying to answer the following question: given certain telecommunications customer characteristics, is a given customer likely to churn? A natural follow up question that we will also be addressing is: what customer characteristics are positively associated with high churn risk? Understanding the answers to these questions will provide actionable insights to telecommunications companies, allowing them to keep their customers for longer periods of time. Ultimately this will lead to higher customer lifetime value. In our project, we are identifying a predictive classification model that can provide insights into which customers (based on their traits) are at higher risk of churning. Answering this question has important practical managerial implications for telecommunication companies because it is a crucial step in understanding how to reduce this risk and thus create higher customer lifetime value. Further, this predictive tool will be considered a contribution to the modern telecommunication customer relationship management systems.


The dataset we are using comes from the public IBM github page, and is made available as part of an effort (by IBM) to teach the public how to use some of their machine learning tools. Unfortunately no mention is made of exactly how the data was collected, or who was responsible for the collection. Here is a link to the [mini-course](https://developer.ibm.com/patterns/predict-customer-churn-using-watson-studio-and-jupyter-notebooks/) that references the dataset we are using. The raw data is [here](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv), and lives inside the data folder of the [public repository](https://github.com/IBM/telco-customer-churn-on-icp4d) for the mini-course. Each row in the dataset corresponds to a single customer. There are 19 feature columns, along with the target column, “churn”.

To begin, we will split the data into train and test sets (80% train/20% test). We will then carry out preliminary EDA on the training data. Specifically, we need to understand whether class imbalance will be an issue in our analysis. Therefore, we will present a table that shows the two class counts. For each of our categorical/binary features, distributions across our two classes will be plotted as stacked bar charts. For each of our numeric features, distributions across our two classes will be plotted as stacked density charts.

At a high level, our plan to answer the predictive question stated above is to build a predictive classification model. We will focus on models that give some indication as to which features are positively or negatively associated with our target class (ex. A logistic regression model), as this is an important aspect of the overall problem. Our primary objective is to be able to identify customers who are at high risk of churning, therefore we will build a model that aims to reduce type II error at the expense of possibly committing more type I error.

We will perform hyperparameter optimization, and then fit the best model on our train data before evaluating the model on our test set. At this point we will assess our final model performance using some combination of recall, precision, roc auc, and average precision. We will present a confusion matrix corresponding to our test results as a table in the final report. Finally, we will present a table showing the features most positively correlated with a high churn risk.

## Report

The final report can be found 
[here](http://htmlpreview.github.io/?https://github.com/adammorphy/Telco_Customer_Churn_Prediction_Group12/blob/main/docs/Telco_Customer_Churn_Prediction_Report.html)

## Usage

There are two suggested ways to run this analysis:

**1. Using Docker**

* note - the instructions in this section also depends on running this in a unix shell (e.g., terminal or Git Bash)

To replicate the analysis, install [Docker](https://www.docker.com/get-started). Then clone this GitHub repository and run the following command at the command line/terminal from the root directory of this project:

**Non Windows Users Command**
```
docker run --rm -v /$(pwd):/home/rstudio/telco_churn_predictor adammorphy/telco_churn_docker:latest make -C /home/rstudio/telco_churn_predictor all
```
**Windows Users Command:** 
```
docker run --rm -v /$(pwd)://home//rstudio//telco_churn_predictor adammorphy/telco_churn_docker:latest make -C //home//rstudio//telco_churn_predictor all
```

To reset the repo to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:

**Non Windows Users Command**
```
docker run --rm -v /$(pwd):/home/rstudio/telco_churn_predictor adammorphy/telco_churn_docker:latest make -C /home/rstudio/telco_churn_predictor clean
```

**Windows Users Command**
```
docker run --rm -v /$(pwd)://home//rstudio//telco_churn_predictor adammorphy/telco_churn_docker:latest make -C //home//rstudio//telco_churn_predictor clean
```



**2. Without Using Docker**

To replicate the analysis, clone this GitHub repository, install the dependencies listed below (or install the given environment yaml file below), and run the following commands at the command line/terminal from the root directory of this project:
```
make all
```

To reset the repo to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:
```
make clean
```

Below image can demonstrate the dependency diagram for making the report by Makefile

<img src="Makefile.png" alt="drawing" style="height:100px;"/>


## Environment

The project environment can be found
[here](https://github.com/UBC-MDS/Telco_Customer_Churn_Prediction_Group12/blob/main/env_telco_churn.yaml)

The environment can be created via
`conda env create --file env_telco_churn.yaml`

Activate the environment via
`conda activate telco`

Deactivate the environment via
`conda deactivate`

**Windows Users**: You may have to run this command to make the `altair` and `altair_server`
run as expected
`npm install -g vega vega-cli vega-lite canvas`

## Dependancies

* Python 3.9.7 and Python packages:
    + docopt==0.6.2
    + pandas==0.24.2
    + numpy==1.21.4
    + requests==2.22.0
    + scikit-learn>=1.0
    + matplotlib>=3.2.2
    + altair==4.1.0
    + altair_saver
    + seaborn==0.8.1
    + jsonschema=3.2.0
    + lxml
- R version 4.1.1 and R packages:
    + knitr==1.26
    + tidyverse==1.2.1

## License
The Telco Customer Churn Predictor materials here are licensed under the MIT License. If you use or re-mix this project please provide attribution and a link to this GitHub repository.

## References

de Jonge, Edwin. 2020. Docopt: Command-Line Interface Specification Language. https://CRAN.R-project.org/package=docopt.

Harris, Charles R., K. Jarrod Millman, Stéfan J van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, et al. 2020. “Array Programming with NumPy.” Nature 585: 357–62. https://doi.org/10.1038/s41586-020-2649-2.

Hunter, John D. 2007. “Matplotlib: A 2d Graphics Environment.” Computing in Science & Engineering 9 (3): 90–95.

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12: 2825–30.

Pustokhina, Irina V, Denis A Pustokhin, Phong Thanh Nguyen, Mohamed Elhoseny, and K Shankar. 2021. “Multi-Objective Rain Optimization Algorithm with WELM Model for Customer Churn Prediction in Telecommunication Sector.” Complex & Intelligent Systems, 1–13.

Team, The pandas development. 2020. Pandas-Dev/Pandas: Pandas (version latest). Zenodo. https://doi.org/10.5281/zenodo.3509134.

Team, OmniSci. 2020. “Strategies for Reducing Churn Rate in the Telecom Industry.” RSS. https://www.omnisci.com/blog/strategies-for-reducing-churn-rate-in-the-telecom-industry.

Van Rossum, Guido, and Fred L. Drake. 2009. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.

VanderPlas, Jacob, Brian Granger, Jeffrey Heer, Dominik Moritz, Kanit Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben Welsh, and Scott Sievert. 2018. “Altair: Interactive Statistical Visualizations for Python.” Journal of Open Source Software 3 (32): 1057.

Wickham, Hadley. 2017. Tidyverse: Easily Install and Load the ’Tidyverse’. https://CRAN.R-project.org/package=tidyverse.

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research in R.” In Implementing Reproducible Computational Research, edited by Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman; Hall/CRC. http://www.crcpress.com/product/isbn/9781466561595.
