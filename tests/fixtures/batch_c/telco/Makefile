# Group 12: Customer Churn Predictor
# contributors: Zihan Zhou, Anupriya Srivastava, Adam Morphy, Jordan Casoli
# date: 2021-10-02

all: docs/Telco_Customer_Churn_Prediction_Report.html

# download data
data/raw/IBM-Telco-Customer-Churn.csv:
	python src/data_download.py --file_path=https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv --out_type=csv  --out_file=data/raw/IBM-Telco-Customer-Churn.csv

# clean & split data
data/processed/train_df.csv data/processed/test_df.csv: data/raw/IBM-Telco-Customer-Churn.csv
	python src/pre_process_script.py --input=data/raw/IBM-Telco-Customer-Churn.csv --out_dir=data/processed/

# exploratory data analysis - visualize predictor distributions across classes
results/figure_1_class_imbalance.png results/figure_2_numeric_feat_dist.png results/figure_3_numeric_feat_corr.png results/figure_4_cat_feat_churn_dist.png results/figure_5_cat_feat_2dhist.png results/table_1_cat_unique_values.png: data/processed/train_df.csv
	python src/eda_script.py --input=data/processed/train_df.csv --out_dir=results/

# train & test model
results/classification_report.csv results/confusion_matrix.png results/feature_importance.csv: data/processed/train_df.csv data/processed/test_df.csv
	python src/analysis.py --train_path=data/processed/train_df.csv --test_path=data/processed/test_df.csv --out_dir=results/

# render report
docs/Telco_Customer_Churn_Prediction_Report.html docs/Telco_Customer_Churn_Prediction_Report.md: docs/references_Telco_Customer_Churn_Prediction.bib docs/Telco_Customer_Churn_Prediction_Report.Rmd results/figure_1_class_imbalance.png results/figure_2_numeric_feat_dist.png results/figure_3_numeric_feat_corr.png results/figure_4_cat_feat_churn_dist.png results/figure_5_cat_feat_2dhist.png results/table_1_cat_unique_values.png results/classification_report.csv results/confusion_matrix.png results/feature_importance.csv
	Rscript -e "rmarkdown::render('docs/Telco_Customer_Churn_Prediction_Report.Rmd')"

clean: 
	rm -rf data
	rm -rf results
	rm -rf docs/Telco_Customer_Churn_Prediction_Report.html
	rm -rf docs/Telco_Customer_Churn_Prediction_Report.md
