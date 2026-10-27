# author: Anupriya Srivastava
# date: 2021-11-24

'''
Performs exploratory data analysis on the Telco Churn data (from hhttps://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). Saves output figures in .png files.

Usage: src/eda_script.py --input=<input> --out_dir=<out_dir>

Options:
--input=<input>       Path (including filename) to cleaned data (csv file)
--out_dir=<out_dir>   Path to directory where the figures should be saved
'''

# Import libraries
import pandas as pd
import altair as alt
from altair_saver import save
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi
from docopt import docopt
import os

alt.renderers.enable('mimetype')

opt = docopt(__doc__)


def main(input, out_dir):

       # Ensure output directory exists
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except:
            print("Wasn't able to create output directory. Check permissions.")

    # Read data and convert class to pandas df
    train_df = pd.read_csv(input)

    # Data Wrangling
    train_df['Churn'] = train_df['Churn'].replace(True, "True")
    train_df['Churn'] = train_df['Churn'].replace(False, "False")

    train_df = train_df.rename(columns={'tenure': 'Tenure',
                                        'SeniorCitizen': 'Senior Citizen',
                                        'MonthlyCharges': 'Monthly Charges',
                                        'TotalCharges': 'Total Charges',
                                        'InternetService': 'Internet Service',
                                        'MultipleLines': 'Multiple Lines',
                                        'OnlineSecurity': 'Online Security',
                                        'OnlineBackup': 'Online Backup',
                                        'DeviceProtection': 'Device Protection',
                                        'StreamingMovies': 'Streaming Movies',
                                        'StreamingTV': 'Streaming TV',
                                        'PhoneService': 'Phone Service',
                                        'TechSupport': 'Tech Support',
                                        'PaperlessBilling': 'Paperless Billing',
                                        'PaymentMethod': 'Payment Method'})

    # Analysing class imbalance for target variable
    target_class_imbalance = alt.Chart(train_df, title='Class imbalance').mark_bar(opacity=0.6).encode(
                                    alt.X('Churn:N'),
                                    alt.Y('count()', title='Count'),
                                    color='Churn',
                                    ).properties(
                                        width=200, height=200
                                    ).configure_axis(
                                        labelFontSize=14,
                                        titleFontSize=16
                                    ).configure_legend(
                                        titleFontSize=14
                                    ).configure_title(
                                        fontSize=18
                                    )

    # Numerical features EDA
    numeric_feat_dist = alt.Chart(train_df).mark_bar(opacity=0.6).encode(
                             alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=40)),
                             alt.Y('count()', title='Count', stack=False),
                             color='Churn'
                             ).properties(
                                width=200, height=200
                            ).repeat(
                                ['Tenure', 'Monthly Charges', 'Total Charges']
                            )

    cor = train_df.corr()
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    numeric_feat_corr = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds);

    # Categorical features EDA
    train_cat = train_df.select_dtypes(include='object').copy()
    catfeatures_stats = pd.DataFrame(columns=['Feature', 'Unique values', 'Categories', 'Missing values'])
    tmp = pd.DataFrame()
    for c in train_cat.columns:
        tmp['Feature'] = [c]
        tmp['Unique values'] = [train_cat[c].unique()]
        tmp['Categories'] = int(train_cat[c].nunique())
        tmp['Missing values'] = train_cat[c].isnull().sum()
        catfeatures_stats = catfeatures_stats.append(tmp)

    cat_feat_churn_dist = alt.Chart(train_df).mark_bar().encode(
                            alt.X('count()', title='Count'),
                            alt.Y(alt.repeat(), type='nominal'),
                            alt.Color('Churn')
                    ).properties(
                            height=100, width=300
                    ).repeat(
                            ['Senior Citizen', 'Partner', 'Dependents'],
                            columns=1
                    )

    cat_feat_2dhist = alt.Chart(train_df).mark_square().encode(
                            alt.X(alt.repeat(), type='nominal'),
                            y='Churn',
                            color='count()',
                            size='count()'
                    ).properties(
                            width=150, height=80
                    ).repeat(
                            ['Contract', 'Internet Service', 'Multiple Lines', 'Online Security',
                             'Online Backup', 'Device Protection', 'Streaming Movies', 'Streaming TV',
                             'Phone Service', 'Tech Support', 'Paperless Billing', 'Payment Method'],
                            columns=3
                    )

    # Test that Figure object have been created

    test_figs(target_class_imbalance)
    test_figs(numeric_feat_corr)
    test_figs(cat_feat_churn_dist)
    test_figs(cat_feat_2dhist)
    test_figs(numeric_feat_dist)


    # Saving all outputs
    dfi.export(catfeatures_stats, f"{out_dir}table_1_cat_unique_values.png", table_conversion='latex')
    target_class_imbalance.save(f"{out_dir}figure_1_class_imbalance.png", scale_factor=3)
    numeric_feat_dist.save(f"{out_dir}figure_2_numeric_feat_dist.png", scale_factor=3)
    numeric_feat_corr.figure.savefig(f"{out_dir}figure_3_numeric_feat_corr.png", scale_factor=3)
    cat_feat_churn_dist.save(f"{out_dir}figure_4_cat_feat_churn_dist.png", scale_factor=3)
    cat_feat_2dhist.save(f"{out_dir}figure_5_cat_feat_2dhist.png", scale_factor=3)

    print("EDA reports successfully stored in: ", (out_dir))


def test_figs(fig):
    
    assert fig != None, "Output figure is empty"


if __name__ == "__main__":
    # Call main method, and have the user input file, out dir
    main(opt["--input"], opt["--out_dir"])