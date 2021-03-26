import sys, os

sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import vizard


def test_clf():
    df = pd.read_csv(os.path.join("tests", "titanic-train.csv"))

    class config:
        PROBLEM_TYPE = "classification"
        DEPENDENT_VARIABLE = "Survived"
        CATEGORICAL_INDEPENDENT_VARIABLES = [
            "Pclass",
            "Sex",
            "SibSp",
            "Parch",
            "Embarked",
        ]
        CONTINUOUS_INDEPENDENT_VARIABLES = ["Age", "Fare", "PassengerId"]
        TEXT_VARIABLES = ["Name", "Cabin", "Ticket"]

    viz = vizard.Vizard(df, config)
    viz.check_missing();
    viz.count_missing();
    viz.count_missing_by_group("Survived");
    viz.count_unique();
    viz.count_unique_by_group("Survived");
    viz.dependent_variable();
    viz.categorical_variables();
    viz.continuous_variables();
    viz.wordcloud_by_group("Survived");
    viz.wordcloud_freq("Survived");
    viz.pairwise_scatter();
    viz.pairwise_violin();
    viz.pairwise_crosstabs();
    viz.trivariate_bubbleplot("Age", "Fare", "PassengerId");
    viz.trivariate_scatterplot("Age", "Fare", "Survived");
    viz.trivariate_violinplot("Survived", "Age", "Sex");
    viz.corr_plot();
    viz.pair_plot();
    viz.chi_sq_plot();
    viz.point_biserial_plot();


def test_reg():
    df = pd.read_csv(os.path.join("tests", "insurance.csv"))

    class config:
        PROBLEM_TYPE = "regression"
        DEPENDENT_VARIABLE = "charges"
        CATEGORICAL_INDEPENDENT_VARIABLES = ["sex", "smoker", "region"]
        CONTINUOUS_INDEPENDENT_VARIABLES = ["age", "bmi", "children"]

    viz = vizard.Vizard(df, config)
    viz.check_missing();
    viz.count_missing();
    viz.count_unique();
    viz.dependent_variable();
    viz.categorical_variables();
    viz.continuous_variables();
    viz.pairwise_scatter();
    viz.pairwise_violin();
    viz.pairwise_crosstabs();
    viz.trivariate_bubbleplot("age", "bmi", "children");
    viz.trivariate_scatterplot("age", "bmi", "sex");
    viz.trivariate_violinplot("smoker", "age", "sex");
    viz.corr_plot();
    viz.pair_plot();
    viz.chi_sq_plot();
    viz.point_biserial_plot();