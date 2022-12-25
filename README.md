# vizard

Intuitive, Interactive, Easy and Quick Visualizations for Data Science Projects

[![Downloads](https://pepy.tech/badge/vizard)](https://pepy.tech/project/vizard)
[![Downloads](https://pepy.tech/badge/vizard/month)](https://pepy.tech/project/vizard)
[![Downloads](https://pepy.tech/badge/vizard/week)](https://pepy.tech/project/vizard)

## Installation

    pip install vizard

or

    pip install git+https://github.com/Ritvik19/vizard.git

## Documentation

### Instantiate Vizard Object

The Vizard or VizardIn object holds the `DataFrame` along with its configurations including the `PROBLEM_TYPE`, `DEPENDENT_VARIABLE`, `CATEGORICAL_INDEPENDENT_VARIABLES`, `CONTINUOUS_INDEPENDENT_VARIABLES`, and `TEXT_VARIABLES`

    import vizard

    class config:
        PROBLEM_TYPE = 'regression' or 'classification' or 'unsupervised'
        DEPENDENT_VARIABLE = 'target_variable'
        CATEGORICAL_INDEPENDENT_VARIABLES = [categorical_features]
        CONTINUOUS_INDEPENDENT_VARIABLES = [continuous features]
        TEXT_VARIABLES = [text features]

    viz = vizard.Vizard(df, config)
    # for interactive plots use:
    viz = vizard.VizardIn(df, config)

### Exploratory Data Analysis

After Instatiating the `Vizard` object, you can try different plots for EDA

- Check Missing Values:

      viz.check_missing()

- Count of Missing Values:

      viz.count_missing()

- Count of Unique Values:

      viz.count_unique()

- Count of Missing Values by Group:

      viz.count_missing_by_group(class_variable)

- Count of Unique Values by Group:
  viz.count_unique_by_group(class_variable)

### Target Column Analysis

Based on the type of problem, perform a univariate analysis of target column

    viz.dependent_variable()

### Segmented Univariate Analysis

Based on the type of problem, preform segmented univariate analysis of all feature columns with respect to the target column

- Categorical Variables

        viz.categorical_variables()

- Continuous Variables

        viz.continuous_variables()

- Text Variables

        viz.wordcloud()

        viz.wordcloud_by_group()

        viz.wordcloud_freq()

### Bivariate Analysis

Based on the type of variables, perform bivariate analysis on all the feature columns

- Pairwise Scatter

        viz.pairwise_scatter()

- Pairwise Violin

        viz.pairwise_violin()

- Pairwise Cross Tabs

        viz.pairwise_crosstabs()

### Trivariate Analysis

Based on the type of variables, perform trivariate analysis on any of the feature columns

- Trivariate Bubble (Continuous vs Continuous vs Continuous)

        viz.trivariate_bubble(x, y, s)

- Trivariate Scatter (Continuous vs Continuous vs Categorical)

        viz.trivariate_scatter(x, y, c)

- Trivariate Violin (Categorical vs Continuous vs Categorical)

        viz.trivariate_violin(x, y, c)

### Correlation Analysis

Based on the type of variables, perform correaltion analysis on all the feature columns

- Correlation Plot

        viz.corr_plot()

- Pair Plot

        viz.pair_plot()

- Chi Square Plot

        viz.chi_sq_plot()

## Save the plots to PDF using Viz2PDF

You can also save the plots to a pdf file in order to generate an EDA report

The `Viz2PDF` object takes in all your `Vizard` plots and creates a pdf report out of them

```
viz = vizard.Vizard(df, config)
viz2pdf = vizard.Viz2PDF('viz_report.pdf')

plots = [
    viz.check_missing(),
    viz.count_missing(),
    viz.count_unique(),
    viz.dependent_variable(),
    viz.categorical_variables(),
    viz.continuous_variables(),
    viz.pairwise_scatter(),
    viz.pairwise_violin(),
    viz.pairwise_crosstabs(),
]
viz2pdf(plots)
```

## Usage

1. [Classification Case](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Classification%20Case.ipynb)
2. [Regression Case](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Regression%20Case.ipynb)
3. [Text Classification Case](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Text%20Classification%20Case.ipynb)
4. [Unsupervised Case](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Unsupervised%20Case.ipynb)
5. [Classification Case (Interactive)](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Classification%20Interactive%20Case.ipynb)
6. [Regression Case (Interactive)](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Regression%20Interactive%20Case.ipynb)
7. [Unsupervised Case (Interactive)](https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Unsupervised%20Interactive%20Case.ipynb)
