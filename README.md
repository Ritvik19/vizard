# vizard
Intuitive, Easy and Quick Vizualizations for Data Science Projects

## Installation

    pip install vizard

or

    pip install git+https://github.com/Ritvik19/vizard.git

## Documentation

### Instantiate Vizard Object
The Vizard object holds the `DataFrame` along with its configurations including the `PROBLEM_TYPE`, `DEPENDENT_VARIABLE`, `CATEGORICAL_INDEPENDENT_VARIABLES`, `CONTINUOUS_INDEPENDENT_VARIABLES`, and `TEXT_VARIABLES`

    import vizard

    class config:
        PROBLEM_TYPE = 'regression' or 'classification'
        DEPENDENT_VARIABLE = 'target_variable'
        CATEGORICAL_INDEPENDENT_VARIABLES = [categorical_features]
        CONTINUOUS_INDEPENDENT_VARIABLES = [continuous features]
        TEXT_VARIABLES = [text features]

    viz = vizard.Vizard(df, config)

### Exploratory Data Analysis
After Instatiating the `Vizard` object, you can try different plots for EDA
* Check Missing Values:
    
      viz.check_missing()

* Count of Missing Values:
    
      viz.count_missing()

* Count of Unique Values:
    
      viz.count_unique()

* Count of Missing Values by Group:
    
      viz.count_missing_by_group(class_variable)

* Count of Unique Values by Group:
    
      viz.count_unique_by_group(class_variable)

### Target Column Analysis
Based on the type of problem, perform a univariate analysis of target column
    
    viz.dependent_variable()

### Segmented Univariate Analysis
Based on the type of problem, preform segmented univariate analysis of all feature columns with respect to the target column

* Categorical Variables
  
        viz.categorical_variables()

* Continuous Variables
  
        viz.continuous_variables()

* Text Variables
  
        viz.wordcloud()

        viz.wordcloud_freq()

### Bivariate/Multivariate Analysis
Based on the type of problem, perform bivariate and multivariate analysis on all the feature columns

* Pairwise Scatter
  
        viz.pairwise_scatter()

* Pairwise Violin
  
        viz.pairwise_violin()

* Pairwise Cross Tabs
  
        viz.pairwise_crosstabs()


### Correlation Analysis
Based on the type of variables, perform correaltion analysis on all the feature columns

* Correlation Plot
  
        viz.corr_plot()

* Pair Plot
  
        viz.pair_plot()

* Chi Square Plot
  
        viz.chi_sq_plot()

## Usage

1. [Classification Case](https://www.kaggle.com/ritvik1909/vizard-usage?scriptVersionId=54381087)
2. [Regression Case](https://www.kaggle.com/ritvik1909/vizard-usage?scriptVersionId=54381676)
3. [Text Classification Case](https://www.kaggle.com/ritvik1909/vizard-usage?scriptVersionId=54875336)