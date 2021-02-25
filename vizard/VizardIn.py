import numpy as np
import pandas as pd

from math import ceil
from itertools import combinations, product
from collections import Counter
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS


class VizardIn:
    """VizardIn object holds the `DataFrame` along with its configurations including the
    `PROBLEM_TYPE`, `DEPENDENT_VARIABLE`, `CATEGORICAL_INDEPENDENT_VARIABLES`,
    `CONTINUOUS_INDEPENDENT_VARIABLES`, and `TEXT_VARIABLES`"""

    def __init__(self, data, config=None):
        self._data = data
        self._config = config

    @property
    def data(self):
        return self._data

    @property
    def config(self):
        return self._config

    @data.setter
    def data(self, data):
        self._data = data

    @config.setter
    def config(self, config):
        self._config = config

    def categorical(self, x):
        fig = px.bar(
            pd.DataFrame(self.data[x].value_counts()), title=f"{x} Distribution"
        )
        fig.show()

    def continuous(self, x):
        fig = px.histogram(self.data, x=x, title=f"{x} Distribution")
        fig.show()

    def categorical_vs_continuous(self, col, target):
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[[{}, {"colspan": 2}, None], [{"colspan": 3}, None, None]],
        )
        fig.add_trace(
            go.Bar(
                x=list(self.data[col].value_counts().index),
                y=list(self.data[col].value_counts()),
                name=col,
            ),
            row=1,
            col=1,
        )
        for val in self.data[col].unique():
            fig.add_trace(
                go.Histogram(
                    x=self.data[self.data[col] == val][target], name=f"{col}: {val}"
                ),
                row=1,
                col=2,
            )
        for val in self.data[col].unique():
            fig.add_trace(
                go.Box(
                    y=self.data[self.data[col] == val][target], name=f"{col}: {val}"
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text=target, row=1, col=2)
        fig.update_yaxes(title_text=target, row=2, col=1)
        fig.update_layout(title=f"{col}")
        fig.show()

    def categorical_vs_categorical(self, col, target):
        fig = make_subplots(
            rows=1,
            cols=2,
        )
        fig.add_trace(
            go.Bar(
                x=list(self.data[col].value_counts().index),
                y=list(self.data[col].value_counts()),
                name=col,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=pd.crosstab(index=self.data[target], columns=self.data[col]),
                name=f"{col}",
                colorscale="blues",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text=col, row=1, col=2, showticklabels=False)
        fig.update_yaxes(title_text=target, row=1, col=2, showticklabels=False)
        fig.update_layout(title=f"{col}")
        fig.show()

    def continuous_vs_continuous(self, col, target):
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [{"colspan": 3}, None, None],
                [{"rowspan": 2}, {"colspan": 2, "rowspan": 2}, None],
                [None, None, None],
            ],
        )
        fig.add_trace(go.Histogram(x=self.data[col], name=f"{col}"), row=1, col=1)
        fig.add_trace(go.Box(y=self.data[col], name=f"{col}"), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=self.data[col], y=self.data[target], mode="markers", name=f"{col}"
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text=col, row=1, col=1)
        fig.update_xaxes(title_text=col, row=2, col=2)
        fig.update_yaxes(title_text=target, row=2, col=2)
        fig.show()

    def continuous_vs_categorical(self, col, target):
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [{"rowspan": 2}, {"colspan": 2}, None],
                [None, {"colspan": 2}, None],
                [{"colspan": 3}, None, None],
            ],
        )
        fig.add_trace(
            go.Box(
                y=self.data[col],
                name=col,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(go.Histogram(x=self.data[col], name=col), row=1, col=2)
        for val in self.data[target].unique():
            fig.add_trace(
                go.Histogram(
                    x=self.data[self.data[target] == val][col], name=f"{target}: {val}"
                ),
                row=2,
                col=2,
            )
        for val in self.data[target].unique():
            fig.add_trace(
                go.Box(
                    y=self.data[self.data[target] == val][col], name=f"{target}: {val}"
                ),
                row=3,
                col=1,
            )
        fig.update_layout(title=f"{col}")
        fig.show()

    def check_missing(self):
        """Plot a heatmap to visualize missing values in the DataFrame"""
        fig = px.imshow(
            self.data.isnull(), color_continuous_scale="viridis", title="Missing Values"
        )
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update(layout_coloraxis_showscale=False)
        fig.show()

    def count_missing(self):
        """Plot to visualize the count of missing values in each column in the DataFrame"""
        fig = px.bar(
            pd.DataFrame(self.data.isnull().sum(), columns=["Missing Count"]),
            title="Count of Missing Values",
        )
        fig.show()

    def count_missing_by_group(self, group_by=None):
        """Plot to visualize the count of missing values in each column in the DataFrame,
        grouped by a categorical variable
        :param group_by: a column in the DataFrame"""
        fig = px.bar(
            self.data.drop(group_by, 1)
            .isna()
            .groupby(self.data[group_by], sort=False)
            .sum()
            .T,
            title=f"Count of Missing Values grouped by {group_by}",
        )
        fig.show()

    def count_unique(self):
        """Plot to visualize the count of unique values in each column in the DataFrame"""
        fig = px.bar(self.data.nunique(), title="Count of Unique Values")
        fig.show()

    def count_unique_by_group(self, group_by=None):
        """Plot to visualize the count of unique values in each column in the DataFrame,
        grouped by a categorical variable
        :param group_by: a column in the DataFrame"""
        fig = px.bar(
            self.data.groupby(group_by).nunique().T,
            title=f"Count of Unique Values Grouped By {group_by}",
        )
        fig.show()

    def dependent_variable(self):
        """Based on the type of problem, plot a univariate visualization of target column"""
        if self.config.PROBLEM_TYPE == "regression":
            self.continuous(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            self.categorical(self.config.DEPENDENT_VARIABLE)
        else:
            print("Invalid Problem Type")

    def pairwise_scatter(self):
        """Plot pairwise scatter plots to visualize the continuous variables in the dataset"""
        comb = list(combinations(self.config.CONTINUOUS_INDEPENDENT_VARIABLES, 2))
        for i, (x, y) in enumerate(comb):
            if self.config.PROBLEM_TYPE == "classification":
                px.scatter(
                    self.data,
                    x=x,
                    y=y,
                    color=self.config.DEPENDENT_VARIABLE,
                    title=f"{y} vs {x}",
                ).show()
            elif self.config.PROBLEM_TYPE == "regression":
                px.scatter(
                    self.data,
                    x=x,
                    y=y,
                    title=f"{y} vs {x}",
                ).show()

    def pairwise_violin(self):
        """Plot pairwise violin plots to visualize the continuous variables grouped by the
        categorical variables in the dataset"""
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)

        comb = list(product(cat_cols, self.config.CONTINUOUS_INDEPENDENT_VARIABLES))

        for i, (x, y) in enumerate(comb):
            px.violin(self.data, y=y, x=x, box=True, title=f"{y} vs {x}").show()

    def pairwise_crosstabs(self):
        """Plot pairwise crosstabs plots to visualize the categorical variables in the dataset"""
        cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cols.append(self.config.DEPENDENT_VARIABLE)

        comb = list(combinations(cols, 2))

        for i, (x, y) in enumerate(comb):
            px.imshow(
                pd.crosstab(index=self.data[y], columns=self.data[x]),
                color_continuous_scale="blues",
                title=f"{y} vs {x}",
            ).show()

    def pair_plot(self):
        """Plot a pairplot to vizualize the complete DataFrame"""
        if self.config.PROBLEM_TYPE == "classification":
            px.scatter_matrix(
                self.data,
                dimensions=self.config.CONTINUOUS_INDEPENDENT_VARIABLES,
                color=self.config.DEPENDENT_VARIABLE,
                title="Pair Plot",
            ).show()
        elif self.config.PROBLEM_TYPE == "regression":
            px.scatter_matrix(
                self.data,
                dimensions=self.config.CONTINUOUS_INDEPENDENT_VARIABLES,
                title="Pair Plot",
            ).show()

    def categorical_variables(self):
        """Create bivariate visualizations for the categorical variables with the target variable"""
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_continuous(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_categorical(col, self.config.DEPENDENT_VARIABLE)
        else:
            pass

    def continuous_variables(self):
        """Create bivariate visualizations for the continuous variables with the target variable"""
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_continuous(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_categorical(col, self.config.DEPENDENT_VARIABLE)
        else:
            pass

    def corr_plot(self):
        """Plot a heatmap to vizualize the correlation between the various continuous columns in the DataFrame"""
        corr = self.data.corr()
        fig = px.imshow(
            corr, color_continuous_scale="blues", title="Pearson Correlation"
        )
        fig.show()

    def chi_sq_plot(self, figsize=(10, 10)):
        """Plot a heatmap to vizualize the chi 2 value between the various categorical columns in the DataFrame"""

        cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cols.append(self.config.DEPENDENT_VARIABLE)

        n = len(cols)
        chi_sq_table = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                stat, p, dof, expected = chi2_contingency(
                    pd.crosstab(index=self.data[cols[i]], columns=self.data[cols[j]])
                )
                chi_sq_table[i][j] = p

        chi_sq_table = pd.DataFrame(chi_sq_table, columns=cols, index=cols)
        fig = px.imshow(
            chi_sq_table, color_continuous_scale="blues", title="Chi Square Values"
        )

        fig.show()

    def wordcloud(self):
        """Plot word clouds for the text variables and grouped word clouds if the problem is a classification one"""
        n_rows = ceil(len(self.config.TEXT_VARIABLES) / 2)
        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, col in enumerate(self.config.TEXT_VARIABLES):
            self._wc(
                " ".join(self.data[col].fillna("").values),
                col,
                fig.add_subplot(n_rows, 2, i + 1),
            )
        plt.show()
        if self.config.PROBLEM_TYPE == "classification":
            n_rows = ceil(
                (
                    len(self.config.TEXT_VARIABLES)
                    * self.data[self.config.DEPENDENT_VARIABLE].nunique()
                )
                / 2
            )
            fig = plt.figure(figsize=(20, 6 * n_rows))
            i = 0
            for col in self.config.TEXT_VARIABLES:
                for label in sorted(self.data[self.config.DEPENDENT_VARIABLE].unique()):
                    i += 1
                    self._wc(
                        " ".join(
                            self.data[
                                self.data[self.config.DEPENDENT_VARIABLE] == label
                            ][col]
                            .fillna("")
                            .values
                        ),
                        f"{col} ({label})",
                        fig.add_subplot(n_rows, 2, i),
                    )
            plt.show()

    def wordcloud_freq(self):
        """Plot word clouds for the text variables and grouped word clouds for the words unique to each group
        if the problem is a classification one"""
        n_rows = ceil(len(self.config.TEXT_VARIABLES) / 2)
        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, col in enumerate(self.config.TEXT_VARIABLES):
            self._wc_freq(
                Counter((" ".join(self.data[col].fillna("").values)).split()),
                col,
                fig.add_subplot(n_rows, 2, i + 1),
            )
        plt.show()
        if self.config.PROBLEM_TYPE == "classification":
            n_rows = ceil(
                (
                    len(self.config.TEXT_VARIABLES)
                    * self.data[self.config.DEPENDENT_VARIABLE].nunique()
                )
                / 2
            )
            fig = plt.figure(figsize=(20, 6 * n_rows))
            i = 0
            for col in self.config.TEXT_VARIABLES:
                for label in sorted(self.data[self.config.DEPENDENT_VARIABLE].unique()):
                    i += 1
                    self._wc_freq(
                        (
                            Counter(
                                " ".join(
                                    self.data[
                                        self.data[self.config.DEPENDENT_VARIABLE]
                                        == label
                                    ][col]
                                    .fillna("")
                                    .values
                                ).split()
                            )
                            - Counter(
                                " ".join(
                                    self.data[
                                        self.data[self.config.DEPENDENT_VARIABLE]
                                        != label
                                    ][col]
                                    .fillna("")
                                    .values
                                ).split()
                            )
                        ),
                        f"{col} ({label})",
                        fig.add_subplot(n_rows, 2, i),
                    )
            plt.show()

    def _wc(self, text, label, ax):
        wordcloud = WordCloud(height=600, width=1000, stopwords=STOPWORDS).generate(
            text
        )
        ax.imshow(wordcloud)
        ax.set_title(f"Word Cloud - {label}", fontsize=18)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return ax

    def _wc_freq(self, text_freq, label, ax):
        wordcloud = WordCloud(
            height=600, width=1000, stopwords=STOPWORDS
        ).generate_from_frequencies(text_freq)
        ax.imshow(wordcloud)
        ax.set_title(f"Word Cloud - {label}", fontsize=18)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return ax