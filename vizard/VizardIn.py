import numpy as np
import pandas as pd

from math import ceil
from itertools import combinations, product
from collections import Counter
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder

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
        return fig

    def continuous(self, x):
        fig = px.histogram(self.data, x=x, title=f"{x} Distribution")
        return fig

    def categorical_vs_continuous(self, col, target, return_fig=True):
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
        return fig if return_fig else fig.show() if fig is not None else None

    def categorical_vs_categorical(self, col, target, return_fig=True):
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
        return fig if return_fig else fig.show() if fig is not None else None

    def continuous_vs_continuous(self, col, target, return_fig=True):
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
        return fig if return_fig else fig.show() if fig is not None else None

    def continuous_vs_categorical(self, col, target, return_fig=True):
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
        return fig if return_fig else fig.show() if fig is not None else None

    def check_missing(self, return_fig=True):
        """Plot a heatmap to visualize missing values in the DataFrame"""
        fig = px.imshow(
            self.data.isnull(), color_continuous_scale="viridis", title="Missing Values"
        )
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update(layout_coloraxis_showscale=False)
        return fig if return_fig else fig.show() if fig is not None else None

    def count_missing(self, return_fig=True):
        """Plot to visualize the count of missing values in each column in the DataFrame"""
        fig = px.bar(
            pd.DataFrame(self.data.isnull().sum(), columns=["Missing Count"]),
            title="Count of Missing Values",
        )
        return fig if return_fig else fig.show() if fig is not None else None

    def count_missing_by_group(self, group_by=None, return_fig=True):
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
        return fig if return_fig else fig.show() if fig is not None else None

    def count_unique(self, return_fig=True):
        """Plot to visualize the count of unique values in each column in the DataFrame"""
        fig = px.bar(self.data.nunique(), title="Count of Unique Values")
        return fig if return_fig else fig.show() if fig is not None else None

    def count_unique_by_group(self, group_by=None, return_fig=True):
        """Plot to visualize the count of unique values in each column in the DataFrame,
        grouped by a categorical variable
        :param group_by: a column in the DataFrame"""
        fig = px.bar(
            self.data.groupby(group_by).nunique().T,
            title=f"Count of Unique Values Grouped By {group_by}",
        )
        return fig if return_fig else fig.show() if fig is not None else None

    def dependent_variable(self, return_fig=True):
        """Based on the type of problem, plot a univariate visualization of target column"""
        if self.config.PROBLEM_TYPE == "regression":
            fig = self.continuous(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            fig = self.categorical(self.config.DEPENDENT_VARIABLE)
        else:
            print("Invalid Problem Type")
            fig = None
        return fig if return_fig else fig.show() if fig is not None else None

    def pairwise_scatter(self):
        """Plot pairwise scatter plots to visualize the continuous variables in the dataset"""
        comb = list(combinations(self.config.CONTINUOUS_INDEPENDENT_VARIABLES, 2))
        for i, (x, y) in enumerate(comb):
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

    def pair_plot(self, return_fig=True):
        """Plot a pairplot to vizualize the complete DataFrame"""
        if self.config.PROBLEM_TYPE == "classification":
            fig = px.scatter_matrix(
                self.data,
                dimensions=self.config.CONTINUOUS_INDEPENDENT_VARIABLES,
                color=self.config.DEPENDENT_VARIABLE,
                title="Pair Plot",
            )
        elif self.config.PROBLEM_TYPE == "regression":
            fig = px.scatter_matrix(
                self.data,
                dimensions=self.config.CONTINUOUS_INDEPENDENT_VARIABLES,
                title="Pair Plot",
            )
        return fig if return_fig else fig.show() if fig is not None else None

    def categorical_variables(self):
        """Create bivariate visualizations for the categorical variables with the target variable"""
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_continuous(
                    col, self.config.DEPENDENT_VARIABLE
                ).show()
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_categorical(
                    col, self.config.DEPENDENT_VARIABLE
                ).show()
        else:
            pass

    def continuous_variables(self):
        """Create bivariate visualizations for the continuous variables with the target variable"""
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_continuous(
                    col, self.config.DEPENDENT_VARIABLE
                ).show()
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_categorical(
                    col, self.config.DEPENDENT_VARIABLE
                ).show()
        else:
            pass

    def corr_plot(self, method="pearson", return_fig=True):
        """Plot a heatmap to vizualize the correlation between the various continuous columns in the DataFrame
        :param method: the method of correlation {'pearson', 'kendall', 'spearman'}
        """
        corr = self.data.corr(method=method)
        fig = px.imshow(
            corr, color_continuous_scale="blues", title="Pearson Correlation"
        )
        return fig if return_fig else fig.show() if fig is not None else None

    def point_biserial_plot(self, return_fig=True):
        """Plot a heatmap to visualize point biserial correaltion between continuous and categorical variables"""
        num_cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            num_cols.append(self.config.DEPENDENT_VARIABLE)
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)

        pb_table = np.zeros((len(cat_cols), len(num_cols)))
        for i in range(len(cat_cols)):
            for j in range(len(num_cols)):
                df_ = self.data.dropna(subset=[cat_cols[i], num_cols[j]])
                pb_table[i][j] = pointbiserialr(
                    LabelEncoder().fit_transform(df_[cat_cols[i]]), df_[num_cols[j]]
                )[0]

        fig = px.imshow(
            pd.DataFrame(pb_table, index=cat_cols, columns=num_cols),
            color_continuous_scale="rdbu",
            title="Point Biserial Plot",
        )
        return fig if return_fig else fig.show() if fig is not None else None

    def chi_sq_plot(self, statistic="Chi Sq", figsize=(10, 10), return_fig=True):
        """Plot a heatmap to vizualize the chi 2 statistic between the various categorical columns in the DataFrame
        :param statistic: the statistic to plot {"Chi Sq", "Phi", "Cramer's V", "Tschuprow's T", "Contingency Coefficient"}
        """

        cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cols.append(self.config.DEPENDENT_VARIABLE)

        n = len(cols)
        chi_table = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                crss_tab = pd.crosstab(
                    index=self.data[cols[i]], columns=self.data[cols[j]]
                )
                chi2 = chi2_contingency(crss_tab)[0]
                crss_sum = np.sum(np.sum(crss_tab))
                if statistic == "Chi Sq":
                    chi_table[i][j] = chi2
                elif statistic == "Phi":
                    chi_table[i][j] = np.sqrt(chi2 / crss_sum)
                elif statistic == "Cramer's V":
                    chi_table[i][j] = np.sqrt(
                        chi2 / (crss_sum * (min(crss_tab.shape) - 1))
                    )
                elif statistic == "Tschuprow's T":
                    chi_table[i][j] = np.sqrt(
                        chi2
                        / (
                            crss_sum
                            * np.sqrt((crss_tab.shape[0] - 1) * (crss_tab.shape[1] - 1))
                        )
                    )
                elif statistic == "Contingency Coefficient":
                    chi_table[i][j] = np.sqrt(chi2 / (crss_sum + chi2))

        chi_table = pd.DataFrame(chi_table, columns=cols, index=cols)
        fig = px.imshow(
            chi_table, color_continuous_scale="blues", title="Chi Square Values"
        )

        return fig if return_fig else fig.show() if fig is not None else None

    def wordcloud(self, return_fig=True):
        """Plot word clouds for the text variables"""
        n_rows = ceil(len(self.config.TEXT_VARIABLES) / 2)
        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, col in enumerate(self.config.TEXT_VARIABLES):
            self._wc(
                " ".join(self.data[col].fillna("").values),
                col,
                fig.add_subplot(n_rows, 2, i + 1),
            )
        return fig if return_fig else plt.show()

    def wordcloud_by_group(self, group_by=None, return_fig=True):
        """Plot grouped Word Clouds for the text variables"""
        n_rows = ceil(
            (len(self.config.TEXT_VARIABLES) * self.data[group_by].nunique()) / 2
        )
        fig = plt.figure(figsize=(20, 6 * n_rows))
        i = 0
        for col in self.config.TEXT_VARIABLES:
            for label in sorted(self.data[group_by].unique()):
                i += 1
                self._wc(
                    " ".join(
                        self.data[self.data[group_by] == label][col].fillna("").values
                    ),
                    f"{col} ({label})",
                    fig.add_subplot(n_rows, 2, i),
                )
        return fig if return_fig else plt.show()

    def wordcloud_freq(self, group_by=None, return_fig=True):
        """Plot grouped word clouds for the words unique to each group"""
        n_rows = ceil(
            (len(self.config.TEXT_VARIABLES) * self.data[group_by].nunique()) / 2
        )
        fig = plt.figure(figsize=(20, 6 * n_rows))
        i = 0
        for col in self.config.TEXT_VARIABLES:
            for label in sorted(self.data[group_by].unique()):
                i += 1
                self._wc_freq(
                    (
                        Counter(
                            " ".join(
                                self.data[self.data[group_by] == label][col]
                                .fillna("")
                                .values
                            ).split()
                        )
                        - Counter(
                            " ".join(
                                self.data[self.data[group_by] != label][col]
                                .fillna("")
                                .values
                            ).split()
                        )
                    ),
                    f"{col} ({label})",
                    fig.add_subplot(n_rows, 2, i),
                )
        return fig if return_fig else plt.show()

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

    def trivariate_bubbleplot(self, x, y, s, return_fig=True):
        """Plot a trivariate bubbleplot with three continuous variables
        :param x: continuous variable
        :param y: continuous variable
        :param s: continuous variable
        """
        num_cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            num_cols.append(self.config.DEPENDENT_VARIABLE)
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)
        if x in num_cols and y in num_cols and s in num_cols:
            fig = px.scatter(
                self.data,
                x=x,
                y=y,
                size=s,
                title=f"{y} vs {x} vs {s}",
            )
        else:
            print("x, y and s should be continuous variables")
            fig = None
        return fig if return_fig else fig.show() if fig is not None else None

    def trivariate_scatterplot(self, x, y, c, return_fig=True):
        """Plot a trivariate scatterplot with two continuous variables and a categorical variable
        :param x: continuous variable
        :param y: continuous variable
        :param c: categorical variable
        """
        num_cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            num_cols.append(self.config.DEPENDENT_VARIABLE)
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)
        if x in num_cols and y in num_cols and c in cat_cols:
            fig = px.scatter(
                self.data,
                x=x,
                y=y,
                color=c,
                title=f"{y} vs {x} vs {c}",
            )
        else:
            print(
                "x and y should be continuous variables and c should be a categorical variable"
            )
            fig = None
        return fig if return_fig else fig.show() if fig is not None else None

    def trivariate_violinplot(self, x, y, c, return_fig=True):
        """Plot a trivariate violinplot with two categorical variables and a continuous variable
        :param x: categorical variable
        :param y: continuous variable
        :param c: categorical variable
        """
        num_cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            num_cols.append(self.config.DEPENDENT_VARIABLE)
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)
        if x in cat_cols and y in num_cols and c in cat_cols:
            fig = px.violin(
                self.data,
                x=x,
                y=y,
                color=c,
                title=f"{y} vs {x} vs {c}",
            )
        else:
            print(
                "x and c should be categorical variables and y should be a continuous variable"
            )
            fig = None
        return fig if return_fig else fig.show() if fig is not None else None
