import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from math import ceil
from itertools import combinations, product
from collections import Counter

sns.set_style("darkgrid")


class Vizard:
    """Vizard object holds the `DataFrame` along with its configurations including the
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
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle(f"{x} Distribution", fontsize=24)
        ax = self.data[x].value_counts().plot.bar()
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        plt.show()

    def continuous(self, x):
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(f"{x} Distribution", fontsize=24)
        sns.histplot(self.data[x], kde=False)
        plt.show()

    def timeseries(self, x):
        fig, ax = plt.subplots(figsize=(20, 6))
        self.data[x].plot(ax=ax)
        plt.show()

    def categorical_vs_continuous(self, x, y):
        fig = plt.figure(figsize=(20, 8))
        ax = [
            plt.subplot2grid((2, 3), (0, 0), colspan=1),
            plt.subplot2grid((2, 3), (0, 1), colspan=2),
            plt.subplot2grid((2, 3), (1, 0), colspan=3),
        ]
        fig.suptitle(x, fontsize=18)
        self.data[x].value_counts().plot.pie(ax=ax[0], autopct="%1.1f%%")
        ax[0].legend()
        sns.histplot(data=self.data, x=y, hue=x, kde=False, ax=ax[1])
        sns.boxplot(y=y, x=x, data=self.data, ax=ax[2])
        plt.show()

    def categorical_vs_categorical(self, x, y):
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(x, fontsize=18)
        self.data[x].value_counts().plot.pie(ax=ax[0], autopct="%1.1f%%")
        ax[0].legend()
        sns.heatmap(
            pd.crosstab(index=self.data[y], columns=self.data[x]),
            ax=ax[1],
            cmap="Blues",
            annot=True,
            square=True,
            fmt="d",
        )
        plt.show()

    def continuous_vs_continuous(self, x, y):
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(x, fontsize=18)
        sns.histplot(self.data[x], ax=ax[0], kde=False)
        sns.scatterplot(x=x, y=y, data=self.data, ax=ax[1])
        plt.show()

    def continuous_vs_categorical(self, x, y):
        fig = plt.figure(figsize=(20, 12))
        ax = [
            plt.subplot2grid((3, 3), (0, 0), colspan=3),
            plt.subplot2grid((3, 3), (1, 0), colspan=3),
            plt.subplot2grid((3, 3), (2, 0)),
            plt.subplot2grid((3, 3), (2, 1), colspan=2),
        ]
        fig.suptitle(x, fontsize=18)
        sns.histplot(self.data[x], kde=False, ax=ax[0])
        sns.histplot(x=x, hue=y, data=self.data, kde=False, ax=ax[1])
        sns.boxplot(x=self.data[x], ax=ax[2])
        sns.boxplot(x=y, y=x, data=self.data, ax=ax[3])
        plt.show()

    def check_missing(self):
        """Plot a heatmap to visualize missing values in the DataFrame"""
        fig, ax = plt.subplots(figsize=(20, 6))
        fig.suptitle("Missing Values", fontsize=24)
        sns.heatmap(self.data.isnull(), cbar=False, yticklabels=False)
        plt.show()

    def count_missing(self):
        """Plot to visualize the count of missing values in each column in the DataFrame"""
        fig = plt.figure(figsize=(20, 4))
        fig.suptitle("Count of Missing Values", fontsize=24)
        ax = self.data.isna().sum().plot.bar()
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        plt.show()

    def count_missing_by_group(self, group_by=None):
        """Plot to visualize the count of missing values in each column in the DataFrame,
        grouped by a categorical variable
        :param group_by: a column in the DataFrame"""
        fig, ax = plt.subplots(figsize=(20, 4))
        fig.suptitle(f"Count of Missing Values Grouped By {group_by}", fontsize=24)
        self.data.drop(group_by, 1).isna().groupby(
            self.data[group_by], sort=False
        ).sum().T.plot.bar(ax=ax)
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        plt.show()

    def count_unique(self):
        """Plot to visualize the count of unique values in each column in the DataFrame"""
        fig = plt.figure(figsize=(20, 4))
        fig.suptitle("Count of Unique Values", fontsize=24)
        ax = self.data.nunique().plot.bar()
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        plt.show()

    def count_unique_by_group(self, group_by=None):
        """Plot to visualize the count of unique values in each column in the DataFrame,
        grouped by a categorical variable
        :param group_by: a column in the DataFrame"""
        fig, ax = plt.subplots(figsize=(20, 4))
        fig.suptitle(f"Count of Unique Values Grouped By {group_by}", fontsize=24)
        self.data.groupby(group_by).nunique().T.plot.bar(ax=ax)
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        plt.show()

    def dependent_variable(self):
        """Based on the type of problem, plot a univariate visualization of target column"""
        if self.config.PROBLEM_TYPE == "regression":
            self.continuous(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            self.categorical(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "time_series":
            self.timeseries(self.config.DEPENDENT_VARIABLE)
        else:
            print("Invalid Problem Type")

    def pairwise_scatter(self):
        """Plot pairwise scatter plots to visualize the continuous variables in the dataset"""
        cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            cols.append(self.config.DEPENDENT_VARIABLE)

        comb = list(combinations(cols, 2))

        n_plots = len(comb)
        n_rows = ceil(n_plots / 2)

        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, (x, y) in enumerate(comb):
            ax = fig.add_subplot(n_rows, 2, i + 1)
            if self.config.PROBLEM_TYPE == "regression":
                sns.scatterplot(data=self.data, x=x, y=y, ax=ax)
            elif self.config.PROBLEM_TYPE == "classification":
                sns.scatterplot(
                    data=self.data, x=x, y=y, hue=self.config.DEPENDENT_VARIABLE, ax=ax
                )
            else:
                pass
            ax.set_title(f"{x} vs {y}", fontsize=18)
        plt.show()

    def pairwise_violin(self):
        """Plot pairwise violin plots to visualize the continuous variables grouped by the
        categorical variables in the dataset"""
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)

        comb = list(product(self.config.CONTINUOUS_INDEPENDENT_VARIABLES, cat_cols))

        n_plots = len(comb)
        n_rows = ceil(n_plots / 2)

        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, (x, y) in enumerate(comb):
            ax = fig.add_subplot(n_rows, 2, i + 1)
            sns.violinplot(data=self.data, x=y, y=x)
            ax.set_title(f"{y} vs {x}", fontsize=18)
        plt.show()

    def pairwise_crosstabs(self):
        """Plot pairwise crosstabs plots to visualize the categorical variables in the dataset"""
        cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cols.append(self.config.DEPENDENT_VARIABLE)

        comb = list(combinations(cols, 2))

        n_plots = len(comb)
        n_rows = ceil(n_plots / 2)

        fig = plt.figure(figsize=(20, 6 * n_rows))
        for i, (x, y) in enumerate(comb):
            ax = fig.add_subplot(n_rows, 2, i + 1)
            sns.heatmap(
                pd.crosstab(index=self.data[y], columns=self.data[x]),
                ax=ax,
                cmap="Blues",
                annot=True,
                square=True,
                fmt="d",
            )
            ax.set_title(f"{y} vs {x}", fontsize=18)
        plt.show()

    def pair_plot(self):
        """Plot a pairplot to vizualize the complete DataFrame"""
        if self.config.PROBLEM_TYPE == "regression":
            g = sns.pairplot(
                data=self.data[
                    self.config.CONTINUOUS_INDEPENDENT_VARIABLES
                    + [self.config.DEPENDENT_VARIABLE]
                ]
            )
            g.fig.suptitle("Pairplot", fontsize=24, y=1.08)
            plt.show()
        elif self.config.PROBLEM_TYPE == "classification":
            g = sns.pairplot(
                data=self.data[
                    self.config.CONTINUOUS_INDEPENDENT_VARIABLES
                    + [self.config.DEPENDENT_VARIABLE]
                ],
                hue=self.config.DEPENDENT_VARIABLE,
            )
            g.fig.suptitle("Pairplot", fontsize=24, y=1.08)
            plt.show()
        else:
            pass

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

    def corr_plot(self, figsize=(10, 10)):
        """Plot a heatmap to vizualize the correlation between the various continuous columns in the DataFrame"""
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Correlation Plot", fontsize=24)

        corr = self.data.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = True

        annot = True if len(corr) <= 10 else False

        sns.heatmap(
            corr, mask=mask, cmap="RdBu", ax=ax, annot=annot, square=True, center=0
        )
        plt.show()

    def chi_sq_plot(self, figsize=(10, 10)):
        """Plot a heatmap to vizualize the chi 2 value between the varios categorical columns in the DataFrame"""
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Chi Square Plot", fontsize=24)

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

        mask = np.zeros_like(chi_sq_table, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = True

        annot = True if len(chi_sq_table) <= 10 else False

        sns.heatmap(
            chi_sq_table,
            mask=mask,
            cmap="Blues_r",
            ax=ax,
            annot=annot,
            square=True,
            center=0.05,
        )
        plt.show()

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