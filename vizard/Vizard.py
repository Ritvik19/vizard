import numpy as np
import pandas as pd

from math import ceil
from itertools import combinations, product
from collections import Counter
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

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
        return fig

    def continuous(self, x):
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(f"{x} Distribution", fontsize=24)
        sns.histplot(self.data[x], kde=False)
        return fig

    def timeseries(self, x):
        fig, ax = plt.subplots(figsize=(20, 6))
        self.data[x].plot(ax=ax)
        return fig

    def categorical_vs_continuous(self, x, y, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def categorical_vs_categorical(self, x, y, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def continuous_vs_continuous(self, x, y, return_fig=True):
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(x, fontsize=18)
        sns.histplot(self.data[x], ax=ax[0], kde=False)
        sns.scatterplot(x=x, y=y, data=self.data, ax=ax[1])
        return fig if return_fig else plt.show() if fig is not None else None

    def continuous_vs_categorical(self, x, y, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def check_missing(self, return_fig=True):
        """Plot a heatmap to visualize missing values in the DataFrame"""
        fig, ax = plt.subplots(figsize=(20, 6))
        fig.suptitle("Missing Values", fontsize=24)
        sns.heatmap(self.data.isnull(), cbar=False, yticklabels=False)
        return fig if return_fig else plt.show() if fig is not None else None

    def count_missing(self, return_fig=True):
        """Plot to visualize the count of missing values in each column in the DataFrame"""
        fig = plt.figure(figsize=(20, 4))
        fig.suptitle("Count of Missing Values", fontsize=24)
        ax = self.data.isna().sum().plot.bar()
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        return fig if return_fig else plt.show() if fig is not None else None

    def count_missing_by_group(self, group_by=None, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def count_unique(self, return_fig=True):
        """Plot to visualize the count of unique values in each column in the DataFrame"""
        fig = plt.figure(figsize=(20, 4))
        fig.suptitle("Count of Unique Values", fontsize=24)
        ax = self.data.nunique().plot.bar()
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        return fig if return_fig else plt.show() if fig is not None else None

    def count_unique_by_group(self, group_by=None, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def dependent_variable(self, return_fig=True):
        """Based on the type of problem, plot a univariate visualization of target column"""
        if self.config.PROBLEM_TYPE == "regression":
            fig = self.continuous(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            fig = self.categorical(self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "time_series":
            fig = self.timeseries(self.config.DEPENDENT_VARIABLE)
        else:
            print("Invalid Problem Type")
            fig = None
        return fig if return_fig else plt.show() if fig is not None else None

    def pairwise_scatter(self, return_fig=True):
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
            sns.scatterplot(data=self.data, x=x, y=y, ax=ax)
            ax.set_title(f"{y} vs {x}", fontsize=18)
        return fig if return_fig else plt.show() if fig is not None else None

    def pairwise_violin(self, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def pairwise_crosstabs(self, return_fig=True):
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
        return fig if return_fig else plt.show() if fig is not None else None

    def pair_plot(self, return_fig=True):
        """Plot a pairplot to vizualize the complete DataFrame"""
        if self.config.PROBLEM_TYPE == "regression":
            g = sns.pairplot(
                data=self.data[
                    self.config.CONTINUOUS_INDEPENDENT_VARIABLES
                    + [self.config.DEPENDENT_VARIABLE]
                ]
            )
            g.fig.suptitle("Pairplot", fontsize=24, y=1.08)
            return g if return_fig else plt.show()
        elif self.config.PROBLEM_TYPE == "classification":
            g = sns.pairplot(
                data=self.data[
                    self.config.CONTINUOUS_INDEPENDENT_VARIABLES
                    + [self.config.DEPENDENT_VARIABLE]
                ],
                hue=self.config.DEPENDENT_VARIABLE,
            )
            g.fig.suptitle("Pairplot", fontsize=24, y=1.08)
            return g if return_fig else plt.show()
        elif self.config.PROBLEM_TYPE == "unsupervised":
            g = sns.pairplot(
                data=self.data[
                    self.config.CONTINUOUS_INDEPENDENT_VARIABLES
                ]
            )
            g.fig.suptitle("Pairplot", fontsize=24, y=1.08)
            return g if return_fig else plt.show()
        else:
            pass

    def categorical_variables(self):
        """Create bivariate visualizations for the categorical variables with the target variable or
            or univariate visualizations in case of unservised problems
        """
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_continuous(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical_vs_categorical(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == 'unsupervised':
            for col in self.config.CATEGORICAL_INDEPENDENT_VARIABLES:
                self.categorical(col)
        else:
            pass

    def continuous_variables(self):
        """Create bivariate visualizations for the continuous variables with the target variable
            or univariate visualizations in case of unservised problems
        """
        if self.config.PROBLEM_TYPE == "regression":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_continuous(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == "classification":
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous_vs_categorical(col, self.config.DEPENDENT_VARIABLE)
        elif self.config.PROBLEM_TYPE == 'unsupervised':
            for col in self.config.CONTINUOUS_INDEPENDENT_VARIABLES:
                self.continuous(col)
        else:
            pass

    def corr_plot(self, figsize=(10, 10), method="pearson", return_fig=True):
        """Plot a heatmap to vizualize the correlation between the various continuous columns in the DataFrame
        :param method: the method of correlation {'pearson', 'kendall', 'spearman'}
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Correlation Plot", fontsize=24)

        corr = self.data.corr(method=method)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = True

        annot = True if len(corr) <= 10 else False

        sns.heatmap(
            corr, mask=mask, cmap="RdBu", ax=ax, annot=annot, square=True, center=0
        )
        return fig if return_fig else plt.show() if fig is not None else None

    def point_biserial_plot(self, figsize=(10, 10), return_fig=True):
        """Plot a heatmap to visualize point biserial correaltion between continuous and categorical variables"""
        num_cols = self.config.CONTINUOUS_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "regression":
            num_cols.append(self.config.DEPENDENT_VARIABLE)
        cat_cols = self.config.CATEGORICAL_INDEPENDENT_VARIABLES[:]
        if self.config.PROBLEM_TYPE == "classification":
            cat_cols.append(self.config.DEPENDENT_VARIABLE)

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Point Biserial Plot", fontsize=24)

        pb_table = np.zeros((len(cat_cols), len(num_cols)))
        for i in range(len(cat_cols)):
            for j in range(len(num_cols)):
                df_ = self.data.dropna(subset=[cat_cols[i], num_cols[j]])
                pb_table[i][j] = pointbiserialr(
                    LabelEncoder().fit_transform(df_[cat_cols[i]]), df_[num_cols[j]]
                )[0]

        annot = True if max(pb_table.shape) <= 10 else False
        pb_table = pd.DataFrame(pb_table, columns=num_cols, index=cat_cols)
        sns.heatmap(pb_table, cmap="RdBu", ax=ax, annot=annot, square=True, center=0)
        return fig if return_fig else plt.show() if fig is not None else None

    def chi_sq_plot(self, statistic="Chi Sq", figsize=(10, 10), return_fig=True):
        """Plot a heatmap to vizualize the chi 2 value between the various categorical columns in the DataFrame"""
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("Chi Square Plot", fontsize=24)

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

        mask = np.zeros_like(chi_table, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = True

        annot = True if len(chi_table) <= 10 else False

        sns.heatmap(
            chi_table,
            mask=mask,
            cmap="Blues_r",
            ax=ax,
            annot=annot,
            square=True,
            center=0.05,
        )
        return fig if return_fig else plt.show() if fig is not None else None

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
        return fig if return_fig else plt.show() if fig is not None else None

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
        return fig if return_fig else plt.show() if fig is not None else None

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
        return fig if return_fig else plt.show() if fig is not None else None

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

    def trivariate_bubbleplot(self, x, y, s, figsize=(10, 10), return_fig=True):
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
            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=self.data, x=x, y=y, size=s, ax=ax)
            fig.suptitle(f"{y} vs {x} vs {s}", fontsize=18)
        else:
            print("x, y and s should be continuous variables")
            fig = None
        return fig if return_fig else plt.show() if fig is not None else None

    def trivariate_scatterplot(self, x, y, c, figsize=(10, 10), return_fig=True):
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
            fig, ax = plt.subplots(figsize=figsize)
            sns.scatterplot(data=self.data, x=x, y=y, hue=c, ax=ax)
            fig.suptitle(f"{y} vs {x} vs {c}", fontsize=18)
        else:
            print(
                "x and y should be continuous variables and c should be a categorical variable"
            )
            fig = None
        return fig if return_fig else plt.show() if fig is not None else None

    def trivariate_violinplot(self, x, y, c, figsize=(10, 10), return_fig=True):
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
            fig, ax = plt.subplots(figsize=figsize)
            sns.violinplot(data=self.data, x=x, y=y, hue=c, ax=ax)
            fig.suptitle(f"{y} vs {x} vs {c}", fontsize=18)
        else:
            print(
                "x and c should be categorical variables and y should be a continuous variable"
            )
            fig = None
        return fig if return_fig else plt.show() if fig is not None else None
