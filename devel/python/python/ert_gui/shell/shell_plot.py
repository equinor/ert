from math import ceil, sqrt, floor

import matplotlib.pyplot as plt
from pandas import DataFrame
import pylab
import numpy
from scipy.stats import gaussian_kde


class ShellPlot(object):
    @staticmethod
    def plot(data, value_column, observation_data=None, color=None, legend=False):
        if color is None:
            clist = plt.rcParams['axes.color_cycle']
            color = clist[0]

        data = data.reset_index()
        data = data.pivot(index="Date", columns="Realization", values=value_column)

        figure = plt.figure()
        figure.autofmt_xdate()

        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.xticks(rotation=30)
        plt.title(value_column)
        plt.plot_date(x=data.index.values, y=data, color=color, alpha=0.8, marker=None, linestyle="-")

        if observation_data is not None:
            observation_data.dropna(inplace=True)
            plt.errorbar(x=observation_data.index.values, y=observation_data[value_column], yerr=observation_data["STD_%s" %value_column],
                         fmt='none', ecolor='k', alpha=0.8)



    @staticmethod
    def plotArea(data, value_column, observation_data=None, color=None):
        if color is None:
            clist = plt.rcParams['axes.color_cycle']
            color = [clist[0]]

        data = data.reset_index()
        data = data.pivot(index="Date", columns="Realization", values=value_column)

        df = DataFrame()

        df["Minimum"] = data.min(axis=1)
        df["Maximum"] = data.max(axis=1)

        figure = plt.figure()
        figure.autofmt_xdate()
        plt.fill_between(df.index.values, df["Minimum"].values, df["Maximum"].values, alpha=0.8, color=color)
        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.xticks(rotation=30)
        plt.title(value_column)

        if observation_data is not None:
            observation_data.dropna(inplace=True)
            plt.errorbar(x=observation_data.index.values, y=observation_data[value_column], yerr=observation_data["STD_%s" %value_column],
                         fmt='none', ecolor='k', alpha=0.8)


    @staticmethod
    def plotQuantiles(data, value_column, observation_data=None, color=None):
        if color is None:
            clist = plt.rcParams['axes.color_cycle']
            color = clist[0]

        data = data.reset_index()
        data = data.pivot(index="Date", columns="Realization", values=value_column)

        df = DataFrame()

        df["Minimum"] = data.min(axis=1)
        df["Maximum"] = data.max(axis=1)
        df["Mean"] = data.mean(axis=1)
        df["p10"] = data.quantile(0.1, axis=1)
        df["p33"] = data.quantile(0.33, axis=1)
        df["p50"] = data.quantile(0.50, axis=1)
        df["p67"] = data.quantile(0.67, axis=1)
        df["p90"] = data.quantile(0.90, axis=1)

        figure = plt.figure()
        figure.autofmt_xdate()
        plt.plot(df.index.values, df["Minimum"].values, alpha=1, linestyle="--", color=color)
        plt.plot(df.index.values, df["Maximum"].values, alpha=1, linestyle="--", color=color)
        plt.plot(df.index.values, df["p50"].values, alpha=1, linestyle="--", color=color)
        plt.fill_between(df.index.values, df["p10"].values, df["p90"].values, alpha=0.3, color=color)
        plt.fill_between(df.index.values, df["p33"].values, df["p67"].values, alpha=0.5, color=color)

        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.xticks(rotation=30)
        plt.title(value_column)

        if observation_data is not None:
            observation_data.dropna(inplace=True)
            plt.errorbar(x=observation_data.index.values, y=observation_data[value_column], yerr=observation_data["STD_%s" %value_column],
                         fmt='none', ecolor='k', alpha=0.8)

    @staticmethod
    def histogram(data, name, log_on_x=False):
        bins = int(ceil(sqrt(len(data.index))))

        if log_on_x:
            bins = ShellPlot._histogramLogBins(data, bins)

        plt.figure()
        plt.hist(data[name].values, alpha=0.8, bins=bins)
        plt.ylabel("Count")
        plt.title(name)

        if log_on_x:
            plt.xticks(bins, ["$10^{%s}$" % (int(value) if value.is_integer() else "%.1f" % value) for value in bins]) #LaTeX formatting

    @staticmethod
    def density(data, name):
        values = data[name].values
        sample_range = values.max() - values.min()
        indexes = numpy.linspace(values.min() - 0.5 * sample_range, values.max() + 0.5 * sample_range, 1000)
        gkde = gaussian_kde(values)
        evaluated_gkde = gkde.evaluate(indexes)

        plt.figure()
        plt.title(name)
        plt.ylabel("Density")
        plt.plot(indexes, evaluated_gkde)


    @staticmethod
    def _histogramLogBins(data, bin_count):
        """
        @type data: pandas.DataFrame
        @rtype: int
        """
        data = data[data.columns[0]]

        min_value = int(floor(float(data.min())))
        max_value = int(ceil(float(data.max())))

        log_bin_count = max_value - min_value

        if log_bin_count < bin_count:
            next_bin_count = log_bin_count * 2

            if bin_count - log_bin_count > next_bin_count - bin_count:
                log_bin_count = next_bin_count
            else:
                log_bin_count = bin_count

        return numpy.linspace(min_value, max_value, log_bin_count)