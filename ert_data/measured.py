import pandas as pd

from ert_data import loader


class MeasuredData(object):
    def __init__(self, facade, keys, index_lists=None, load_data=True):
        self._facade = facade
        self._set_data(self._get_data(keys, index_lists, load_data))

    @property
    def data(self):
        return self._data

    def _set_data(self, data):
        expected_keys = ["OBS", "STD"]
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Invalid type: {}, should be type: {}".format(type(data), pd.DataFrame)
            )
        elif not set(expected_keys).issubset(data.index):
            raise ValueError(
                "{} should be present in DataFrame index, missing: {}".format(
                    ["OBS", "STD"], set(expected_keys) - set(data.index)
                )
            )
        else:
            self._data = data

    def remove_failed_realizations(self):
        self._set_data(self._remove_failed_realizations())

    def get_simulated_data(self):
        return self._get_simulated_data()

    def _get_simulated_data(self):
        return self.data[~self.data.index.isin(["OBS", "STD"])]

    def _remove_failed_realizations(self):
        """Removes rows with no simulated data, leaving observations and
        standard deviations as-is."""
        pre_index = self.data.index
        post_index = list(self.data.dropna(axis=0, how="all").index)
        drop_index = set(pre_index) - set(post_index + ["STD", "OBS"])
        return self.data.drop(index=drop_index)

    def remove_inactive_observations(self):
        self._set_data(self._remove_inactive_observations())

    def _remove_inactive_observations(self):
        """Removes columns with one or more NaN values."""
        filtered_dataset = self.data.dropna(axis=1)
        if filtered_dataset.empty:
            raise ValueError(
                "This operation results in an empty dataset (could be due to one or more failed realizations)"
            )
        return filtered_dataset

    def is_empty(self):
        return self.data.empty

    def _get_data(self, observation_keys, index_lists, load_data=True):
        """
        Adds simulated and observed data and returns a dataframe where ensamble
        members will have a data key, observed data will be named OBS and
        observed standard deviation will be named STD.
        """
        measured_data = pd.DataFrame()
        case_name = self._facade.get_current_case_name()

        if index_lists is None:
            index_lists = [None] * len(observation_keys)

        if len(index_lists) != len(observation_keys):
            raise ValueError("index list must be same length as observations keys")

        for key, index_list in zip(observation_keys, index_lists):
            observation_type = self._facade.get_impl_type_name_for_obs_key(key)
            data_loader = loader.data_loader_factory(observation_type)

            data = data_loader(self._facade, key, case_name, load_data)

            # Simulated data and observations both refer to the data
            # index at some levels, so having that information available is
            # helpful
            _add_index_range(data)

            data = MeasuredData._filter_on_column_index(data, index_list)
            data = pd.concat({key: data}, axis=1)

            measured_data = pd.concat([measured_data, data], axis=1)

        return measured_data.astype(float)

    def filter_ensemble_std(self, std_cutoff):
        self._set_data(self._filter_ensemble_std(std_cutoff))

    def filter_ensemble_mean_obs(self, alpha):
        self._set_data(self._filter_ensemble_mean_obs(alpha))

    def _filter_ensemble_std(self, std_cutoff):
        """
        Filters on ensamble variation versus a user defined standard
        deviation cutoff. If there is not enough variation in the measurements
        the data point is removed.
        """
        ens_std = self.get_simulated_data().std()
        std_filter = ens_std <= std_cutoff
        return self.data.drop(columns=std_filter[std_filter].index)

    def _filter_ensemble_mean_obs(self, alpha):
        """
        Filters on distance between the observed data and the ensamble mean
        based on variation and a user defined alpha.
        """
        ens_mean = self.get_simulated_data().mean()
        ens_std = self.get_simulated_data().std()
        obs_values = self.data.loc["OBS"]
        obs_std = self.data.loc["STD"]

        mean_filter = abs(obs_values - ens_mean) > alpha * (ens_std + obs_std)

        return self.data.drop(columns=mean_filter[mean_filter].index)

    @staticmethod
    def _filter_on_column_index(dataframe, index_list):
        """
        Retuns a subset where the columns in index_list are filtered out
        """
        if isinstance(index_list, (list, tuple)):
            if max(index_list) > dataframe.shape[1]:
                msg = (
                    "Index list is larger than observation data, please check input, max index list:"
                    "{} number of data points: {}".format(
                        max(index_list), dataframe.shape[1]
                    )
                )
                raise IndexError(msg)
            return dataframe.iloc[:, list(index_list)]
        else:
            return dataframe


def _add_index_range(data):
    """
    Adds a second column index with which corresponds to the data
    index. This is because in libres simulated data and observations
    are connected through an observation key and data index, so having
    that information available when the data is joined is helpful.
    """
    arrays = [data.columns.to_list(), list(range(len(data.columns)))]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])
    data.columns = index
