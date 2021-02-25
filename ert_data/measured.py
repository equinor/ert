from collections import defaultdict

import pandas as pd
import numpy as np

from ert_data import loader


class MeasuredData(object):
    def __init__(self, facade, keys, index_lists=None, load_data=True, case_name=None):
        self._facade = facade

        if not keys:
            raise loader.ObservationError("No observation keys provided")
        if case_name is None:
            case_name = self._facade.get_current_case_name()
        if index_lists is not None and len(index_lists) != len(keys):
            raise ValueError("index list must be same length as observations keys")

        self._set_data(self._get_data(keys, load_data, case_name))
        self._set_data(self.filter_on_column_index(keys, index_lists))

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

    def _get_data(self, observation_keys, load_data, case_name):
        """
        Adds simulated and observed data and returns a dataframe where ensamble
        members will have a data key, observed data will be named OBS and
        observed standard deviation will be named STD.
        """

        # Because several observations can be linked to the same response we create
        # a grouping to avoid reading the same response for each of the corresponding
        # observations, as that is quite slow.
        key_map = defaultdict(list)
        for key in observation_keys:
            try:
                data_key = self._facade.get_data_key_for_obs_key(key)
            except KeyError:
                raise loader.ObservationError(f"No data key for obs key: {key}")
            key_map[data_key].append(key)

        measured_data = []

        for obs_keys in key_map.values():
            obs_types = [
                self._facade.get_impl_type_name_for_obs_key(key) for key in obs_keys
            ]
            assert (
                len(set(obs_types)) == 1
            ), f"\nMore than one observation type found for obs keys: {obs_keys}"
            observation_type = obs_types[0]
            data_loader = loader.data_loader_factory(observation_type)
            data = data_loader(
                self._facade, obs_keys, case_name, include_data=load_data
            )
            if data.empty:
                raise loader.ObservationError(f"No observations loaded for {obs_keys}")
            measured_data.append(data)

        return pd.concat(measured_data, axis=1)

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

    def filter_on_column_index(self, obs_keys, index_lists):
        if index_lists is None or all(index_list is None for index_list in index_lists):
            return self.data
        names = self.data.columns.get_level_values(0)
        data_index = self.data.columns.get_level_values("data_index")
        cond = self._create_condition(names, data_index, obs_keys, index_lists)
        return self.data.iloc[:, cond]

    @staticmethod
    def _filter_on_column_index(dataframe, index_list):
        """
        Returns a subset where the columns in index_list are filtered out
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

    @staticmethod
    def _create_condition(names, data_index, obs_keys, index_lists):
        conditions = []
        for obs_key, index_list in zip(obs_keys, index_lists):
            if index_list is not None:
                index_cond = [data_index == index for index in index_list]
                index_cond = np.logical_or.reduce(index_cond)
                conditions.append(np.logical_and(index_cond, (names == obs_key)))
        return np.logical_or.reduce(conditions)
