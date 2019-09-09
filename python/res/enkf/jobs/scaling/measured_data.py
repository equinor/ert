import pandas as pd

from res.enkf.export import GenDataCollector, SummaryCollector, SummaryObservationCollector
from res.enkf.plot_data import PlotBlockDataLoader


class MeasuredData(object):

    def __init__(self, ert, events):
        self.data = self._get_data(ert, events.keys)
        self.remove_nan()
        self.filter_on_column_index(events.index)
        self.filter_out_outliers(events)

    def remove_nan(self):
        self.data = self.data.dropna(axis=1)

    def filter_on_column_index(self, index_list):
        self.data = self._filter_on_column_index(index_list)

    def filter_out_outliers(self, events):
        self.data = self._filter_out_outliers(events)

    def _get_data(self, ert, observation_keys):
        """
        Adds simulated and observed data and returns a dataframe where ensamble members will
        have a data key, observed data will be prefixed with OBS_ and observated standard
        deviation will be prefixed with STD_.
        """
        measured_data = pd.DataFrame()
        case_name = ert.getEnkfFsManager().getCurrentFileSystem().getCaseName()
        for key in observation_keys:
            observation_type = ert.getObservations()[key].getImplementationType().name
            if observation_type == "GEN_OBS":
                measured_data = measured_data.append(self._get_general_data(ert, key, case_name))
            elif observation_type == "SUMMARY_OBS":
                measured_data = measured_data.append(self._get_summary_data(ert, key, case_name))
            elif observation_type == "BLOCK_OBS":
                measured_data = measured_data.append(self._get_block_data(
                    ert.getObservations(), key, ert.getEnsembleSize(), ert.getEnkfFsManager().getCurrentFileSystem())
                )
            else:
                raise TypeError("Unknown observation type: {}".format(observation_type))
        return measured_data

    def _get_block_data(self, observations, key, ensamble_size, storage):
        obs_vector = observations[key]
        loader = PlotBlockDataLoader(obs_vector)

        data = pd.DataFrame()
        for report_step in obs_vector.getStepList().asList():

            block_data = loader.load(storage, report_step)
            obs_block = loader.getBlockObservation(report_step)

            data = (
                data
                .append(pd.DataFrame(
                    [self._get_block_observations(obs_block.getValue, obs_block)], index=["OBS_" + key]))
                .append(pd.DataFrame(
                    [self._get_block_observations(obs_block.getStd, obs_block)], index=["STD_" + key]))
                .append(self._get_block_measured(key, ensamble_size, block_data))
            )
        return data

    def _get_general_data(self, ert, observation_key, case_name):
        ert_obs = ert.getObservations()
        data_key = ert_obs[observation_key].getDataKey()

        general_data = pd.DataFrame()

        for time_step in ert_obs[observation_key].getStepList().asList():
            data = GenDataCollector.loadGenData(ert, case_name, data_key, time_step)

            general_data = (
               general_data
               .append(pd.DataFrame(self._get_observations(ert_obs, observation_key), index=["OBS_" + observation_key]))
               .append(pd.DataFrame(self._get_std(ert_obs, observation_key), index=["STD_" + observation_key]))
               .append(pd.concat([pd.DataFrame([data[key]], index=[observation_key]) for key in data.keys()]))
            )
        return general_data

    def _filter_on_column_index(self, index_list):
        """
        Retuns a subset where the columns in index_list are filtered out
        """
        if isinstance(index_list, (list, tuple)):
            if max(index_list) > self.data.shape[1]:
                msg = ("Index list is larger than observation data, please check input, max index list:"
                       "{} number of data points: {}".format(max(index_list), self.data.shape[1]))
                raise IndexError(msg)
            return self.data.iloc[:, list(index_list)]
        else:
            return self.data

    def _filter_out_outliers(self, events):
        """
        Goes through the observation keys and filters out outliers. It first extracts
        key data such as ensamble mean and std, and observation values and std. It creates
        a filtered index which is a pandas series of indexes where the data will be removed.
        This can have duplicates of indicies.
        """
        filters = []

        for key in events.keys:

            ens_mean = self.data.loc[key].mean(axis=0)
            ens_std = self.data.loc[key].std(axis=0)
            obs_values = self.data.loc["OBS_" + key]
            obs_std = self.data.loc["STD_" + key]

            filters.append(self._filter_ensamble_std(ens_std, events.std_cutoff))
            filters.append(self._filter_ens_mean_obs(ens_mean, ens_std, obs_values, obs_std, events.alpha))

        combined_filter = self._combine_filters(filters)
        return self.data.drop(columns=combined_filter[combined_filter].index)

    def _get_summary_data(self, ert, observation_key, case_name):
        data_key = ert.getObservations()[observation_key].getDataKey()
        return pd.concat([self._add_summary_observations(ert, data_key, observation_key, case_name),
                          self._add_summary_data(ert, data_key, observation_key, case_name)])

    @staticmethod
    def _filter_ensamble_std(ensamble_std, std_cutoff):
        """
        Filters on ensamble variation versus a user defined standard
        deviation cutoff.
        """
        return ensamble_std < std_cutoff

    @staticmethod
    def _filter_ens_mean_obs(ensamble_mean, ensamble_std, observation_values, observation_std, alpha):
        """
        Filters on distance between the observed data and the ensamble mean based on variation and
        a user defined alpha.
        """
        return abs(observation_values - ensamble_mean) > alpha * (ensamble_std + observation_std)

    @staticmethod
    def _combine_filters(filters):
        combined_filter = pd.Series()
        for filter in filters:
            combined_filter = filter | combined_filter
        return combined_filter

    @staticmethod
    def _add_summary_data(ert, data_key, observation_key, case_name):
        data = SummaryCollector.loadAllSummaryData(ert, case_name, [data_key])
        data = data[data_key].unstack(level=-1)
        return data.set_index([[observation_key] * len(data)])

    @staticmethod
    def _add_summary_observations(ert, data_key, observation_key, case_name):
        data = SummaryObservationCollector.loadObservationData(ert, case_name, [data_key]).transpose()
        data = data.set_index(data.index.str.replace(r"\b" + data_key, "OBS_" + data_key, regex=True))
        return data.set_index(data.index.str.replace(data_key, observation_key))

    @staticmethod
    def _get_block_observations(func, observation_block):
        return [func(nr) for nr in observation_block]

    @staticmethod
    def _get_block_measured(key, ensamble_size, block_data):
        data = pd.DataFrame()
        for ensamble_nr in range(ensamble_size):
            data = data.append(pd.DataFrame([block_data[ensamble_nr]], index=[key]))
        return data

    @staticmethod
    def _get_observations(all_obs, obs_key):
        return [obs_node.get_data_points() for obs_node in all_obs[obs_key]]

    @staticmethod
    def _get_std(all_obs, obs_key):
        return [obs_node.get_std() for obs_node in all_obs[obs_key]]
