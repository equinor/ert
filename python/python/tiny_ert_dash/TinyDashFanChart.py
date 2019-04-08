import pandas as pd

class TinyDashFanChart:
    def get_figure(self, tem_data, title, case_color):
        return self.get_data(tem_data, case_color)

    def get_data(self, tem_data, case_color):
        l = pd.DataFrame()
        for case in tem_data:
            data = tem_data[case]
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    mean = data.mean(axis=1)
                    p10 = data.quantile(0.1, axis=1)
                    p90 = data.quantile(0.9, axis=1)
                    min_val = data.min(axis=1)
                    max_val = data.max(axis=1)
                    name = '{}'.format(case)
                    data = {
                        'name': [name]*len(data.values),
                        'index': data.index.tolist(),
                        'mean': mean.values.tolist(),
                        'p10': p10.values.tolist(),
                        'p90': p90.values.tolist(),
                        'min': min_val.values.tolist(),
                        'max': max_val.values.tolist()
                        }
                    l = l.append(pd.DataFrame(data))
        return l.iterrows()