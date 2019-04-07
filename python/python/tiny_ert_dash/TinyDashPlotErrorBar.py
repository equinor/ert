import pandas as pd
class TinyDashPlotErrorBar:
    def get_figure(self, tem_data, title, case_color):
        layout = self.get_layout()
        layout['title'] = title
        return {'data': self.get_data(tem_data, case_color), 'layout': layout}

    def get_data(self, tem_data, case_color):
        l = []
        for case in tem_data:
            data = tem_data[case]
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    std = data.std(axis=1)
                    mean = data.mean(axis=1)
                    x_data_time = data.index
                    y_data_val = mean
                    l = l + [{
                        'x': x_data_time,
                        'y': y_data_val,
                        'error_y': {
                            'type':'data',
                            'symmetric': True,
                            'array': std.values
                        },
                        'name':  '{}'.format(case),
                        'line': {
                            'width': 3,
                            'shape': 'spline',
                            'color': case_color[case]
                        }}]
        return l
    
    def get_layout(self): 
        return {'margin': {'l': 10,'r': 10,'b': 25,'t': 25}}