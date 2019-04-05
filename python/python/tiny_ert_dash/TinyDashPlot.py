from TinyDashDataView import TinyDashDataView

class TinyDashPlot(TinyDashDataView):
    def _dict_data(self, data, label, color):
        x_data_time = data.index
        y_data_val = data.values
        return {
            'x': x_data_time,
            'y': y_data_val,
            'name': label,
            'line': {
                'width': 3,
                'shape': 'spline',
                'color': color
                }}