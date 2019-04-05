from TinyDashDataView import TinyDashDataView
import numpy as np

class TinyDashHistogram(TinyDashDataView):
    def _dict_data(self, data, label, color):
        x_data, y_data = np.histogram(data.values, bins=3)
        return {
            'x': x_data, 
            'y': y_data, 
            'type': 'bar', 
            'name': label,
            'color': color
            }