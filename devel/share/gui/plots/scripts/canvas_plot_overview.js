// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'canvas_plot_overview.js' is part of ERT - Ensemble based Reservoir Tool.
//
// ERT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ERT is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.
//
// See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
// for more details.


function OverviewPlot(element) {
    this.plot = new BasePlot(element);

    var self = this;

    var renderEnsemble = function(context, data) {
        if(data.hasEnsembleData()) {
            var case_list = data.caseList();

            for(var i = 0; i < case_list.length; i++) {
                var style = STYLES[("ensemble_" + (i + 1))];
                var case_name = case_list[i];
                var ensemble_data = data.ensembleData(case_name);

                var x_values = ensemble_data.xValues();

                var y_min_values = ensemble_data.yMinValues();
                var y_max_values = ensemble_data.yMaxValues();

                var x_area_values = [];
                var y_area_values = [];


                for (var j = 0; j < x_values.length; j++) {
                    x_area_values.push(x_values[j]);
                    y_area_values.push(y_min_values[j]);
                }

                for (var k = x_values.length - 1; k >= 0; k--) {
                    x_area_values.push(x_values[k]);
                    y_area_values.push(y_max_values[k]);
                }


                self.plot.area_renderer.style(style);
                self.plot.area_renderer(context, x_area_values, y_area_values);

                self.plot.addLegend(style, case_name, CanvasPlotLegend.filledCircle);
            }

        }
    };


    this.plot.setRenderCallback(renderEnsemble);
}

OverviewPlot.prototype.resize = function(width, height) {
    this.plot.resize(width, height);
};

OverviewPlot.prototype.setYScales = function(min, max) {
    this.plot.setYScales(min, max);
};

OverviewPlot.prototype.setYDomain = function(min_y, max_y) {
    this.plot.setYDomain(min_y, max_y);
};

OverviewPlot.prototype.setXDomain = function(min_x, max_x) {
    this.plot.setXDomain(min_x, max_x);
};

OverviewPlot.prototype.setData = function(data) {
    this.plot.setData(data);
};

