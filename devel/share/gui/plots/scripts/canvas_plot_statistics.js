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


function StatisticsPlot(element, x_dimension, y_dimension) {
    this.plot = new BasePlot(element, x_dimension, y_dimension);
    this.plot.setRenderObservations(false);
    this.plot.setRenderRefcase(false);

    this.horizontal_draw_direction = true;

    this.line_renderers = [];
    for (var index = 1; index <= 5; index++) {
        var renderer = this.plot.createLineRenderer();
        renderer.style(STYLES["ensemble_" + (index)]);
        this.line_renderers.push(renderer)
    }

    this.stippled_line_renderers = [];
    for (var index = 1; index <= 5; index++) {
        var renderer = this.plot.createStippledLineRenderer();
        renderer.style(STYLES["ensemble_" + (index)]);
        this.stippled_line_renderers.push(renderer)
    }

    var self = this;

    var renderEnsemble = function (context, data) {
        if (data.hasEnsembleData()) {
            var case_list = data.caseList();

            for (var i = 0; i < case_list.length; i++) {
                var style = STYLES[("ensemble_" + (i + 1))];
                var case_name = case_list[i];
                var ensemble_data = data.ensembleData(case_name);

                var values = ensemble_data.xValues();
                if (!self.horizontal_draw_direction) {
                    values = ensemble_data.yValues();
                }

                var y_min_values = ensemble_data.yMinValues();
                var y_max_values = ensemble_data.yMaxValues();
                var median = ensemble_data.xPercentile(0.5);
                var p10 = ensemble_data.xPercentile(0.1);
                var p90 = ensemble_data.xPercentile(0.9);

                var line_renderer = self.line_renderers[i];
                var stippled_line_renderer = self.stippled_line_renderers[i];
                if (self.horizontal_draw_direction) {
                    line_renderer(context, values, y_max_values);
                    line_renderer(context, values, y_min_values);
                    stippled_line_renderer(context, values, median);
                    line_renderer(context, values, p10);
                    line_renderer(context, values, p90);
                } else {
                    line_renderer(context, y_max_values, values);
                    line_renderer(context, y_min_values, values);
                }


                self.plot.addLegend(style, case_name, CanvasPlotLegend.filledCircle);
            }
            self.plot.renderCallbackFinishedRendering();
        }
    };


    this.plot.setRenderCallback(renderEnsemble);
}

StatisticsPlot.prototype.resize = function (width, height) {
    this.plot.resize(width, height);
};

StatisticsPlot.prototype.setScales = function (x_min, x_max, y_min, y_max) {
    this.plot.setScales(x_min, x_max, y_min, y_max);
};

StatisticsPlot.prototype.setData = function (data) {
    this.plot.setData(data);
};


StatisticsPlot.prototype.setVerticalErrorBar = function (vertical) {
    this.plot.setVerticalErrorBar(vertical);
};

StatisticsPlot.prototype.setHorizontalDrawDirection = function (horizontal) {
    this.horizontal_draw_direction = horizontal;
};

StatisticsPlot.prototype.setCustomSettings = function (settings) {
    this.plot.setCustomSettings(settings);
};

StatisticsPlot.prototype.setRenderingFinishedCallback = function(callback) {
    this.plot.setRenderingFinishedCallback(callback);
};

StatisticsPlot.prototype.renderNow = function(){
    this.plot.render();
};

StatisticsPlot.prototype.getTitle = function(){
    return this.plot.getTitle();
};

