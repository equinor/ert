// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'canvas_plot.js' is part of ERT - Ensemble based Reservoir Tool.
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


function Plot(element) {
    this.plot = new BasePlot(element);

    this.line_renderers = [];
    for (var index = 1; index <= 5; index++) {
        var renderer = this.plot.createLineRenderer();
        renderer.style(STYLES["ensemble_" + (index)]);
        this.line_renderers.push(renderer)
    }

    this.tracker = new IncrementalRenderTracker();

    var self = this;

    var progressivePreRenderer = function(context, data) {
        var case_list = data.caseList();

        for(var case_index = 0; case_index < case_list.length; case_index++) {
            var style = STYLES["ensemble_" + (case_index + 1)];
            var case_name = case_list[case_index];

            self.plot.addLegend(style, case_name, CanvasPlotLegend.simpleLine);
        }

        self.progressiveRenderer(context, data, self, 0, 0);
    };

    var renderEnsembleProgressively = function(context, data) {
        if(data.hasEnsembleData()) {
            self.tracker.start(function() { progressivePreRenderer(context, data); });
        }
    };

    this.progressiveRenderer = function(context, data, self, case_index, realization) {
        if(self.tracker.shouldStop()) {
            self.tracker.stoppedRendering();
            return;
        }

        var case_list = data.caseList();
        var case_name = case_list[case_index];
        var line_renderer = self.line_renderers[case_index];

        var ensemble_data = data.ensembleData(case_name);
        var x_values = ensemble_data.xValues();
        var y_values_list = ensemble_data.yValues();

        self.tracker.loopStart();
        for (var i = realization; i < y_values_list.length; i++) {
            line_renderer(context, x_values, y_values_list[i]);
            realization++;

            if(self.tracker.shouldLoopStop()) {
                break;
            }
        }

        if(realization == y_values_list.length) {
            case_index++;
            realization = 0;
        }

        if (case_index < case_list.length) {
            window.setTimeout(function() {
                self.progressiveRenderer(context, data, self, case_index, realization);
            }, 15);
        } else {
            self.tracker.stoppedRendering();
        }
    };


//    var renderEnsembleDirect = function(context, data) {
//        if(data.hasEnsembleData()) {
//            var case_list = data.caseList();
//
//            for(var case_index = 0; case_index < case_list.length; case_index++) {
//                var style = STYLES["ensemble_" + (case_index + 1)];
//                var case_name = case_list[case_index];
//                var line_renderer = self.line_renderers[case_index];
//
//                var x_values = data.ensembleXValues(case_name);
//                var y_values_list = data.ensembleYValues(case_name);
//
//                for (var realization = 0; realization < y_values_list.length; realization++) {
//                    line_renderer(context, x_values[realization], y_values_list[realization]);
//                }
//
//                self.plot.addLegend(style, case_name, CanvasPlotLegend.simpleLine);
//            }
//        }
//    };

    this.plot.setRenderCallback(renderEnsembleProgressively);
}

Plot.prototype.resize = function(width, height) {
    this.plot.resize(width, height);
};

Plot.prototype.setYScales = function(min, max) {
    this.plot.setYScales(min, max);
};

Plot.prototype.setYDomain = function(min_y, max_y) {
    this.plot.setYDomain(min_y, max_y);
};

Plot.prototype.setXDomain = function(min_x, max_x) {
    this.plot.setXDomain(min_x, max_x);
};

Plot.prototype.setData = function(data) {
    this.plot.setData(data);
};
