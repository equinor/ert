// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'base_plot.js' is part of ERT - Ensemble based Reservoir Tool.
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

function BasePlot(element) {
    this.stored_data = [];
    this.margin = {left: 90, right: 20, top: 20, bottom: 30};
    this.root_elemenet = element;

    this.custom_y_min = null;
    this.custom_y_max = null;


    var group = this.root_elemenet.append("div")
        .attr("class", "plot");

    this.title = group.append("div")
        .attr("class", "plot-title")
        .text("No data");

    var plot_area = group.append("div").attr("class", "plot-area");

    this.width = 1024 - this.margin.left - this.margin.right;
    this.height = 512 - this.margin.top - this.margin.bottom;

    this.canvas = plot_area.append("canvas")
        .attr("id", "plot-canvas")
        .attr("width", this.width)
        .attr("height", this.height)
        .style("position", "absolute")
        .style("top", (this.margin.top) + "px")
        .style("left", (this.margin.left) + "px")
        .style("z-index", 5);

    this.plot_group = plot_area.append("svg")
        .attr("class", "plot-svg")
        .style("z-index", 10);

    this.legend_group = group.append("div")
        .attr("class", "plot-legend-group");

    this.x_scale = d3.time.scale().range([0, this.width]);
    this.x_time_scale = d3.time.scale().range([0, this.width]);
    this.y_scale = d3.scale.linear().range([this.height, 0]).nice();


    this.y_axis = d3.svg.axis()
        .scale(this.y_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("left")
        .tickSize(-this.width, -this.width);

    this.x_axis = d3.svg.axis()
        .scale(this.x_time_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("bottom")
        .tickSubdivide(true);

    this.plot_group.append("g")
        .attr("class", "y axis pale")
        .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")")
        .call(this.y_axis);

    this.plot_group.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(" + this.margin.left + ", " + (this.height + this.margin.top) + ")")
        .call(this.x_axis);


    var self = this;


    this.x = function (d) {
        return self.x_scale(d);
    };

    this.y = function (d) {
        return self.y_scale(d);
    };

    this.legend = CanvasPlotLegend();
    this.legend_list = [];

    this.line_renderer = CanvasPlotLine().x(this.x).y(this.y);
    this.area_renderer = CanvasPlotArea().x(this.x).y(this.y);
    this.error_bar_renderer = CanvasErrorBar().x(this.x).y(this.y);
    this.circle_renderer = CanvasCircle().x(this.x).y(this.y);

    this.render_callback = null;

//    console.log("BasePlot initialized!");
}

BasePlot.prototype.resize = function(width, height) {
    //Some magic margins...
    width = width - 80;
    height = height - 70;

    this.width = width - this.margin.left - this.margin.right;
    this.height = height - this.margin.top - this.margin.bottom;

    this.x_scale.range([0, this.width]);
    this.x_time_scale.range([0, this.width]);
    this.y_scale.range([this.height, 0]).nice();

    this.y_axis.tickSize(-this.width, -this.width);

    this.plot_group.style("width", width + "px").style("height", height + "px");

    this.canvas.attr("width", this.width).attr("height", this.height);

    this.plot_group.select(".x.axis").attr("transform", "translate(" + this.margin.left + ", " + (this.height + this.margin.top) + ")");

    this.setData(this.stored_data);
};

BasePlot.prototype.setYScales = function(min, max) {
    this.custom_y_min = min;
    this.custom_y_max = max;

    this.setData(this.stored_data);
};


BasePlot.prototype.setYDomain = function(min_y, max_y) {
    var min = min_y;
    var max = max_y;

    if (this.custom_y_min != null) {
        min = this.custom_y_min;
    }

    if (this.custom_y_max != null) {
        max = this.custom_y_max;
    }

    this.y_scale.domain([min, max]).nice();
};

BasePlot.prototype.setXDomain = function(min_x, max_x) {
    this.x_time_scale.domain([new Date(min_x * 1000), new Date(max_x * 1000)]).nice();
    var domain = this.x_time_scale.domain();
    this.x_scale.domain([domain[0].getTime() / 1000, domain[1].getTime() / 1000]);
};


BasePlot.prototype.setData = function(data) {
    this.stored_data = data;

    this.title.text(data.name());

    if(data.hasBoundaries()) {
        this.setYDomain(data.minY(), data.maxY());
        this.setXDomain(data.minX(), data.maxX());
    }

    this.render();
};

BasePlot.prototype.render = function() {
//    var now = Date.now();

    var data = this.stored_data;

    this.resetLegends();

    var context = this.canvas.node().getContext("2d");
    context.save();
    context.clearRect(0, 0, this.width, this.height);

    this.renderObservations(context, data);
    this.renderRefcase(context, data);
    this.render_callback(context, data);

    this.legend_group.selectAll(".plot-legend").data(this.legend_list).call(this.legend);

    this.plot_group.select(".y.axis").transition().duration(0).call(this.y_axis);
    this.plot_group.select(".x.axis").transition().duration(0).call(this.x_axis);

    context.restore();

//    console.log("Time: " + (Date.now() - now));
};

BasePlot.prototype.setRenderCallback = function(callback) {
    this.render_callback = callback;
};


BasePlot.prototype.resetLegends = function() {
    this.legend_list = [];
};

BasePlot.prototype.addLegend = function(style, name, render_function) {
    this.legend_list.push({"style": style, "name": name,"render_function": render_function});
};


BasePlot.prototype.renderObservations = function(context, data) {
    if(data.hasObservationData()) {
        if(data.observationIsContinuous()) {
            var x_values = data.observationXValues();
            var y_values = data.observationYValues();
            var std_values = data.observationStdValues();

            var obs_x_area_samples = [];
            var obs_y_area_samples = [];

            for (var index = 0; index < x_values.length; index++) {
                obs_x_area_samples.push(x_values[index]);
                obs_y_area_samples.push(y_values[index] + std_values[index]);
            }

            for (var index = x_values.length - 1; index >= 0; index--) {
                obs_x_area_samples.push(x_values[index]);
                obs_y_area_samples.push(y_values[index] - std_values[index]);
            }

            this.area_renderer.style(STYLES["observation_area"]);
            this.area_renderer(context, obs_x_area_samples, obs_y_area_samples);


            this.line_renderer.style(STYLES["observation"]);
            this.line_renderer(context, x_values, y_values);


            this.circle_renderer.style(STYLES["observation"]);

            var circle_count = this.width / 20;
            var step = y_values.length / circle_count;
            for(var index = 0; index < y_values.length; index += step) {
                var idx = Math.min(y_values.length, Math.round(index));
                var x = x_values[idx];
                var y = y_values[idx];
                this.circle_renderer(context, x, y);
            }

            this.circle_renderer(context, x_values[y_values.length - 1], y_values[y_values.length - 1]);

            this.addLegend(STYLES["observation"], "Observation", CanvasPlotLegend.circledLine);
            this.addLegend(STYLES["observation_area"], "Observation", CanvasPlotLegend.filledCircle);
        } else {

            var obs_x_samples = data.observationXValues();
            var obs_y_samples = data.observationYValues();
            var obs_std_samples = data.observationStdValues();

            for (var index = 0; index < obs_x_samples.length; index++) {
                var x = obs_x_samples[index];
                var y = obs_y_samples[index];
                var error = obs_std_samples[index];

                this.error_bar_renderer.style(STYLES["observation_error_bar"]);
                this.error_bar_renderer(context, x, y, error);
            }
            this.addLegend(STYLES["observation_error_bar"], "Observation error bar", CanvasPlotLegend.errorBar);
        }
    }
};


BasePlot.prototype.renderRefcase = function(context, data) {
    if(data.hasRefcaseData()) {
        var style = STYLES["refcase"];

        this.line_renderer.style(style);
        this.line_renderer(context, data.refcaseXValues(), data.refcaseYValues());

        this.addLegend(style, "Refcase", CanvasPlotLegend.simpleLine);
    }
};

BasePlot.prototype.createLineRenderer = function() {
    return CanvasPlotLine().x(this.x).y(this.y);
};



