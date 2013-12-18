// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'histogram.js' is part of ERT - Ensemble based Reservoir Tool.
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


function Histogram(element) {
    var stored_data = [];
    var stored_case_name = "";

    var margin = {left: 30, right: 30, top: 20, bottom: 30};

    var style = STYLES["default"];

    var count_format = d3.format("d");
    var value_format = d3.format("0.02sdfs");

    var group = element.append("div")
        .attr("class", "histogram");


    var title = group.append("div")
        .attr("class", "histogram-title")
        .text("Histogram");

    var histogram_area = group.append("div").attr("class", "plot-area");

    var legend_group = group.append("div")
        .attr("class", "plot-legend-group");

    var histogram_group = histogram_area.append("svg")
        .attr("class", "plot-svg");

    var width = 384 - margin.left - margin.right;
    var height = 256 - margin.top - margin.bottom;

    var svg = histogram_group.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .style("width", width + "px")
        .style("height", height + "px");

    var canvas = histogram_area.append("canvas")
        .attr("id", "histogram-canvas")
        .attr("width", width)
        .attr("height", height)
        .style("position", "absolute")
        .style("top", (margin.top) + "px")
        .style("left", (margin.left) + "px")
        .style("z-index", 5);


    var x_scale = d3.scale.linear().range([0, width]).nice();
    var y_scale = d3.scale.linear().range([height, 0]).nice();

    var y_axis = d3.svg.axis()
        .scale(y_scale)
        .tickFormat(count_format)
        .ticks(5)
        .orient("left")
        .tickSize(-width, -width);

    var x_axis = d3.svg.axis()
        .scale(x_scale)
        .tickFormat(value_format)
        .orient("bottom");

    histogram_group.append("g")
        .attr("class", "y axis pale")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(y_axis);

    histogram_group.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(" + margin.left + ", " + (height + margin.top) + ")")
        .call(x_axis);

    var histogram_renderer = HistogramRenderer().x(x_scale).y(y_scale).margin(0, 1, 0, 1);
    var line_renderer = CanvasPlotLine().x(x_scale).y(y_scale);
    var area_renderer = CanvasPlotArea().x(x_scale).y(y_scale);
    var circle_renderer = CanvasCircle().x(x_scale).y(y_scale);

    var legend = CanvasPlotLegend();
    var legend_list = [];

    var setYDomain = function(min_y, max_y) {
        y_scale.domain([min_y, max_y]).nice();

        if(y_scale.domain()[1] == max_y) {
            y_scale.domain([min_y, max_y + 1]).nice();
        }
    };

    var setXDomain = function(min_x, max_x) {
        var min = min_x;
        var max = max_x;

//        if (this.custom_y_min != null) {
//            min = this.custom_y_min;
//        }
//
//        if (this.custom_y_max != null) {
//            max = this.custom_y_max;
//        }

        x_scale.domain([min, max]).nice();
    };

    var resetLegends = function() {
        legend_list = [];
    };

    var addLegend = function(style, name, render_function) {
        legend_list.push({"style": style, "name": name,"render_function": render_function});
    };

    function histogram(data, case_name) {
        stored_data = data;
        stored_case_name = case_name;

        resetLegends();

        title.text(data.name());

        var context = canvas.node().getContext("2d");
        context.save();
        context.clearRect(0, 0, width, height);


        if(data.hasObservation()) {
            line_renderer.style(STYLES["observation"]);
            var obs = data.observation();
            var top = data.maxCount() + 1;
            line_renderer(context, [obs, obs], [0, top]);
            addLegend(STYLES["observation"], "Observation", CanvasPlotLegend.circledLine);

            var error = data.observationError();
            area_renderer.style(STYLES["observation_area"]);
            area_renderer(context, [obs - error, obs + error, obs + error, obs - error], [top, top, 0, 0]);

            var circle_count = 5;
            var step = (top) / (circle_count - 1);
            for(var index = 0; index < circle_count; index++) {
                circle_renderer(context, obs, step * index);
            }


            addLegend(STYLES["observation_area"], "Observation error", CanvasPlotLegend.filledCircle);


        }

        if(data.hasRefcase()) {
            line_renderer.style(STYLES["refcase"]);
            line_renderer(context, [data.refcase(), data.refcase()], [0, data.maxCount() + 1]);
            addLegend(STYLES["refcase"], "Refcase", CanvasPlotLegend.simpleLine);
        }

        if(data.hasCaseHistogram(case_name)) {

            var case_histogram = data.caseHistogram(case_name);

            var bin_count = data.numberOfBins();
            var min_x = data.min();
            var max_x = data.max();

            var max_count = data.maxCount();

            var bins = d3.layout.histogram()
                .bins(bin_count)
                .range([min_x, max_x])
                .bins(bin_count)(case_histogram.samples());

            setYDomain(0, max_count);
            setXDomain(min_x, max_x);

            histogram_renderer.style(style);
            histogram_renderer(context, bins);

            addLegend(style, case_name, CanvasPlotLegend.filledCircle);

            legend_group.selectAll(".plot-legend").data(legend_list).call(legend);
            histogram_group.select(".y.axis").transition().duration(0).call(y_axis);
            histogram_group.select(".x.axis").transition().duration(0).call(x_axis);
        }

        context.restore();
    }

    histogram.resize = function(w, h) {
        w = w - 80;
        h = h - 70;

        width = w - margin.left - margin.right;
        height = h - margin.top - margin.bottom;

        x_scale.range([0, width]);
        y_scale.range([height, 0]).nice();

        y_axis.tickSize(-width, -width);

        histogram_group.style("width", w + "px");
        histogram_group.style("height", h + "px");

        canvas.attr("width", width).attr("height", height);

        svg.style("width", width + "px");
        svg.style("height", height + "px");

        histogram_group.select(".x.axis")
            .attr("transform", "translate(" + margin.left + ", " + (height + margin.top) + ")");

        histogram(stored_data, stored_case_name);
    };

    histogram.style = function (value) {
        if (!arguments.length) return style;
        style = value;
        return histogram;
    };

    return histogram;
}