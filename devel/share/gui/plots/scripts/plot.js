// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'plot.js' is part of ERT - Ensemble based Reservoir Tool.
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


function Plot(element, data) {
    var margin = {left: 90, right: 20, top: 20, bottom: 30};
    this.root_elemenet = element;

    var group = this.root_elemenet.append("div")
        .attr("class", "plot");


    this.title = group.append("div")
        .attr("class", "plot-title")
        .text(data.name);

    var plot_group = group.append("div")
        .attr("class", "plot-svg");

    this.svg = plot_group.append("svg")
        .attr("width", plot_group.style("width"))
        .attr("height", plot_group.style("height"));

    this.width = this.svg.attr("width").replace("px", "") - margin.left - margin.right;
    this.height = this.svg.attr("height").replace("px", "") - margin.top - margin.bottom;

    this.svg = this.svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    this.x_scale = d3.time.scale().range([0, this.width]);
    this.y_scale = d3.scale.linear().range([this.height, 0]).nice();

    this.y_axis = d3.svg.axis()
        .scale(this.y_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("left")
        .tickSize(-this.width, -this.width);

    this.x_axis = d3.svg.axis()
        .scale(this.x_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("bottom")
        .tickSubdivide(true);

    this.svg.append("g")
        .attr("class", "y axis pale")
        .attr("transform", "translate(0, 0)")
        .call(this.y_axis);

    this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0, " + (this.height) + ")")
        .call(this.x_axis);


    var self = this;
    this.plot = this.svg.append("g");

    this.adjustY = function(y, std) {
        if(y >= 0) {
            y = Math.max(0, y - std);
        } else {
            y -= std
        }
        return y;
    };

    var top_function = function(d) {
        return d["value"] + d["std"];
    };

    var bottom_function = function(d) {
        return self.adjustY(d["value"], d["std"]);
    };

    this.x = function (d) {
        return self.x_scale(new Date(d["x"] * 1000));
    };

    this.y = function (d) {
        return self.y_scale(d["value"]);
    };

    this.y_min = function (d) {
        return  self.y_scale(bottom_function(d));
    };

    this.y_max = function (d) {
        return self.y_scale(top_function(d));
    };

    this.duration = 250;

    this.std_plot = StdPlot()
        .radius(2.5)
        .x(this.x)
        .y(this.y)
        .y_max(this.y_max)
        .y_min(this.y_min)
        .style("observation-std-point")
        .duration(this.duration);

    this.observation_line = PlotLine()
        .x(this.x)
        .y(this.y)
        .style("observation-plot-line")
        .duration(this.duration);

    this.observation_std_area = PlotArea()
        .x(this.x)
        .y_min(this.y_min)
        .y_max(this.y_max)
        .style("observation-plot-area")
        .duration(this.duration);

    this.refcase_line = PlotLine()
        .x(this.x)
        .y(this.y)
        .style("refcase-plot-line")
        .duration(this.duration);

}

Plot.prototype.setData = function(data) {

//    var data = data["observations"];

    this.title.text(data["name"]);

    var min = data["min_value"];
    var max = data["max_value"];
    this.y_scale.domain([min, max]).nice();
    this.x_scale.domain([new Date(data["min_x"] * 1000), new Date(data["max_x"] * 1000)]).nice();


    var observation_std_points;
    var observation_line;
    var observation_std_area;


    if(data["observations"] != null) {

        var observation_samples = data["observations"]["samples"];

        if(data["observations"]["continuous_line"]) {
            observation_line = this.plot.selectAll(".observation-plot-line").data([observation_samples]);
            observation_std_area = this.plot.selectAll(".observation-plot-area").data([observation_samples]);
            observation_std_points = this.plot.selectAll(".observation-std-point").data([]);
        } else {
            observation_line = this.plot.selectAll(".observation-plot-line").data([]);
            observation_std_area = this.plot.selectAll(".observation-plot-area").data([]);
            observation_std_points = this.plot.selectAll(".observation-std-point").data(observation_samples);
        }

        observation_line.call(this.observation_line);
        observation_std_area.call(this.observation_std_area);
        observation_std_points.call(this.std_plot);
    } else {
        observation_line = this.plot.selectAll(".observation-plot-line").data([]);
        observation_std_area = this.plot.selectAll(".observation-plot-area").data([]);
        observation_std_points = this.plot.selectAll(".observation-std-point").data([]);

        observation_line.call(this.observation_line);
        observation_std_area.call(this.observation_std_area);
        observation_std_points.call(this.std_plot);
    }


    var refcase_line;

    if(data["refcase"] != null) {
        var refcase_samples = data["refcase"]["samples"];
        refcase_line = this.plot.selectAll(".refcase-plot-line").data([refcase_samples]);
        refcase_line.call(this.refcase_line);

    } else {
        refcase_line = this.plot.selectAll(".refcase-plot-line").data([]);
        refcase_line.call(this.refcase_line);
    }


    this.svg.select(".y.axis").transition().duration(this.duration).call(this.y_axis);
    this.svg.select(".x.axis").transition().duration(this.duration).call(this.x_axis);

};