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
    this.data = null;

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

    this.svg = this.svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    this.x_scale = d3.time.scale().range([0, this.width]);
    this.y_scale = d3.scale.linear().range([this.height, 0]).nice();

    this.y_axis = d3.svg.axis()
        .scale(this.y_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("left")
        .tickSize(-this.width, -this.width);
//        .tickSubdivide(1);
//        .tickValues([-30, -20, -10, 0, 10, 20, 30, 40]);



    this.x_axis = d3.svg.axis()
        .scale(this.x_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("bottom")
        .tickSubdivide(true);
//        .tickSubdivide(12)
//        .tickFormat(d3.time.format("%b %Y"));


    this.svg.append("g")
        .attr("class", "y axis pale")
        .attr("transform", "translate(0, 0)")
        .call(this.y_axis);

    this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0, " + (this.height) + ")")
        .call(this.x_axis);


    this.area_plot = this.svg.append("g");

    this.area_plot.append("path")
        .attr("class", "area area-fill");

    this.top_function = function(d) {
        return d["value"] + d["std"];
    };

    this.bottom_function = function(d) {
        return self.adjustY(d["value"], d["std"]);
    };

    this.area_function = d3.svg.area()
        .x(function (d) {
//            console.log(d + " " + d["report_step_date"] + " " + self.x_scale(d["report_step_date"]));
            return self.x_scale(new Date(d["x"] * 1000));
        })
        .y0(function (d) {
            return  self.y_scale(self.bottom_function(d));
        })
        .y1(function (d) {
            return self.y_scale(self.top_function(d));
        })
        .interpolate("basis");


    this.plot = this.svg.append("g");
    this.plot.append("path")
        .attr("class", "plot-line");

    var self = this;
    this.line = d3.svg.line()
        .x(function (d, i) {
            return self.x_scale(new Date(d["x"] * 1000));
        })
        .y(function (d, i) {
            return self.y_scale(d["value"]);
        })
        .interpolate("basis");



}

Plot.prototype.adjustY = function(y, std) {
    if(y >= 0) {
        y = Math.max(0, y - std);
    }
    return y;
};

Plot.prototype.setData = function(data) {
    var self = this;

    this.title.text(data["group"]);

    var stat = data["statistics"];

    var std = stat["max_std"];
    var min = stat["min_value"];

    min = this.adjustY(min, std);
    var max = stat["max_value"] + std;
    this.y_scale.domain([min, max]).nice();
    this.x_scale.domain([new Date(data["min_x"] * 1000), new Date(data["max_x"] * 1000)]).nice();

//    var delay = 0;
//    if(this.data != null) {
//
//        var flat_line = d3.svg.line()
//            .x(function (d) {
//                return self.x_scale(d["report_step_date"]);
//            })
//            .y(function (d) {
//                return self.y_scale(min);
//            })
//            .interpolate("basis");
//
//        this.plot.datum(this.data["data"]);
//        this.plot.select(".plot-line")
//            .transition()
//            .duration(500)
//            .attr("d", flat_line);
//
//        delay = 500;
//    }



    this.plot.datum(data["samples"]);


    this.plot.select(".plot-line")
        .transition()
        .delay(0)
        .duration(500)
        .attr("d", this.line);


    this.area_plot.datum(data["samples"]);
    this.area_plot.select(".area")
        .transition()
        .duration(500)
        .attr("d", this.area_function);


    this.svg.select(".y.axis").transition().duration(500).call(this.y_axis);
    this.svg.select(".x.axis").transition().duration(500).call(this.x_axis);

    this.data = data;


};