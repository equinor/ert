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
    var margin = {left: 90, right: 20, top: 20, bottom: 30};
    var duration = 250;
    var bins = 20;
    var histogram_style = "";

    var group = element.append("div")
        .attr("class", "plot");


    var title = group.append("div")
        .attr("class", "histogram-title")
        .text("Histogram");

    var histogram_group = group.append("svg")
        .attr("class", "histogram-svg");

    var width = 384 - margin.left - margin.right;
    var height = 256 - margin.top - margin.bottom;

    var svg = histogram_group
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .style("width", width + "px")
        .style("height", height + "px");


    var x_scale = d3.time.scale().range([0, width]);
    var y_scale = d3.scale.linear().range([height, 0]).nice();

    var y_axis = d3.svg.axis()
        .scale(y_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("left")
        .tickSize(-width, -width);

    var x_axis = d3.svg.axis()
        .scale(x_scale)
        .ticks(10)
        .tickPadding(10)
        .orient("bottom")
        .tickSubdivide(false);

    histogram_group.append("g")
        .attr("class", "y axis pale")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(y_axis);

    histogram_group.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(" + margin.left + ", " + (height + margin.top) + ")")
        .call(x_axis);


    function histogram(selection) {
//        selection.enter().append("g").attr("class", "histogram");
//
//        selection.exit().transition().duration(duration).style("opacity", 0).remove();
//
//        selection.selectAll(".histogram").data(function(d) {return [d];});
//
//
//        var area = d3.svg.area().x(x).y0(y_min).y1(y_max).interpolate("basis");
//
//        selection.transition()
//            .duration(duration)
//            .attr("d", area);
//
//        selection.enter()
//            .append("path")
//            .attr("class", histogram_style)
//            .attr("d", area)
//            .style("opacity", 0.0)
//            .transition()
//            .duration(duration)
//            .style("opacity", 1.0);
//
//        selection.exit()
//            .transition()
//            .duration(duration)
//            .style("opacity", 0.0)
//            .remove();

        histogram_group.select(".y.axis").transition().duration(duration).call(y_axis);
        histogram_group.select(".x.axis").transition().duration(duration).call(x_axis);

    }

    histogram.bins = function(value) {
        if (!arguments.length) return bins;
        bins = value;
        return histogram;
    };

    histogram.duration = function(value) {
        if (!arguments.length) return duration;
        duration = value;
        return histogram;
    };


    histogram.style = function(value) {
        if (!arguments.length) return histogram_style;
        histogram_style = value;
        return histogram;
    };

    histogram.resize = function(w, h) {
        w = w - 80;
        h = h - 70;

        width = w - margin.left - margin.right;
        height = h - margin.top - margin.bottom;

        x_scale.range([0, width]);
        y_scale.range([height, 0]).nice();

        y_axis.tickSize(-10, -10);

        histogram_group.style("width", w + "px");
        histogram_group.style("height", h + "px");

        svg.style("width", width + "px");
        svg.style("height", height + "px");

        histogram_group.select(".x.axis")
            .attr("transform", "translate(" + margin.left + ", " + (height + margin.top) + ")");

        histogram(stored_data);
    };

    return histogram;

}