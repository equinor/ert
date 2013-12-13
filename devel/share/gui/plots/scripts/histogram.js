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
    var bin_count = 30;
    var histogram_style = "";

    var count_format = d3.format("d");
    var value_format = d3.format("s");

    var group = element.append("div")
        .attr("class", "plot");


    var title = group.append("div")
        .attr("class", "histogram-title")
        .text("Histogram");

    var histogram_area = group.append("div").attr("class", "plot-area");

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

    var rect_renderer = CanvasRect().x(x_scale).y(y_scale).margin(0, 1, 0, 1);


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


    function histogram(data) {
        stored_data = data;
        title.text(data.name());

        if(data.hasBoundaries()) {
            setXDomain(data.minY(), data.maxY());
        }

        if(data.hasEnsembleData()) {
            var context = canvas.node().getContext("2d");
            context.save();
            context.clearRect(0, 0, width, height);

            var case_list = data.caseList();

            for(var case_index = 0; case_index < case_list.length; case_index++)  {
                var case_name = case_list[case_index];
                var samples = data.reportStepSamples(case_name);
                var style = STYLES[("ensemble_" + (case_index + 1))];

                var bins = d3.layout.histogram().bins(x_scale.ticks(bin_count))(samples);
                setYDomain(0, d3.max(bins, function(d) { return d.y; }));

                rect_renderer.style(style);
                for(var i = 0; i < bins.length; i++) {
                    var value = bins[i];
                    rect_renderer(context, value.x, 0, value.x + value.dx, value.y);
                }
            }

            context.restore();
        }


        histogram_group.select(".y.axis").transition().duration(0).call(y_axis);
        histogram_group.select(".x.axis").transition().duration(0).call(x_axis);
    }

    histogram.bins = function(value) {
        if (!arguments.length) return bin_count;
        bin_count = value;
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

        y_axis.tickSize(-width, -width);

        histogram_group.style("width", w + "px");
        histogram_group.style("height", h + "px");

        canvas.attr("width", width).attr("height", height);

        svg.style("width", width + "px");
        svg.style("height", height + "px");

        histogram_group.select(".x.axis")
            .attr("transform", "translate(" + margin.left + ", " + (height + margin.top) + ")");

        histogram(stored_data);
    };

    return histogram;

}