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


function Plot(element, data) {
    this.stored_data = [];
    this.margin = {left: 90, right: 20, top: 20, bottom: 30};
    this.root_elemenet = element;

    this.custom_y_min = null;
    this.custom_y_max = null;


    var group = this.root_elemenet.append("div")
        .attr("class", "plot");

    this.title = group.append("div")
        .attr("class", "plot-title")
        .text(data.name);

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

    this.line_renderer = CanvasPlotLine().x(this.x).y(this.y);
    this.area_renderer = CanvasPlotArea().x(this.x).y(this.y);
    this.error_bar_renderer = CanvasErrorBar().x(this.x).y(this.y);
    this.circle_renderer = CanvasCircle().x(this.x).y(this.y);
}

Plot.prototype.resize = function(width, height) {
    //Some magic margins...
    width = width - 80;
    height = height - 70;

    this.width = width - this.margin.left - this.margin.right;
    this.height = height - this.margin.top - this.margin.bottom;

    this.x_scale.range([0, this.width]);
    this.x_time_scale.range([0, this.width]);
    this.y_scale.range([this.height, 0]).nice();

    this.y_axis.tickSize(-this.width, -this.width);

    this.plot_group.style("width", width + "px");
    this.plot_group.style("height", height + "px");

    this.canvas
        .attr("width", this.width)
        .attr("height", this.height);

    this.plot_group.select(".x.axis")
        .attr("transform", "translate(" + this.margin.left + ", " + (this.height + this.margin.top) + ")");

    this.setData(this.stored_data);
};

Plot.prototype.setYScales = function(min, max) {
    this.custom_y_min = min;
    this.custom_y_max = max;

    this.setData(this.stored_data);
};


Plot.prototype.setYDomain = function(min_y, max_y) {
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

Plot.prototype.setXDomain = function(min_x, max_x) {
    this.x_time_scale.domain([new Date(min_x * 1000), new Date(max_x * 1000)]).nice();
    var domain = this.x_time_scale.domain();
    this.x_scale.domain([domain[0].getTime() / 1000, domain[1].getTime() / 1000]);
};


Plot.prototype.setData = function(data) {
    this.stored_data = data;

    this.title.text(data["name"]);

    this.setYDomain(data["min_y"], data["max_y"]);
    this.setXDomain(data["min_x"], data["max_x"]);




    var legends = [];

    var x_samples = [];
    var y_samples = [];
    for(var ensemble_index in data["ensemble_names"]) {
        var case_samples = []
        y_samples[ensemble_index] = case_samples;

        for (var index = 0; index < data["ensemble"][ensemble_index].length; index++) {
            var ensemble_samples = data["ensemble"][ensemble_index][index]["samples"];
            case_samples[index] = [];
            for(var sample_index = 0; sample_index < ensemble_samples.length; sample_index++) {
                var sample = ensemble_samples[sample_index];
                x_samples[sample_index] = sample["x"];
                case_samples[index][sample_index] = sample["y"];

            }
        }
    }


//    var now = Date.now();



    var context = this.canvas.node().getContext("2d");
    context.clearRect(0, 0, this.width, this.height);




    if(data["observations"] != null) {
        if(data["observations"]["continuous_line"]) {
            var obs_x_area_samples = [];
            var obs_y_area_samples = [];
            var obs_y_samples = [];
            var observation_samples = data["observations"]["samples"];

            for (var index = 0; index < observation_samples.length; index++) {
                var sample = observation_samples[index];
                obs_x_area_samples.push(x_samples[index]);
                obs_y_area_samples.push(sample["y"] + sample["std"]);
                obs_y_samples.push(sample["y"]);

            }

            for (var index = observation_samples.length - 1; index >= 0; index--) {
                var sample = observation_samples[index];
                obs_x_area_samples.push(x_samples[index]);
                obs_y_area_samples.push(sample["y"] - sample["std"]);
            }

            this.area_renderer.style(STYLES["observation_area"]);
            this.area_renderer(context, obs_x_area_samples, obs_y_area_samples);


            this.line_renderer.style(STYLES["observation"]);
            this.line_renderer(context, x_samples, obs_y_samples);

            this.circle_renderer.style(STYLES["observation"]);

            var circle_count = this.width / 20;
            var step = obs_y_samples.length / circle_count;
            for(var index = 0; index < obs_y_samples.length; index += step) {
                var idx = Math.min(obs_y_samples.length, Math.round(index));
                var x = x_samples[idx];
                var y = obs_y_samples[idx];
                this.circle_renderer(context, x, y);
            }

            this.circle_renderer(context, x_samples[obs_y_samples.length - 1], obs_y_samples[obs_y_samples.length - 1]);

            legends.push({"style": STYLES["observation"], "name": "Observation", "render_function": CanvasPlotLegend.circledLine});
            legends.push({"style": STYLES["observation_area"], "name": "Observation error", "render_function": CanvasPlotLegend.filledCircle});
        } else {

            var observation_samples = data["observations"]["samples"];

            for (var index = 0; index < observation_samples.length; index++) {
                var sample = observation_samples[index];
                var x = sample["x"];
                var y = sample["y"];
                var error = sample["std"];

                this.error_bar_renderer.style(STYLES["observation_error_bar"]);
                this.error_bar_renderer(context, x, y, error);
            }

            legends.push({"style": STYLES["observation_error_bar"], "name": "Observation error bar", "render_function": CanvasPlotLegend.errorBar});
        }

    }



    if(data["refcase"] != null) {
        var refcase_samples = data["refcase"]["samples"];
        var style = STYLES["refcase"];

        var refcase_y_samples = [];
        for(var index in refcase_samples) {
            refcase_y_samples[index] = refcase_samples[index]["y"];
        }

        this.line_renderer.style(style);
        this.line_renderer(context, x_samples, refcase_y_samples);

        legends.push({"style": style, "name": "Refcase", "render_function": CanvasPlotLegend.simpleLine});
    }


    for(var ensemble_index in data["ensemble_names"]) {
        var style_name = "ensemble_" + (parseInt(ensemble_index) + 1);
        var style = STYLES[style_name];

        this.line_renderer.style(style);

        for (var index = 0; index < y_samples[ensemble_index].length; index++) {
            this.line_renderer(context, x_samples, y_samples[ensemble_index][index]);
        }

        legends.push({"style": style, "name": data["ensemble_names"][ensemble_index], "render_function": CanvasPlotLegend.simpleLine});
    }

//    console.log("Time: " + (Date.now() - now));

//    console.log(JSON.stringify(legends, null, 4));
    this.legend_group.selectAll(".plot-legend").data(legends).call(this.legend);

    this.plot_group.select(".y.axis").transition().duration(0).call(this.y_axis);
    this.plot_group.select(".x.axis").transition().duration(0).call(this.x_axis);
};