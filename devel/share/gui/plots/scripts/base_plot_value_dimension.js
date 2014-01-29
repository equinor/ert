// Copyright (C) 2014 Statoil ASA, Norway.
//
// The file 'base_plot_value_dimension.js' is part of ERT - Ensemble based Reservoir Tool.
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

function BasePlotValueDimension(flip_range){
    if (!arguments.length) {
        var flip_range = false;
    }
    var scale = d3.scale.linear().range([1, 0]).domain([0, 1]).nice();
    var value_format = d3.format(".4s");

    var scaler = function(d) {
        return scale(d);
    };

    function dimension(value) {
        return scaler(value);
    }

    dimension.setDomain = function(min, max) {

        if(min == max) {
            min = min - 0.1 * min
            max = max + 0.1 * max
        }

        if(flip_range){
            var tmp = min;
            min = max;
            max = tmp;
        }
        scale.domain([min, max]).nice();
    };

    dimension.setRange = function(min, max) {
        scale.range([min, max]).nice();
    };

    dimension.scale = function(new_scale) {
        if (!arguments.length) return scale;
        scale = new_scale;
        return dimension;
    };

    dimension.scaler = function(new_scaler) {
        if (!arguments.length) return scaler;
        scaler = new_scaler;
        return dimension;
    };

    dimension.format = function(axis, max_length){
        axis.ticks(10)
            .tickPadding(10)
            .tickSize(-max_length, -max_length)
            .tickFormat(value_format);

        return dimension;
    };

    return dimension;
}