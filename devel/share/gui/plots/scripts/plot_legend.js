// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'plot_legend.js' is part of ERT - Ensemble based Reservoir Tool.
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


function PlotLegend() {

    var size = 12;

    function legend(selection) {
        var groups = selection.enter()
            .append("div")
            .attr("class", "plot-legend");

        groups.append("svg")
            .attr("width", size + "px")
            .attr("height", size + "px")
            .append("rect")
            .attr("width",  (size - 2) + "px")
            .attr("height", (size - 2) + "px")
            .attr("transform", "translate(1,1)")
            .style("fill", function(d) {
                var c = parseColor(d["style"]["fill"]);
                return asRgb(c[0], c[1], c[2]);
            })
            .style("stroke", function(d) {
                var c = parseColor(d["style"]["stroke"]);
                return asRgb(c[0], c[1], c[2]);
            })
            .style("stroke-opacity", function(d) {
                var c = parseColor(d["style"]["stroke"]);
                return c[3];
            })
            .style("fill-opacity", function(d) {
                var c = parseColor(d["style"]["fill"]);
                return c[3];
            })
            .style("stroke-width", function(d) {
                return d["style"]["stroke_width"] + "px";
            })
            .style("stroke-dasharray", function(d) {
                return d["style"]["dash_array"];
            })
            .attr("class", "legend-marker");


        selection.selectAll(".legend-marker")
            .data(function(d) { return [d];})
            .transition()
            .style("fill", function(d) {
                var c = parseColor(d["style"]["fill"]);
                return asRgb(c[0], c[1], c[2]);
            })
            .style("stroke", function(d) {
                var c = parseColor(d["style"]["stroke"]);
                return asRgb(c[0], c[1], c[2]);
            })
            .style("stroke-opacity", function(d) {
                var c = parseColor(d["style"]["stroke"]);
                return c[3];
            })
            .style("fill-opacity", function(d) {
                var c = parseColor(d["style"]["fill"]);
                return c[3];
            })
            .style("stroke-width", function(d) {
                return d["style"]["stroke_width"] + "px";
            })
            .style("stroke-dasharray", function(d) {
                return d["style"]["dash_array"];
            });

        groups.append("div")
            .attr("class", "plot-legend-label")
            .text(function(d) {
                return d["name"];
            });

        selection.selectAll(".plot-legend-label")
            .data(function(d) {
                return [d];
            })
            .transition()
            .text(function(d) {
                return d["name"];
            });

        selection.exit()
            .remove();
    }

    var parseColor = function(input) {
        var result = [255, 255, 255, 1];

//        match = input.match(/^#([0-9a-f]{3})$/i);
//        if(match) {
//            match = match[1];
//            // in three-character format, each value is multiplied by 0x11 to give an
//            // even scale from 0x00 to 0xff
//            result[0] = parseInt(match.charAt(0), 16) * 0x11;
//            result[1] = parseInt(match.charAt(1), 16) * 0x11;
//            result[2] = parseInt(match.charAt(2), 16) * 0x11;
//            return result;
//        }
//
//        console.log("2");
//        match = input.match(/^#([0-9a-f]{6})$/i);
//        if(match) {
//            match = match[1];
//            result[0] = parseInt(match.substr(0,2), 16);
//            result[1] = parseInt(match.substr(2,4), 16);
//            result[2] = parseInt(match.substr(4,6), 16);
//            return result;
//        }
//
//        match = input.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i);
//        if(match) {
//            result[0] = match[1];
//            result[1] = match[2];
//            result[2] = match[3];
//            return result;
//        }
//        if(input === undefined) {
//            console.log("UNDEFINED!");
//            return result;
//        }
        var match = input.match(/^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+\.?\d*)\s*\)$/i);
        if(match) {
            result[0] = match[1];
            result[1] = match[2];
            result[2] = match[3];
            result[3] = parseFloat(match[4]);
            return result;
        }

        return result;
    };

    var asRgb = function(r, g, b) {
        return "rgb(" + r + "," + g + "," + b + ")";
    };


    var componentToHex = function(c) {
        var hex = c.toString(16);
        return hex.length == 1 ? "0" + hex : hex;
    };

    var rgbToHex = function(r, g, b) {
        return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
    };

    return legend;

}