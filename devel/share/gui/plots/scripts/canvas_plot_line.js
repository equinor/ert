// Copyright (C) 2013 Statoil ASA, Norway.
//
// The file 'canvas_plot_line.js' is part of ERT - Ensemble based Reservoir Tool.
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


var CP = window.CanvasRenderingContext2D && CanvasRenderingContext2D.prototype;
if (CP && CP.lineTo) {
    CP.dashedLine = function (x, y, x2, y2, dash_array) {
        if (!dash_array) dash_array = [10, 5];

        this.save();

        var dx = (x2 - x), dy = (y2 - y);
        var length = Math.sqrt(dx * dx + dy * dy);
        var rotation = Math.atan2(dy, dx);

        this.translate(x, y);
        this.moveTo(0, 0);
        this.rotate(rotation);

        var dash_count = dash_array.length;
        var dash_index = 0, draw = true;
        x = 0;
        while (length > x) {
            x += dash_array[dash_index++ % dash_count];
            if (x > length) x = length;
            draw ? this.lineTo(x, 0) : this.moveTo(x, 0);
            draw = !draw;
        }
        this.restore();

    }
}


function CanvasPlotLine() {
    var X = function (d) { return d; };
    var Y = function (d) { return d; };
    var style = STYLES["default"];

    function render(context, x_samples, y_samples) {
        context.lineWidth = style["stroke_width"];
        context.strokeStyle = style["stroke"];
        context.lineCap = style["line_cap"];

        var dashed = false;
        if (style["dash_array"].length > 0) {
            dashed = true;
        }

        if (dashed) {
            var prev_x = null;
            var prev_y = null;
        }

        context.beginPath();
        for (var index in x_samples) {
            var x_sample = x_samples[index];
            var y_sample = y_samples[index];
            var x = X(x_sample);
            var y = Y(y_sample);

            if (index == 0) {
                context.moveTo(x, y);
                prev_x = x;
                prev_y = y;
            } else {
                if (dashed) {
                    context.dashedLine(prev_x, prev_y, x, y, style["dash_array"]);
                    prev_x = x;
                    prev_y = y;
                } else {
                    context.lineTo(x, y);
                }
            }
        }


        context.stroke();

    }

    render.x = function (value) {
        if (!arguments.length) return X;
        X = value;
        return render;
    };

    render.y = function (value) {
        if (!arguments.length) return Y;
        Y = value;
        return render;
    };

    render.style = function (value) {
        if (!arguments.length) return style;
        style = value;
        return render;
    };

    return render;
}