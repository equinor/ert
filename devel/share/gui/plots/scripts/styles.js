var STYLES = {
    default: {
        stroke: "rgba(0, 0, 0, 1.0)",
        fill: "rgba(200, 200, 200, 1.0)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    observation: {
        stroke: "rgba(0, 0, 0, 1.0)",
        fill: "rgba(0, 0, 0, 0.0)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    observation_error_bar: {
        stroke: "rgba(0, 0, 0, 1.0)",
        fill: "rgba(0, 0, 0, 0.0)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    observation_area: {
        stroke: "rgba(0, 0, 0, 0.15)",
        fill: "rgba(0, 0, 0, 0.2)",
        stroke_width: 2,
        dash_array: [],
        line_cap: "butt"
    },
    refcase: {
        stroke: "rgba(0, 0, 0, 0.7)",
        fill: "rgba(0, 0, 0, 0.0)",
        stroke_width: 2,
        dash_array: [],
        line_cap: "butt"
    },
    ensemble_1: {
        stroke: "rgba(56, 108, 176, 0.8)",
        fill: "rgba(56, 108, 176, 0.5)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    ensemble_2: {
        stroke: "rgba(127, 201, 127, 0.8)",
        fill: "rgba(127, 201, 127, 0.5)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    ensemble_3: {
        stroke: "rgba(253, 192, 134, 0.8)",
        fill: "rgba(253, 192, 134, 0.5)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    ensemble_4: {
        stroke: "rgba(240, 2, 127, 0.8)",
        fill: "rgba(240, 2, 127, 0.5)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },
    ensemble_5: {
        stroke: "rgba(191, 91, 23, 0.8)",
        fill: "rgba(191, 91, 23, 0.5)",
        stroke_width: 1,
        dash_array: [],
        line_cap: "butt"
    },

    ensemble_colors: ["ensemble_1", "ensemble_2", "ensemble_3", "ensemble_4", "ensemble_5"]

};


STYLES.parseColor = function(input) {
    var result = [255, 255, 255, 1];

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


STYLES.asRgb = function(r, g, b) {
    return "rgb(" + r + "," + g + "," + b + ")";
};

STYLES.asRgba = function(r, g, b, a) {
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
};


STYLES.componentToHex = function(c) {
    var hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
};

STYLES.rgbToHex = function(r, g, b) {
    return "#" + CanvasPlotLegend.componentToHex(r) + CanvasPlotLegend.componentToHex(g) + CanvasPlotLegend.componentToHex(b);
};


STYLES.blendWithWhite = function(color, result_alpha) {
    var rgba = STYLES.parseColor(color);

    var a = rgba[3];
    var ab = (1 - rgba[3]) * 255;
    var r = parseInt(rgba[0] * a + ab);
    var g = parseInt(rgba[1] * a + ab);
    var b = parseInt(rgba[2] * a + ab);

    return STYLES.asRgba(r, g, b, result_alpha);
};


var alpha = 0.7;
STYLES["ensemble_1"]["fill"] = STYLES.blendWithWhite(STYLES["ensemble_1"]["fill"], alpha);
STYLES["ensemble_2"]["fill"] = STYLES.blendWithWhite(STYLES["ensemble_2"]["fill"], alpha);
STYLES["ensemble_3"]["fill"] = STYLES.blendWithWhite(STYLES["ensemble_3"]["fill"], alpha);
STYLES["ensemble_4"]["fill"] = STYLES.blendWithWhite(STYLES["ensemble_4"]["fill"], alpha);
STYLES["ensemble_5"]["fill"] = STYLES.blendWithWhite(STYLES["ensemble_5"]["fill"], alpha);