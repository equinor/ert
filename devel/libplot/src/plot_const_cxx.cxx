#include "plot_const_cxx.hxx"

#include <boost/assign/list_of.hpp> // for 'map_list_of()'

using namespace boost::assign ; 
using namespace std ;

std::map<std::string,plot_line_style_type> plot_line_style =
  map_list_of
  (string("solid_line"),solid_line) 
  (string("line"),      solid_line) 
  (string("short_dash"),short_dash) 
  (string("shortdash"), short_dash) 
  (string("dash"),      short_dash) 
  (string("long_dash"), long_dash)
  (string("longdash"),  long_dash)
  ;

std::vector<std::string> plot_line_styles =
  list_of
  (string("solid_line")) 
  (string("short_dash")) 
  (string("long_dash"))
  ;


std::map<std::string,plot_style_type> plot_style =
  map_list_of
  (string("BLANK"),BLANK) 
  (string("blank"),BLANK) 
  (string("LINE"),LINE) 
  (string("line"),LINE) 
  (string("POINTS"),POINTS) 
  (string("points"),POINTS) 
  (string("LINE_POINTS"),LINE_POINTS) 
  (string("line_points"),LINE_POINTS) 
  ;

std::vector<std::string> plot_styles =
  list_of
  (string("blank")) 
  (string("line")) 
  (string("points")) 
  (string("line_points")) 
  ;

std::map<std::string,plot_color_type> plot_color =
  map_list_of
  (string("WHITE"),WHITE) 
  (string("RED"),RED) 
  (string("YELLOW"),YELLOW) 
  (string("GREEN"),GREEN) 
  (string("AQUAMARINE"),AQUAMARINE) 
  (string("PINK"),PINK) 
  (string("WHEAT"),WHEAT) 
  (string("GRAY"),GRAY) 
  (string("BROWN"),BROWN) 
  (string("BLUE"),BLUE) 
  (string("VIOLET"),VIOLET) 
  (string("CYAN"),CYAN) 
  (string("TURQUOISE"),TURQUOISE) 
  (string("MAGENTA"),MAGENTA) 
  (string("SALMON"),SALMON) 
  (string("BLACK"),BLACK) 
  (string("white"),WHITE) 
  (string("red"),RED) 
  (string("yellow"),YELLOW) 
  (string("green"),GREEN) 
  (string("aquamarine"),AQUAMARINE) 
  (string("pink"),PINK) 
  (string("wheat"),WHEAT) 
  (string("gray"),GRAY) 
  (string("brown"),BROWN) 
  (string("blue"),BLUE) 
  (string("violet"),VIOLET) 
  (string("cyan"),CYAN) 
  (string("turquiose"),TURQUOISE) 
  (string("magenta"),MAGENTA) 
  (string("salmon"),SALMON) 
  (string("black"),BLACK) 
  ;


std::vector<std::string> plot_colors =
  list_of
  (string("white")) 
  (string("red")) 
  (string("yellow")) 
  (string("green"),GREEN) 
  (string("aquamarine")) 
  (string("pink")) 
  (string("wheat")) 
  (string("gray")) 
  (string("brown")) 
  (string("blue")) 
  (string("violet")) 
  (string("cyan")) 
  (string("turquiose")) 
  (string("magenta")) 
  (string("salmon")) 
  (string("black")) 
  ;

