/*
 * $Id$
 */

#ifndef __PLOT_CONST_HXX__
#define __PLOT_CONST_HXX__

#include "plot_const.h"
#include <map>
#include <vector>
#include <string>

extern std::map<std::string,plot_line_style_type> plot_line_style ;
extern std::vector<std::string> plot_line_styles ;

extern std::map<std::string,plot_style_type> plot_style ;
extern std::vector<std::string> plot_styles ;

extern std::map<std::string,plot_color_type> plot_color ;
extern std::vector<std::string> plot_colors ;
  

#endif
