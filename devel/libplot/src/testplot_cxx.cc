#include "plot.hxx"
#include "plot_const_cxx.hxx"

#include <vector>
#include <cstdlib>
#include <iostream>

using namespace std;

int main()
{
  plot plot1("png","xx.png") ;

  plot1.set_title("A test plot") ;
  plot1.set_top_padding(0.1) ;
  plot1.set_bottom_padding(0.1) ;
  plot1.set_left_padding(0.1) ;
  plot1.set_right_padding(0.1) ;
  plot1.set_xlabel("X") ;
  plot1.set_ylabel("Y") ;

  int dataset = plot1.new_dataset(plot_xy) ;
  plot1.set_style(dataset,plot_style["points"]) ;
  plot1.set_symbol_type(dataset,1) ;
  plot1.set_symbol_size(dataset,5.0) ;
  plot1.set_point_color(dataset,plot_color["red"]) ;
  plot1.append_point_xy(dataset,0.5,0.3) ;
  plot1.append_point_xy(dataset,0.6,0.4) ;

  dataset = plot1.new_dataset(plot_xy) ;
  plot1.set_style(dataset,LINE) ;
  plot1.set_line_style(dataset,short_dash) ;
  plot1.set_line_width(dataset,2.0) ;
  plot1.set_line_color(dataset,BLUE) ;
  plot1.append_point_xy(dataset,0.6,0.42) ;
  plot1.append_point_xy(dataset,0.5,0.2) ;
  plot1.append_point_xy(dataset,0.4,0.3) ;

  vector<double> x ;
  vector<double> y ;
  
  for (int i=0 ; i<100 ; i++) {
    x.push_back(0.01*i) ;
    y.push_back(0.001*i*i) ;
  } ;
  
  dataset = plot1.new_dataset(plot_xy) ;
  plot1.set_style(dataset,LINE) ;
  plot1.set_line_style(dataset,solid_line) ;
  plot1.set_line_width(dataset,1.5) ;
  plot1.set_line_color(dataset,VIOLET) ;
  plot1.append_point_xy(dataset,0.0,1.0) ;
  plot1.append_vector_xy(dataset,x,y) ;
  plot1.append_point_xy(dataset,1.3,8.0) ;

  dataset = plot1.new_dataset(plot_xy1y2) ;
  plot1.set_line_color(dataset,GREEN) ;
  plot1.append_point_xy1y2(dataset,0.3,0.1,0.2) ;
  plot1.append_point_xy1y2(dataset,0.33,0.11,0.15) ;


  int numcol = plot_colors.size() ;
  for (int i=0 ; i<numcol ; i++) {
    dataset = plot1.new_dataset(plot_xy) ;
    plot1.set_style(dataset,plot_style["points"]) ;
    plot1.set_point_color(dataset,plot_color[plot_colors[i]]) ;
    plot1.append_point_xy(dataset,1.0 + 0.1*i, 2.0 + 0.1*i) ;    
  } ;

  plot_range range ;
  plot1.get_extrema(range) ;

  cout << "Range: "
       << range.get_xmin() << " , " 
       << range.get_xmax() << " , " 
       << range.get_ymin() << " , " 
       << range.get_ymax() << endl ; 

  double x1,x2,y1,y2 ;
  range.get_window(x1,x2,y1,y2) ;

  cout << "Window: "
       << x1 << " , "
       << x2 << " , "
       << y1 << " , "
       << y2 << endl ;

  plot1.plot_data() ;

  system("display xx.png &") ;


  return 0 ;
} ;
