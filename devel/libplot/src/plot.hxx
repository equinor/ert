/*
 * $Id$
 */

#ifndef __PLOT_HXX__
#define __PLOT_HXX__

#include "plot.h"
#include "plot_dataset.h"

#include "plot_range.hxx"

#include <string>
#include <vector>
#include <stdexcept>

class plot ;

/**
 * The class plot is a C++ wrapper for the corresponding C class
 * plot_type
 */

class plot 
{
public:

  plot(const std::string & dev,
       const std::string & filename) {
    m_plot = plot_alloc() ;
    plot_initialize(m_plot,dev.c_str(),filename.c_str()) ;
  } ;

  ~plot() {
    if (m_plot) { plot_free(m_plot) ;} ;
  } ;

  void initialize(const std::string & dev,
                  const std::string & filename) {
    if (m_plot) { plot_free(m_plot) ;} ;
    m_plot = plot_alloc() ;
    plot_initialize(m_plot,dev.c_str(),filename.c_str()) ;
  } ;

  void set_xlabel(const std::string & label) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_xlabel() called after plot_data()") ; 
    } ;
    plot_set_xlabel(m_plot,label.c_str()) ;
  } ;

  void set_ylabel(const std::string & label) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_ylabel() called after plot_data()") ; 
    } ;
    plot_set_ylabel(m_plot,label.c_str()) ;
  } ;

  void set_title(const std::string & label) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_title() called after plot_data()") ; 
    } ;
    plot_set_title(m_plot,label.c_str()) ;
  } ;

  void set_labels(const std::string & xlabel ,
                  const std::string & ylabel ,
                  const std::string & title  ) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_labels() called after plot_data()") ; 
    } ;
    plot_set_labels(m_plot,xlabel.c_str(),ylabel.c_str(),title.c_str()) ;
  } ;

  int new_dataset(plot_data_type data_type) {
    if (!m_plot) { 
      throw std::logic_error("class plot: new_dataset() called after plot_data()") ; 
    } ;
    plot_alloc_new_dataset(m_plot,data_type,false) ;
    return get_num_datasets() - 1 ;
  } ;

  void plot_data() {
    if (!m_plot) { 
      throw std::logic_error("class plot: plot_data() called after plot_data()") ; 
    } ;
    ::plot_data(m_plot) ;
    plot_free(m_plot) ; // Plot is not finalized without this ..
    m_plot = 0 ;
  } ;

  /* For some reason get_extrema does not return a range with correct padding ... */

  void get_extrema(plot_range & range) {
    if (!m_plot) { 
      throw std::logic_error("class plot: get_extrema() called after plot_data()") ; 
    } ;
    plot_get_extrema(m_plot,range.c_data()) ;
  } ;

  void set_window_size(int width, int height) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_window_size() called after plot_data()") ; 
    } ;
    plot_set_window_size(m_plot,width,height) ;
  } ;

  void invert_y_axis() {
    if (!m_plot) { 
      throw std::logic_error("class plot: invert_y_axis() called after plot_data()") ; 
    } ;
    plot_invert_y_axis(m_plot) ;
  } ;

  void invert_x_axis() {
    if (!m_plot) { 
      throw std::logic_error("class plot: invert_x_axis() called after plot_data()") ; 
    } ;
    plot_invert_x_axis(m_plot) ;
  } ;

  void set_top_padding(double pad) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_top_padding() called after plot_data()") ; 
    } ;
    plot_set_top_padding(m_plot,pad) ;
  } ;

  void set_bottom_padding(double pad) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_bottom_padding() called after plot_data()") ; 
    } ;
    plot_set_bottom_padding(m_plot,pad) ;
  } ;

  void set_left_padding(double pad) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_left_padding() called after plot_data()") ; 
    } ;
    plot_set_left_padding(m_plot,pad) ;
  } ;

  void set_right_padding(double pad) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_right_padding() called after plot_data()") ; 
    } ;
    plot_set_right_padding(m_plot,pad) ;
  } ;

  void set_label_color(plot_color_type col) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_label_color() called after plot_data()") ; 
    } ;
    plot_set_label_color(m_plot,col) ;
  } ;

  void set_box_color(plot_color_type col) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_box_color() called after plot_data()") ; 
    } ;
    plot_set_box_color(m_plot,col) ;
  } ;

  void set_label_fontsize(double siz) {
    if (!m_plot) { 
      throw std::logic_error("class plot: set_label_fontsize() called after plot_data()") ; 
    } ;
    plot_set_label_fontsize(m_plot,siz) ;
  } ;

  /* ------------------ */
  
  void set_symbol_type(int dataset, plot_symbol_type symb) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_symbol_type(plot_get_datasets(m_plot)[dataset],symb) ;
  } ;
  
  void set_style(int dataset, plot_style_type style) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_style(plot_get_datasets(m_plot)[dataset],style) ;
  } ;
  
  void set_line_color(int dataset, plot_color_type col) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_line_color(plot_get_datasets(m_plot)[dataset],col) ;
  } ;
  
  void set_point_color(int dataset, plot_color_type col) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_point_color(plot_get_datasets(m_plot)[dataset],col) ;
  } ;
  
  void set_line_style(int dataset, plot_line_style_type sty) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_line_style(plot_get_datasets(m_plot)[dataset],sty) ;
  } ;
  
  void set_symbol_size(int dataset, double siz) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_symbol_size(plot_get_datasets(m_plot)[dataset],siz) ;
  } ;
  
  void set_line_width(int dataset, double siz) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_set_line_width(plot_get_datasets(m_plot)[dataset],siz) ;
  } ;
  
  int get_num_datasets() {
    if (!m_plot) { return 0 ;}
    else { return  plot_get_num_datasets(m_plot) ; }
  } ;

  void append_point_xy(int dataset, double x, double y) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_point_xy(plot_get_datasets(m_plot)[dataset],x,y) ;
  } ;

  void append_point_xy1y2(int dataset, double x, double y1, double y2) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_point_xy1y2(plot_get_datasets(m_plot)[dataset],x,y1,y2) ;
  } ;

  void append_point_x1x2y(int dataset, double x1, double x2, double y) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_point_x1x2y(plot_get_datasets(m_plot)[dataset],x1,x2,y) ;
  } ;

  void append_vector_xy(int dataset, 
                        const std::vector<double> & x, 
                        const std::vector<double> & y) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    int siz = x.size() ;
    if (siz != y.size()) { return ; } ;
    plot_dataset_append_vector_xy(plot_get_datasets(m_plot)[dataset],
                                  siz, &x[0], &y[0]) ;
  } ;

 void append_vector_xy1y2(int dataset, 
                          const std::vector<double> & x , 
                          const std::vector<double> & y1 ,
                          const std::vector<double> & y2 ) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    int siz = x.size() ;
    if (siz != y1.size() || siz != y2.size()) { return ; } ;
    plot_dataset_append_vector_xy1y2(plot_get_datasets(m_plot)[dataset],
                                  siz, &x[0], &y1[0], &y2[0]) ;
  } ;


 void append_vector_x1x2y(int dataset, 
                          const std::vector<double> & x1 , 
                          const std::vector<double> & x2 ,
                          const std::vector<double> & y  ) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    int siz = x1.size() ;
    if (siz != x2.size() || siz != y.size()) { return ; } ;
    plot_dataset_append_vector_x1x2y(plot_get_datasets(m_plot)[dataset],
                                  siz, &x1[0], &x2[0], &y[0]) ;
  } ;


  void append_point_xline(int dataset,double val) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_point_xline(plot_get_datasets(m_plot)[dataset],val) ;
  } ;

  void append_point_yline(int dataset,double val) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_point_yline(plot_get_datasets(m_plot)[dataset],val) ;
  } ;

  void append_vector_xline(int dataset,const std::vector<double> & val) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_vector_xline(plot_get_datasets(m_plot)[dataset],
                                     val.size(),&val[0]) ;
  } ;

  void append_vector_yline(int dataset,const std::vector<double> & val) {
    if (dataset < 0 || dataset >= get_num_datasets() ) { return ; } ;
    plot_dataset_append_vector_yline(plot_get_datasets(m_plot)[dataset],
                                     val.size(),&val[0]) ;
  } ;

private:

  plot() ;
  plot(const plot & src) ;

  plot_type* m_plot ;

} ;


#endif
