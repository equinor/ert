/*
 * $Id:$
 */

#ifndef __PLOT_RANGE_HXX__
#define __PLOT_RANGE_HXX__

#include "plot_range.h"

class plot_range ;

/**
 * The class plot_range is a C++ wrapper for the C corresponding c class
 * plot_range_type
 */

class plot_range 
{
public:

  plot_range() {
    m_plot_range = plot_range_alloc() ;
  } ;

  ~plot_range() {
    plot_range_free(m_plot_range) ;
  } ;

  
  double get_xmax() const {
    return  plot_range_get_xmax(m_plot_range) ;
  } ;

  double get_ymax() const {
    return  plot_range_get_ymax(m_plot_range) ;
  } ;

  double get_xmin() const {
    return  plot_range_get_xmin(m_plot_range) ;
  } ;

  double get_ymin() const {
    return  plot_range_get_ymin(m_plot_range) ;
  } ;

  void set_xmax(double val) {
    return  plot_range_set_xmax(m_plot_range,val) ;
  } ;

  void set_ymax(double val) {
    return  plot_range_set_ymax(m_plot_range,val) ;
  } ;

  void set_xmin(double val) {
    return  plot_range_set_xmin(m_plot_range,val) ;
  } ;

  void set_ymin(double val) {
    return  plot_range_set_ymin(m_plot_range,val) ;
  } ;

  void set_top_padding(double val) {
    return  plot_range_set_top_padding(m_plot_range,val) ;
  } ;

  void set_bottom_padding(double val) {
    return  plot_range_set_bottom_padding(m_plot_range,val) ;
  } ;

  void set_left_padding(double val) {
    return  plot_range_set_left_padding(m_plot_range,val) ;
  } ;

  void set_right_padding(double val) {
    return  plot_range_set_right_padding(m_plot_range,val) ;
  } ;

  void invert_x_axis(bool val) {
    return  plot_range_invert_x_axis(m_plot_range,val) ;
  } ;

  void invert_y_axis(bool val) {
    return  plot_range_invert_y_axis(m_plot_range,val) ;
  } ;

  /**
   * Returns window extent in user cooridnates;
   * Window extent is max/min + padding
   *
   * The function wraps the C function plot_range_apply
   */
  void get_window(double & x1 ,
                  double & x2 ,
                  double & y1 ,
                  double & y2  ) {
    plot_range_apply(m_plot_range,&x1,&x2,&y1,&y2) ;
  } ;

  /* ****** */

  plot_range_type* c_data() {
    return m_plot_range ;
  } ;
  
private:

  plot_range_type* m_plot_range ;
} ;


#endif
