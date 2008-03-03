program main
  integer, parameter :: ntries = 5
  integer, parameter :: nx = 200
  integer, parameter :: ny = 200
  integer, parameter :: ndim = nx*ny
  integer, parameter :: ns = 10

  integer :: nbd(ndim)

  real    :: y(nx,ny,ns)
  real    :: alpha(ns)
  real    :: low_bnd(ndim)
  real    :: high_bnd(ndim)
  real    :: x(ndim)


  !******************************************!

  integer :: i,j

  call m_pseudo2d_mp_pseudo2d(y,nx,ny,ns,300.0,100.0,25.0,25.0,nx+20,ny+20,40,.true.)
  call random_number(alpha)

  nbd = 0
  low_bnd = 0.0
  high_bnd = 1.0

  call fw_pre_image_approx_2xdot_xpsqx(ntries,ndim,ns,y,alpha,low_bnd,high_bnd,nbd,x)
  call fw_pre_image_approx_dot_xpsqx(ntries,ndim,ns,y,alpha,low_bnd,high_bnd,nbd,x)

end program main
