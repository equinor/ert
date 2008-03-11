program main
  integer, parameter :: ntries = 5
  integer, parameter :: nx = 40
  integer, parameter :: ny = 40
  integer, parameter :: ndim = nx*ny
  integer, parameter :: ns = 10

  integer :: nbd(ndim)

  real    :: y(nx,ny,ns)
  real    :: alpha(ns)
  real    :: low_bnd(ndim)
  real    :: high_bnd(ndim)
  real    :: x(ndim)


  !******************************************!

  call m_pseudo2d_mp_pseudo2d(y,nx,ny,ns,300.0,100.0,25.0,25.0,nx+20,ny+20,40,.true.)
  call random_number(alpha)

  nbd = 0
  low_bnd = 0.0
  high_bnd = 1.0


  write(*,*) "Solving pre-image problem for a 1st order dot product kernel"
  call fw_pre_image_approx_dot_xpsqx(ntries,ndim,ns,y,alpha,low_bnd,high_bnd,nbd,x)

  write(*,*) "Solving pre-image problem for a 2nd order dot product kernel"
  call fw_pre_image_approx_2xdot_xpsqx(ntries,ndim,ns,y,alpha,low_bnd,high_bnd,nbd,x)

  write(*,*) "Solving pre-image problem for a 3rd order dot product kernel"
  call fw_pre_image_approx_3xdot_xpsqx(ntries,ndim,ns,y,alpha,low_bnd,high_bnd,nbd,x)

end program main
