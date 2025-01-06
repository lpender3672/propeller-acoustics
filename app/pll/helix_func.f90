module helix_func

contains
!***************************************************************************
! dimensional
!***************************************************************************
subroutine velocity_dim(myfunc, ham, gamma, a, b, n, control, m, p, phi_deg, int_type)
use points_mod
implicit none
real(wp), intent(in) :: a, b, n, m, gamma, p, phi_deg
type(point), intent(in) :: control
type(vector), intent(out) :: myfunc, ham
character, intent(in) :: int_type
real(wp) :: dzeta, theta, denom, s, c, zeta_temp, phi
real(wp) :: c0, c1, c2, cx1, cx2, cx3, cx4, cy1, cy2, cy3, cy4, cz1, cz2, cz3, d1
type(vector) :: v, w, k1, k2, k3, k4, r1, r2, dv_o, dv_n
integer :: i, steps
real(wp), allocatable, dimension(:) :: zeta
! initialize values
phi = phi_deg
steps = nint(n*m)
allocate(zeta(0:steps))
! calculate constants
c0 = 2._wp * pi * n
c1 = gamma / 4._wp / pi
c2 = gamma / 2._wp
cx1 = -c1 * b * n * control%y
cx2 = c1 * a * b * n
cx3 = c2 * a * n * control%z
cx4 = -c2 * a * b * n ** 2._wp
cy1 = c1 * b * n * control%x
cy2 = -c1 * a * b * n
cy3 = c2 * a * n * control%z
cy4 = -c2 * a * b * n ** 2._wp
cz1 = c2 * a ** 2._wp * n
cz2 = -c2 * a * n * control%x
cz3 = -c2 * a * n * control%y
d1 = b * n
! calculate size of integration step and number of steps
dzeta = 1._wp / n / m
do i = 0, steps
	zeta(i) = (real(i,wp)*dzeta)**p
end do
! initialize vectors
v%x = 0._wp
v%y = 0._wp
v%z = 0._wp
v%mag = 0._wp
k1 = v
k2 = v
k3 = v
k4 = v
w = v
! initialize values for loop
theta = phi
s = sin(theta)
c = cos(theta)
denom = ((control%x-a*c)**2._wp+(control%y-a*s)**2._wp+control%z**2._wp)**(1.5_wp)
dv_o%x = dvx(0._wp,denom,theta,cx1,cx2,cx3,cx4,s,c)
dv_o%y = dvy(0._wp,denom,theta,cy1,cy2,cy3,cy4,s,c)
dv_o%z = dvz(0._wp,denom,theta,cz1,cz2,cz3,s,c)
! check for trapezoidal rule or runge-kutta 4
if (int_type == 't') then
	! loop for all segments
	do i = 1, steps
		! calculate common items between the x, y, and z velocities
		dzeta = zeta(i) - zeta(i-1)
		theta = c0 * zeta(i) + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-a*c)**2._wp+(control%y-a*s)**2._wp+(control%z-d1*zeta(i))**2._wp)**(1.5_wp)
		dv_n%x = dvx(zeta(i),denom,theta,cx1,cx2,cx3,cx4,s,c)
		dv_n%y = dvy(zeta(i),denom,theta,cy1,cy2,cy3,cy4,s,c)
		dv_n%z = dvz(zeta(i),denom,theta,cz1,cz2,cz3,s,c)
		! add on the trapezoidal area
		v = v + dzeta / 2._wp * (dv_o + dv_n)
		! save values
		dv_o = dv_n
		! ham method
		call geometry_vec_dim(r1,r2,zeta(i),dzeta,a,phi,control,c0,d1)
		w = w + dv_dim(r1,r2,gamma)
	end do
elseif (int_type == 'r') then
	! loop for all segments
	do i = 1, steps
		! calculate dzeta
		dzeta = zeta(i) - zeta(i-1)
		! calculate k1
		k1 = dv_o
		! calculate k2
		zeta_temp = zeta(i-1) + dzeta / 2._wp
		theta = c0 * zeta_temp + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-a*c)**2._wp+(control%y-a*s)**2._wp+(control%z-d1*zeta_temp)**2._wp)**(1.5_wp)
		k2%x = dvx(zeta_temp,denom,theta,cx1,cx2,cx3,cx4,s,c)
		k2%y = dvy(zeta_temp,denom,theta,cy1,cy2,cy3,cy4,s,c)
		k2%z = dvz(zeta_temp,denom,theta,cz1,cz2,cz3,s,c)
		! calculate k3
		!k3 = k2
		! calculate k4
		theta = c0 * zeta(i) + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-a*c)**2._wp+(control%y-a*s)**2._wp+(control%z-d1*zeta(i))**2._wp)**(1.5_wp)
		k4%x = dvx(zeta(i),denom,theta,cx1,cx2,cx3,cx4,s,c)
		k4%y = dvy(zeta(i),denom,theta,cy1,cy2,cy3,cy4,s,c)
		k4%z = dvz(zeta(i),denom,theta,cz1,cz2,cz3,s,c)
		dv_n = k4
		! add on to the final values
		v = v + dzeta / 6._wp * (k1 + 4._wp * k2 + k4)
		! save values
		dv_o = dv_n
		! ham method
		call geometry_vec_dim(r1,r2,zeta(i),dzeta,a,phi,control,c0,d1)
		w = w + dv_dim(r1,r2,gamma)
	end do
else
	write(*,*)
	write(*,*) '**************************************'
	write(*,*) "Error int_type must be 't' or 'r'"
	write(*,*) '**************************************'
	write(*,*)
end if
! now that we have the x, y, and z velocities, we can calculate the magnitude
v%mag = sqrt(v%x**2 + v%y**2 + v%z**2)
w%mag = sqrt(w%x**2 + w%y**2 + w%z**2)
! set the output variable
myfunc = v
ham = w
deallocate(zeta)
end subroutine velocity_dim
!***************************************************************************
! non dimensional
!***************************************************************************
subroutine velocity_nondim(myfunc, ham, lambda, n, control, m, p, phi_deg, int_type)
use points_mod
implicit none
real(wp), intent(in) :: lambda, n, m, p, phi_deg
type(point), intent(in) :: control
type(vector), intent(out) :: myfunc, ham
character, intent(in) :: int_type
real(wp) :: dzeta, theta, denom, s, c, zeta_temp, phi
real(wp) :: c0, c1, c2, cx1, cx2, cx3, cx4, cy1, cy2, cy3, cy4, cz1, cz2, cz3, d1
type(vector) :: v, w, k1, k2, k3, k4, r1, r2, dv_o, dv_n
integer :: i, steps
real(wp), allocatable, dimension(:) :: zeta
! initialize values
phi = pi / 180._wp * phi_deg
steps = nint(n*m)
allocate(zeta(0:steps))
! calculate constants
c0 = 2._wp * pi * n
c1 = n / 4._wp / pi
c2 = n / 2._wp
cx1 = -c1 * lambda * control%y
cx2 = c1 * lambda
cx3 = c2 * control%z
cx4 = -c2 * lambda * n
cy1 = c1 * lambda * control%x
cy2 = -c1 * lambda
cy3 = c2 * control%z
cy4 = -c2 * lambda * n
cz1 = c2
cz2 = -c2 * control%x
cz3 = -c2 * control%y
d1 = lambda * n
! calculate size of integration step and number of steps
dzeta = 1._wp / n / m
do i = 0, steps
	zeta(i) = (real(i,wp)*dzeta)**p
end do
! initialize vectors
v%x = 0._wp
v%y = 0._wp
v%z = 0._wp
v%mag = 0._wp
k1 = v
k2 = v
k3 = v
k4 = v
w = v
! initialize values for loop
theta = phi
s = sin(theta)
c = cos(theta)
denom = ((control%x-c)**2._wp+(control%y-s)**2._wp+control%z**2._wp)**(1.5_wp)
dv_o%x = dvx(0._wp,denom,theta,cx1,cx2,cx3,cx4,s,c)
dv_o%y = dvy(0._wp,denom,theta,cy1,cy2,cy3,cy4,s,c)
dv_o%z = dvz(0._wp,denom,theta,cz1,cz2,cz3,s,c)
! check for trapezoidal rule or runge-kutta 4
if (int_type == 't') then
	! loop for all segments
	do i = 1, steps
		! calculate common items between the x, y, and z velocities
		dzeta = zeta(i) - zeta(i-1)
		theta = c0 * zeta(i) + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-c)**2._wp+(control%y-s)**2._wp+(control%z-d1*zeta(i))**2._wp)**(1.5_wp)
		dv_n%x = dvx(zeta(i),denom,theta,cx1,cx2,cx3,cx4,s,c)
		dv_n%y = dvy(zeta(i),denom,theta,cy1,cy2,cy3,cy4,s,c)
		dv_n%z = dvz(zeta(i),denom,theta,cz1,cz2,cz3,s,c)
		! add on the trapezoidal area
		v = v + dzeta / 2._wp * (dv_o + dv_n)
		! save values
		dv_o = dv_n
		! ham method
		call geometry_vec_nondim(r1,r2,zeta(i),dzeta,phi,control,c0,d1)
		w = w + dv_nondim(r1,r2)
	end do
elseif (int_type == 'r') then
	! loop for all segments
	do i = 1, steps
		! calculate dzeta
		dzeta = zeta(i) - zeta(i-1)
		! calculate k1
		k1 = dv_o
		! calculate k2
		zeta_temp = zeta(i-1) + dzeta / 2._wp
		theta = c0 * zeta_temp + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-c)**2._wp+(control%y-s)**2._wp+(control%z-d1*zeta_temp)**2._wp)**(1.5_wp)
		k2%x = dvx(zeta_temp,denom,theta,cx1,cx2,cx3,cx4,s,c)
		k2%y = dvy(zeta_temp,denom,theta,cy1,cy2,cy3,cy4,s,c)
		k2%z = dvz(zeta_temp,denom,theta,cz1,cz2,cz3,s,c)
		! calculate k3
		!k3 = k2
		! calculate k4
		theta = c0 * zeta(i) + phi
		s = sin(theta)
		c = cos(theta)
		denom = ((control%x-c)**2._wp+(control%y-s)**2._wp+(control%z-d1*zeta(i))**2._wp)**(1.5_wp)
		k4%x = dvx(zeta(i),denom,theta,cx1,cx2,cx3,cx4,s,c)
		k4%y = dvy(zeta(i),denom,theta,cy1,cy2,cy3,cy4,s,c)
		k4%z = dvz(zeta(i),denom,theta,cz1,cz2,cz3,s,c)
		dv_n = k4
		! add on to the final values
		v = v + dzeta / 6._wp * (k1 + 4._wp * k2 + k4)
		! save values
		dv_o = dv_n
		! ham method
		call geometry_vec_nondim(r1,r2,zeta(i),dzeta,phi,control,c0,d1)
		w = w + dv_nondim(r1,r2)
	end do
else
	write(*,*)
	write(*,*) '**************************************'
	write(*,*) "Error int_type must be 't' or 'r'"
	write(*,*) '**************************************'
	write(*,*)
end if
! now that we have the x, y, and z velocities, we can calculate the magnitude
v%mag = sqrt(v%x**2 + v%y**2 + v%z**2)
w%mag = sqrt(w%x**2 + w%y**2 + w%z**2)
! set the output variable
myfunc = v
ham = w
deallocate(zeta)
end subroutine velocity_nondim
!***************************************************************************
! my functions
!***************************************************************************
function dvx(zeta,denom,theta,cx1,cx2,cx3,cx4,s,c)
use points_mod
implicit none
real(wp), intent(in) :: zeta,denom,theta,cx1,cx2,cx3,cx4,s,c
real(wp) :: dvx
dvx = (cx1 + cx2 * s + (cx3 + cx4 * zeta) * c)/denom
end function dvx
!***************************************************************************
function dvy(zeta,denom,theta,cy1,cy2,cy3,cy4,s,c)
use points_mod
implicit none
real(wp), intent(in) :: zeta,denom,theta,cy1,cy2,cy3,cy4,s,c
real(wp) :: dvy
dvy = (cy1 + cy2 * c + (cy3 + cy4 * zeta) * s)/denom
end function dvy
!***************************************************************************
function dvz(zeta,denom,theta,cz1,cz2,cz3,s,c)
use points_mod
implicit none
real(wp), intent(in) :: zeta,denom,theta,cz1,cz2,cz3,s,c
real(wp) :: dvz
dvz = (cz1 + cz2 * c + cz3 * s)/denom
end function dvz
!***************************************************************************
! HAM functions
!***************************************************************************
function dv_dim(r1,r2,gamma)
use points_mod
implicit none
real(wp), intent(in) :: gamma
type(vector), intent(in) :: r1, r2
type(vector) :: dv_dim
dv_dim = gamma / 4._wp / pi * (r1%mag + r2%mag) * (r1 .cross. r2) &
	/ r1%mag / r2%mag / (r1%mag * r2%mag + (r1 .dot. r2))
end function dv_dim
!***************************************************************************
subroutine geometry_vec_dim(r1,r2,zeta,dzeta,a,phi,control,c0, d1)
use points_mod
implicit none
real(wp), intent(in) :: a, zeta, dzeta, c0, d1, phi
type(point), intent(in) :: control
type(vector), intent(out) :: r1, r2
type(point) :: p1, p2
real(wp) :: theta
! point 1
theta = c0 * (zeta - dzeta) + phi
p1%x = a * cos(theta)
p1%y = a * sin(theta)
p1%z = d1 * (zeta - dzeta)
! point 2
theta = c0 * zeta + phi
p2%x = a * cos(theta)
p2%y = a * sin(theta)
p2%z = d1 * zeta
! calculate vectors
r1 = control .vec. p1
r2 = control .vec. p2
end subroutine geometry_vec_dim
!***************************************************************************
function dv_nondim(r1_hat,r2_hat)
use points_mod
implicit none
type(vector), intent(in) :: r1_hat, r2_hat
type(vector) :: dv_nondim
dv_nondim = (r1_hat%mag + r2_hat%mag) * (r1_hat .cross. r2_hat) / 4._wp / pi &
	/ r1_hat%mag / r2_hat%mag / (r1_hat%mag * r2_hat%mag + (r1_hat .dot. r2_hat))
end function dv_nondim
!***************************************************************************
subroutine geometry_vec_nondim(r1_hat,r2_hat,zeta,dzeta,phi,control_hat,c0, d1)
use points_mod
implicit none
real(wp), intent(in) :: zeta, dzeta, c0, d1, phi
type(point), intent(in) :: control_hat
type(vector), intent(out) :: r1_hat, r2_hat
type(point) :: p1, p2
real(wp) :: theta
! point 1
theta = c0 * (zeta - dzeta) + phi
p1%x = cos(theta)
p1%y = sin(theta)
p1%z = d1 * (zeta - dzeta)
! point 2
theta = c0 * zeta + phi
p2%x = cos(theta)
p2%y = sin(theta)
p2%z = d1 * zeta
! calculate vectors
r1_hat = control_hat .vec. p1
r2_hat = control_hat .vec. p2
end subroutine geometry_vec_nondim
!***************************************************************************
end module helix_func
