module propeller_mod
use points_mod
implicit none
!***************************************************************************
type propeller
	integer :: nblades, ncontrolpoint
	real(wp) :: r_prop, r_hub, blade_offset, rot_dir, J, b, phi, n, m, p, relax_factor, rho
	character(11) :: rotation_direction
	type(point),allocatable,dimension(:,:) :: node
	type(point),allocatable,dimension(:) :: cp
	real(wp), allocatable, dimension(:,:) :: chord, Jac
	real(wp), allocatable, dimension(:) :: twist, cla, G, al0, Clmax, R, dG
	type(vector),allocatable,dimension(:) :: V, Vp, un, ua, dF, dM
	type(vector),allocatable,dimension(:,:) :: v_ng
	character(100) :: chord_f, twist_f, cl_f
	type(vector) :: Vinf, w, Force_Total, Moment_Total
	character :: int_type
end type propeller
!***************************************************************************

!***************************************************************************
!***************************************************************************
contains
!***************************************************************************
!***************************************************************************
subroutine allocate_propeller(prop)
	use points_mod
	implicit none
	type(propeller), intent(inout) :: prop
	type(vector) :: zv
	type(point) :: zp
	real(wp) :: zr
	integer :: i,j
	i = prop%nblades
	j = prop%ncontrolpoint
	allocate(prop%node(i*j,2), prop%cp(i*j))
	allocate(prop%chord(i*j,2), prop%twist(i*j), prop%cla(i*j), prop%G(i*j))
	allocate(prop%V(i*j), prop%Vp(i*j), prop%v_ng(i*j,i*j), prop%un(i*j), prop%ua(i*j))
	allocate(prop%al0(i*j), prop%Clmax(i*j), prop%Jac(i*j,i*j), prop%R(i*j), prop%dG(i*j))
	allocate(prop%dF(i*j), prop%dM(i*j))
	zr = 0._wp
	zp%x = zr
	zp%y = zr
	zp%z = zr
	zv%mag = zr
	zv%x = zr
	zv%y = zr
	zv%z = zr
	prop%node = zp
	prop%cp = zp
	prop%chord = zr
	prop%twist = zr
	prop%cla = zr
	prop%G = zr
	prop%V = zv
	prop%Vp = zv
	prop%v_ng = zv
	prop%un = zv
	prop%ua = zv
	prop%al0 = zr
	prop%Clmax = zr
	prop%Jac = zr
	prop%R = zr
	prop%dG = zr
	prop%dF = zv
	prop%dM = zv
end subroutine allocate_propeller
!***************************************************************************
subroutine deallocate_propeller(prop)
	use points_mod
	implicit none
	type(propeller), intent(inout) :: prop
	deallocate(prop%node, prop%cp, prop%chord, prop%twist, prop%cla, prop%G)
	deallocate(prop%V, prop%Vp, prop%v_ng, prop%ua, prop%un, prop%al0, prop%Clmax)
	deallocate(prop%Jac, prop%R, prop%dG, prop%dF, prop%dM)
end subroutine deallocate_propeller
!***************************************************************************
subroutine propeller_points(prop)
	use points_mod
	implicit none
	type(propeller), intent(inout) :: prop
	real(wp) :: r, chord1, chord2, twist1, twist2, cla1, cla2, al01, al02, clmax1, clmax2
	integer :: i
	prop%cp(1)%x = 0._wp
	prop%cp(1)%y = 0._wp
	prop%cp(1)%z = 0._wp
	prop%cp(:) = prop%cp(1)
	prop%node(:,:) = prop%cp(1)
	open(unit = 10, file = prop%chord_f, action = 'read')
	read(10,*)
	read(10,*) r, chord1
	read(10,*) r, chord2
	close(10)
	open(unit = 10, file = prop%twist_f, action = 'read')
	read(10,*)
	read(10,*) r, twist1
	read(10,*) r, twist2
	close(10)
	open(unit = 10, file = prop%cl_f, action = 'read')
	read(10,*)
	read(10,*) r, cla1
	read(10,*) r, cla2
	read(10,*)
	read(10,*)
	read(10,*) r, al01
	read(10,*) r, al02
	read(10,*)
	read(10,*)
	read(10,*) r, clmax1
	read(10,*) r, clmax2
	close(10)
	do i = 0, prop%ncontrolpoint
		r = (prop%r_prop - prop%r_hub) * (1._wp - cos(real(i,wp)*pi/real(prop%ncontrolpoint,wp))) / 2._wp + prop%r_hub
		if (i > 0) then
			prop%node(i,2) = offset_chord(r,prop%r_hub,prop%r_prop,prop%blade_offset)
			prop%chord(i,2) = chordline(prop%r_prop, r)
		end if
		if (i < prop%ncontrolpoint) then
			prop%node(i+1,1) = offset_chord(r,prop%r_hub,prop%r_prop,prop%blade_offset)
			prop%chord(i+1,1) = chordline(prop%r_prop, r)
		end if
	end do
	do i = 1, prop%ncontrolpoint
		r = (prop%r_prop - prop%r_hub) * (1._wp - cos(real(i,wp)*pi/real(prop%ncontrolpoint,wp) &
			- pi/2._wp/real(prop%ncontrolpoint,wp)))/2._wp + prop%r_hub
		prop%cp(i) = offset_chord(r,prop%r_hub,prop%r_prop,prop%blade_offset)
		prop%twist(i) = calc_twist(prop%r_prop, r)
		prop%cla(i) = interpolate(prop%r_hub, prop%r_prop, r, cla1, cla2)
		prop%al0(i) = interpolate(prop%r_hub, prop%r_prop, r, al01, al02) * pi / 180._wp
		prop%Clmax(i) = interpolate(prop%r_hub, prop%r_prop, r, clmax1, clmax2)
	end do
	call rotate_blade(prop)
end subroutine propeller_points
!***************************************************************************
function offset_chord(r,rh,rp,k)
use points_mod
implicit none
real(wp), intent(in) :: r, rh, rp, k
real(wp) :: theta, x, a, b, c
type(point) :: offset_chord
a = 1._wp + (k / rp) ** 2._wp
b = -2._wp * k ** 2._wp / rp
c = k**2._wp - r**2._wp
if (r >= rh) then
	x = (-b+sqrt(b**2._wp-4._wp*a*c)) / 2._wp / a
	offset_chord%x = x
	offset_chord%y = -k*x/rp+k
	offset_chord%z = 0._wp
else
	c = k**2._wp - rh**2._wp
	x = (-b+sqrt(b**2._wp-4._wp*a*c)) / 2._wp / a
	theta = acos(x/rh)
	offset_chord%x = r*cos(theta)
	offset_chord%y = r*sin(theta)
	offset_chord%z = 0._wp
end if
end function offset_chord
!***************************************************************************
subroutine rotate_blade(prop)
use points_mod
implicit none
type(propeller), intent(inout) :: prop
real(wp) :: rotation_angle
integer :: i, j, k
type(point) :: polar
do i = 2, prop%nblades
	rotation_angle = 2._wp * pi / real(prop%nblades,wp) * real(i-1,wp)
	do j = 1, prop%ncontrolpoint
		do k = 1, 2
			polar%x = sqrt(prop%node(j,k)%x**2._wp + prop%node(j,k)%y**2._wp)
			polar%y = atan2(prop%node(j,k)%y, prop%node(j,k)%x) + rotation_angle + prop%phi
			prop%node(j+(i-1)*prop%ncontrolpoint,k)%x = polar%x * cos(polar%y)
			prop%node(j+(i-1)*prop%ncontrolpoint,k)%y = polar%x * sin(polar%y)
			prop%chord(j+(i-1)*prop%ncontrolpoint,k) = prop%chord(j,k)
		end do
	end do
	do j = 1, prop%ncontrolpoint
		polar%x = sqrt(prop%cp(j)%x**2._wp + prop%cp(j)%y**2._wp)
		polar%y = atan2(prop%cp(j)%y, prop%cp(j)%x) + rotation_angle + prop%phi
		prop%cp(j+(i-1)*prop%ncontrolpoint)%x = polar%x * cos(polar%y)
		prop%cp(j+(i-1)*prop%ncontrolpoint)%y = polar%x * sin(polar%y)
		prop%twist(j+(i-1)*prop%ncontrolpoint) = prop%twist(j)
		prop%cla(j+(i-1)*prop%ncontrolpoint) = prop%cla(j)
		prop%al0(j+(i-1)*prop%ncontrolpoint) = prop%al0(j)
		prop%Clmax(j+(i-1)*prop%ncontrolpoint) = prop%Clmax(j)
	end do
end do
i = 1
rotation_angle = 2._wp * pi / real(prop%nblades,wp) * real(i-1,wp)
do j = 1, prop%ncontrolpoint
	do k = 1, 2
		polar%x = sqrt(prop%node(j,k)%x**2._wp + prop%node(j,k)%y**2._wp)
		polar%y = atan2(prop%node(j,k)%y, prop%node(j,k)%x) + rotation_angle + prop%phi
		prop%node(j+(i-1)*prop%ncontrolpoint,k)%x = polar%x * cos(polar%y)
		prop%node(j+(i-1)*prop%ncontrolpoint,k)%y = polar%x * sin(polar%y)
	end do
end do
do j = 1, prop%ncontrolpoint
	polar%x = sqrt(prop%cp(j)%x**2._wp + prop%cp(j)%y**2._wp)
	polar%y = atan2(prop%cp(j)%y, prop%cp(j)%x) + rotation_angle + prop%phi
	prop%cp(j+(i-1)*prop%ncontrolpoint)%x = polar%x * cos(polar%y)
	prop%cp(j+(i-1)*prop%ncontrolpoint)%y = polar%x * sin(polar%y)
end do

end subroutine rotate_blade
!***************************************************************************
subroutine get_data(prop, filename)
	use points_mod
	implicit none
	type(propeller), intent(out) :: prop
	character(100), intent(in) :: filename
	open(unit = 10, file = filename, action = 'read')
	read(10,*) prop%nblades
	read(10,*) prop%ncontrolpoint
	read(10,*) prop%r_prop
	read(10,*) prop%r_hub
	read(10,*) prop%blade_offset
	read(10,*) prop%rotation_direction
	if(prop%rotation_direction == 'lefthanded') then
		prop%rot_dir = -1
	elseif(prop%rotation_direction == 'righthanded') then
		prop%rot_dir = 1
	else
		write(*,*) 'Wrong input for rotation direction in ',filename
		read(*,*)
	end if
	prop%blade_offset = -real(prop%rot_dir,wp)*prop%blade_offset
	read(10,*) prop%Vinf%z
	prop%Vinf%x = 0._wp
	prop%Vinf%y = 0._wp
	prop%Vinf%mag = prop%Vinf%z
	read(10,*) prop%w%z
	prop%w%x = 0._wp
	prop%w%y = 0._wp
	prop%w%z = -2._wp*pi/60._wp*prop%w%z
	prop%w%mag = abs(prop%w%z)
	prop%b = 2._wp*pi*prop%Vinf%z/abs(prop%w%z)
	prop%J = .5_wp*prop%b/prop%r_prop
	read(10,*) prop%rho
	read(10,*) prop%phi
	prop%phi = prop%phi * pi / 180._wp
	read(10,*) prop%n
	read(10,*) prop%m
	read(10,*) prop%p
	read(10,*) prop%int_type
	read(10,*) prop%relax_factor
	read(10,*) prop%chord_f
	read(10,*) prop%twist_f
	read(10,*) prop%cl_f
	close(10)
end subroutine get_data
!***************************************************************************
function nu(prop, i, j)
use points_mod
use helix_func
implicit none
type(propeller), intent(in) :: prop
integer, intent(in) :: i, j
integer :: b, blade
type(vector) :: nu, temp, r1, r2
real(wp) :: a, rot_ang

blade = 1
do b = 1, prop%nblades
	if (j > b * prop%ncontrolpoint) blade = b + 1
end do
rot_ang = 2._wp * pi / real(prop%nblades,wp) * real(blade-1,wp)
a = sqrt( prop%node(j,2)%x**2._wp + prop%node(j,2)%y**2._wp )
call velocity_dim(nu, r1, 1._wp, a, prop%b, prop%n, prop%cp(i), prop%m, prop%p, rot_ang+prop%phi, prop%int_type)
if (i /= j) then
	r1 = prop%node(j,1) .vec. prop%cp(i)
	r2 = prop%node(j,2) .vec. prop%cp(i)
	temp = dv_dim(r1,r2,1._wp)
	nu = nu + temp
end if
a = sqrt( prop%node(j,1)%x**2._wp + prop%node(j,1)%y**2._wp )
call velocity_dim(temp, r1, 1._wp, a, prop%b, prop%n, prop%cp(i), prop%m, prop%p, rot_ang+prop%phi, prop%int_type)
nu = nu - temp
end function nu
!***************************************************************************
function interpolate(y1, y2, y, x1, x2)
	use points_mod
	implicit none
	real(wp), intent(in) :: y1, y2, y, x1, x2
	real(wp) :: interpolate
	interpolate = (y-y1)*(x2-x1)/(y2-y1)+x1
end function interpolate
!***************************************************************************
subroutine propeller_velocities(prop)
use points_mod
implicit none
type(propeller),intent(inout) :: prop
integer :: i, j
type(point) :: p0
type(vector) :: k, temp, r

p0%x = 0._wp
p0%y = 0._wp
p0%z = 0._wp
k%mag = 1._wp
k%x = 0._wp
k%y = 0._wp
k%z = 1._wp

do i = 1, prop%nblades*prop%ncontrolpoint
	r = p0 .vec. prop%cp(i)
	prop%Vp(i) = prop%Vinf + (r .cross. prop%w)
	do j = 1, prop%nblades*prop%ncontrolpoint
		prop%v_ng(i,j) = nu(prop, i, j)
	end do
	! calculate unit normal and unit axial vectors
	temp = r .cross. prop%w
	prop%ua(i) = temp + (tan(prop%twist(i)) * temp%mag * k)
	prop%ua(i) = prop%ua(i) / prop%ua(i)%mag
	prop%un(i) = temp + (tan(prop%twist(i) - pi/2._wp) * temp%mag * k)
	prop%un(i) = prop%un(i) / prop%un(i)%mag
	write(*,*) i
end do

end subroutine propeller_velocities
!***************************************************************************
function calc_aoa(prop, i)
	use points_mod
	implicit none
	type(propeller),intent(in) :: prop
	integer,intent(in) :: i
	real(wp) :: calc_aoa
	calc_aoa = atan( (prop%V(i) .dot. prop%un(i)) / (prop%V(i) .dot. prop%ua(i)) )
end function calc_aoa
!***************************************************************************
subroutine calc_R(prop)
	use points_mod
	implicit none
	type(propeller),intent(inout) :: prop
	integer :: i
	real(wp) :: cl, s1, s2, dS, aoa
	type(vector) :: temp, dl
	do i = 1, prop%nblades*prop%ncontrolpoint
		! calculate dl
		dl = prop%node(i,1) .vec. prop%node(i,2)
		! calculate temp vector
		temp = prop%G(i) * (prop%V(i) .cross. dl)
		! calculate coefficient of lift
		aoa = calc_aoa(prop,i)
		cl = calc_Cl(aoa)
		! calculate dS
		s1 = sqrt( prop%node(i,1)%x**2._wp + prop%node(i,1)%y**2._wp )
		s2 = sqrt( prop%node(i,2)%x**2._wp + prop%node(i,2)%y**2._wp )
		dS = (s2 - s1) * (prop%chord(i,2)+prop%chord(i,1)) / 2._wp
		! calculate R
		prop%R(i) = 2._wp * temp%mag - prop%V(i)%mag ** 2._wp * cl * dS
	end do
end subroutine calc_R
!***************************************************************************
subroutine calc_V(prop)
	use points_mod
	implicit none
	type(propeller),intent(inout) :: prop
	integer :: i
	type(vector) :: temp
	integer :: j
	do i = 1, prop%nblades*prop%ncontrolpoint
		temp%x = 0._wp
		temp%y = 0._wp
		temp%z = 0._wp
		temp%mag = 0._wp
		do j = 1, prop%nblades*prop%ncontrolpoint
			temp = temp + (prop%G(j) * prop%v_ng(i,j))
		end do
		prop%V(i) = prop%Vp(i) + temp
	end do
end subroutine calc_V
!***************************************************************************
function calc_Cl(a)
	use points_mod
	implicit none
	real(wp),intent(in) :: a
	real(wp) :: calc_Cl
	if (a <= .25_wp) then
		calc_Cl = 2._wp * pi * a
	else
		calc_Cl = pi / 2._wp * cos(a) / cos(.25_wp)
	end if
end function calc_Cl
!***************************************************************************
subroutine jacobian(prop)
use points_mod
implicit none
type(propeller),intent(inout) :: prop
integer :: i, j, maximum
type(vector) :: w, dl
real(wp) :: s1, s2, dS, Va, Vn, cl

maximum = prop%nblades * prop%ncontrolpoint

do i = 1, maximum
	
	dl = prop%node(i,1) .vec. prop%node(i,2)
	w = prop%V(i) .cross. dl
	s1 = sqrt( prop%node(i,1)%x**2._wp + prop%node(i,1)%y**2._wp )
	s2 = sqrt( prop%node(i,2)%x**2._wp + prop%node(i,2)%y**2._wp )
	dS = (s2 - s1) * (prop%chord(i,2)+prop%chord(i,1)) / 2._wp
	Va = prop%V(i) .dot. prop%ua(i)
	Vn = prop%V(i) .dot. prop%un(i)
	cl = calc_Cl(calc_aoa(prop,i))
	
	do j = 1, maximum
		
		if (j == i) then
			prop%Jac(i,j) = 2._wp*prop%G(i)*(w.dot.(prop%v_ng(i,j).cross.dl))/w%mag - &
				prop%V(i)%mag**2._wp*prop%cla(i)*(Va*(prop%v_ng(i,j).dot.prop%un(i))-&
				Vn*(prop%v_ng(i,j).dot.prop%ua(i)))/(Vn**2._wp+Va**2._wp)*dS - &
				2._wp*cl*(prop%V(i).dot.prop%v_ng(i,j))*dS + &
				2._wp*w%mag
		else
			prop%Jac(i,j) = 2._wp*prop%G(i)*(w.dot.(prop%v_ng(i,j).cross.dl))/w%mag - &
				prop%V(i)%mag**2._wp*prop%cla(i)*(Va*(prop%v_ng(i,j).dot.prop%un(i))-&
				Vn*(prop%v_ng(i,j).dot.prop%ua(i)))/(Vn**2._wp+Va**2._wp)*dS - &
				2._wp*cl*(prop%V(i).dot.prop%v_ng(i,j))*dS
		end if
		
	end do
end do

end subroutine jacobian
!***************************************************************************
subroutine calc_dG(prop)
	use points_mod
	use matrix_mod
	implicit none
	type(propeller),intent(inout) :: prop
	real(wp),allocatable,dimension(:,:) :: a
	real(wp) :: tol
	integer,allocatable,dimension(:) :: o
	real(wp),allocatable,dimension(:) :: b, x
	integer :: er, n
	n = prop%nblades*prop%ncontrolpoint
	allocate(a(n,n), o(n), b(n), x(n))
	x = 0._wp
	b = -prop%R
	a = prop%Jac
	tol = .0001_wp
	er = 0
	o = 0
	call ludecomp(a, n, tol, o, er)
	if (er /= -1) then
		call substitute(a, o, n, b, x)
	end if
	prop%dG = x
	deallocate(a, o, b, x)
end subroutine calc_dG
!***************************************************************************
subroutine renew_G(prop)
	use points_mod
	implicit none
	type(propeller), intent(inout) :: prop
	integer :: i
	do i = 1, prop%nblades*prop%ncontrolpoint
		prop%G(i) = prop%G(i) + prop%relax_factor * prop%dG(i)
	end do
end subroutine renew_G
!***************************************************************************
subroutine calc_gammas(prop)
use points_mod
implicit none
type(propeller),intent(inout) :: prop
integer :: iter
real(wp) :: tol, error
tol = .0000001_wp
error = 100._wp
iter = 0
do while(tol < error)
	iter = iter + 1
	call calc_V(prop)
	call calc_R(prop)
	call jacobian(prop)
	call calc_dG(prop)
	call renew_G(prop)
	error = maxval(abs(prop%R))
end do
write(*,*)
write(*,*) 'Total iterations',iter
write(*,*) 'R values ',prop%R
write(*,*)
write(*,*) 'Gamma values ',prop%G
end subroutine calc_gammas
!***************************************************************************
subroutine calc_dF(prop)
	use points_mod
	implicit none
	type(propeller),intent(inout) :: prop
	integer :: i
	type(vector) :: dl
	prop%Force_Total%x = 0._wp
	prop%Force_Total%y = 0._wp
	prop%Force_Total%z = 0._wp
	prop%Force_Total%mag = 0._wp
	call calc_V(prop)
	do i = 1, prop%nblades * prop%ncontrolpoint
		dl = prop%node(i,1) .vec. prop%node(i,2)
		prop%dF(i) = prop%rho * prop%G(i) * (prop%V(i) .cross. dl)
		prop%Force_Total = prop%Force_Total + prop%dF(i)
	end do
end subroutine calc_dF
!***************************************************************************
subroutine calc_dM(prop)
	use points_mod
	implicit none
	type(propeller),intent(inout) :: prop
	integer :: i
	type(vector) :: r
	type(point) :: cg
	cg%x = 0._wp
	cg%y = 0._wp
	cg%z = 0._wp
	prop%Moment_Total%x = 0._wp
	prop%Moment_Total%y = 0._wp
	prop%Moment_Total%z = 0._wp
	prop%Moment_Total%mag = 0._wp
	do i = 1, prop%nblades * prop%ncontrolpoint
		r = cg .vec. prop%cp(i)
		!prop%dM(i) = 
		prop%Moment_Total = prop%Moment_Total + (r .cross. prop%dF(i))
	end do
end subroutine calc_dM
!***************************************************************************
function chordline(rp, r)
	use points_mod
	implicit none
	real(wp),intent(in) :: r, rp
	real(wp) :: chordline, zeta
	zeta = r / rp
	chordline = .075_wp * rp * 2._wp * sqrt(1._wp - zeta ** 2._wp)
end function chordline
!***************************************************************************
function calc_twist(rp, r)
	use points_mod
	implicit none
	real(wp),intent(in) :: rp, r
	real(wp) :: calc_twist, zeta, K, Kc, t
	zeta = r / rp
	Kc = .9_wp
	t = tan(-2.1_wp*pi/180._wp)
	K = pi*zeta*(Kc-pi*zeta*t)/(pi*zeta+Kc*t)
	calc_twist = atan(K/pi/zeta)
	if (calc_twist < 0._wp) then
		write(*,*) 'Twist is ',calc_twist,'		Adding 90 deg!'
		calc_twist = calc_twist + pi/2._wp
	end if
end function calc_twist
!***************************************************************************
subroutine iter(prop, J)
	use points_mod
	implicit none
	type(propeller), intent(inout) :: prop
	real(wp), intent(in) :: J
	
	prop%J = J
	prop%Vinf%z = J * prop%r_prop * prop%w%mag / pi
	prop%b = 2._wp*pi*prop%Vinf%z/abs(prop%w%z)
	
	call propeller_velocities(prop)
	call calc_gammas(prop)
	call calc_dF(prop)
	call calc_dM(prop)
	
end subroutine iter
!***************************************************************************
end module propeller_mod










!dl = prop%node(i,1) .vec. prop%node(i,2)
!s1 = sqrt( prop%node(i,1)%x**2._wp + prop%node(i,1)%y**2._wp )
!s2 = sqrt( prop%node(i,2)%x**2._wp + prop%node(i,2)%y**2._wp )
!dS = (s2 - s1) * (prop%chord(i,2)+prop%chord(i,1)) / 2._wp
