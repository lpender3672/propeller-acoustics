module matrix_mod

contains
!***************************************************************************
subroutine ludecomp(a, n, tol, o, er)
use points_mod
implicit none
integer, intent(in) :: n
integer, intent(inout), dimension(n) :: o
real(wp), intent(in) :: tol
real(wp), intent(inout), dimension(n,n) :: a
real(wp), dimension(n) :: s
integer, intent(inout) :: er
er = 0
call decompose(a, n, tol, o, s, er)
end subroutine ludecomp
!***************************************************************************
subroutine decompose(a, n, tol, o, s, er)
use points_mod
implicit none
integer, intent(in) :: n
integer, intent(inout) :: er
integer, intent(inout), dimension(n) :: o
integer :: k, i, j
real(wp), intent(in) :: tol
real(wp), intent(inout), dimension(n) :: s
real(wp), intent(inout), dimension(n,n) :: a
real(wp) :: factor
do i = 1, n
	o(i) = i
	s(i) = abs(a(i,1))
	do j = 2, n
		if (abs(a(i,j)) > s(i)) s(i) = abs(a(i,j))
	end do
end do
do k = 1, n-1
	call pivot(a, o, s, n, k)
	if (abs(a(o(k),k)/s(o(k))) < tol) then
		er = -1
		write(*,*) a(o(k),k)/s(o(k))
		exit
	end if
	do i = k+1, n
		factor = a(o(i),k)/a(o(k),k)
		a(o(i),k) = factor
		do j = k+1,n
			a(o(i),j) = a(o(i),j) - factor*a(o(k),j)
		end do
	end do
end do
if (abs(a(o(k),k)/s(o(k)))<tol) then
	er = -1
	write(*,*) a(o(k),k)/s(o(k))
end if
end subroutine decompose
!***************************************************************************
subroutine pivot(a, o, s, n, k)
use points_mod
implicit none
integer, intent(in) :: n, k
integer, intent(inout), dimension(n) :: o
real(wp), intent(inout), dimension(n) :: s
real(wp), intent(inout), dimension(n,n) :: a
real(wp) :: big, dummy
integer :: p, dumb, i
p = k
big = abs(a(o(k),k)/s(o(k)))
do i = k+1,n
	dummy = abs(a(o(i),k)/s(o(i)))
	if (dummy > big) then
		big = dummy
		p = i
	end if
end do
dumb = o(p)
o(p) = o(k)
o(k) = dumb
end subroutine pivot
!***************************************************************************
subroutine substitute(a, o, n, b, gamma)
use points_mod
implicit none
integer, intent(in) :: n
integer, intent(inout), dimension(n) :: o
real(wp), intent(inout), dimension(n) :: b, gamma
real(wp), intent(inout), dimension(n,n) :: a
real(wp) :: total
integer :: i, j
do i = 2, n
	total = b(o(i))
	do j = 1, i-1
		total = total - a(o(i),j)*b(o(j))
	end do
	b(o(i)) = total
end do
gamma(n) = b(o(n))/a(o(n),n)
do i = n-1,1,-1
	total = 0._wp
	do j = i+1,n
		total = total + a(o(i),j)*gamma(j)
	end do
	gamma(i) = (b(o(i)) - total)/a(o(i),i)
end do
end subroutine substitute
!***************************************************************************

!***************************************************************************
end module matrix_mod
