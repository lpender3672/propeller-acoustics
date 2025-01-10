! ZachMontgomery 2018
! A Propeller Model Based on a Modern Numerical Lifting-LineA Propeller Model Based on a Modern Numerical Lifting-Line
! Algorithm with an Iterative Semi-Free Wake SolverAlgorithm with an Iterative Semi-Free Wake Solver
! https://github.com/ZachMontgomery/PropellerLiftingLineMethod/tree/master
module pll
use iso_c_binding
implicit none
contains

subroutine propeller_lifting_line(prop, oper)
use points_mod
use helix_func
use propeller_mod
use operating_mod
implicit none
!***************************************************************************
type(propeller), intent(inout) :: prop
type(operating), intent(in) :: oper
character(100) :: filename
real(wp) :: J, CT, CP, CL
integer :: i
!***************************************************************************

! all done in python now
!call get_command_argument(1,filename)
!call get_data(prop, filename)
!call allocate_propeller(prop)
!call propeller_points(prop)
prop%G = 100._wp

!open(unit = 10, file = 'variable_n_J_.25.txt')

!do i = 0, 10
	J = .9_wp
	prop%n = 100!ceiling(interpolate(0._wp,1._wp,real(i,wp)/10._wp,500._wp,100._wp))
!	write(*,*) 'Running case ',i
	call iter(prop,J)
	CT = prop%Force_Total%mag / prop%rho / prop%w%mag **2._wp / (prop%r_prop * 2._wp) ** 4._wp * 4._wp * pi ** 2._wp
	CL = prop%Moment_Total%mag / prop%rho / prop%w%mag **2._wp / (prop%r_prop * 2._wp) ** 5._wp * 4._wp * pi ** 2._wp
	CP = CL * 2._wp * pi
	
	write(*,'(1x, f5.1, 4x, 3(f20.18, 4x))') prop%n, CT, CP, CL
	
!end do

!close(10)
!***************************************************************************
call deallocate_propeller(prop)
!***************************************************************************
!***************************************************************************
!***************************************************************************
!***************************************************************************
contains
!***************************************************************************
!***************************************************************************
!***************************************************************************
!***************************************************************************

!***************************************************************************
end subroutine propeller_lifting_line

end module pll