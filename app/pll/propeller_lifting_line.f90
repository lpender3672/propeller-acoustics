program propeller_lifting_line
use points_mod
use helix_func
use propeller_mod
implicit none
!***************************************************************************
type(propeller) :: prop
character(100) :: filename
real :: cpu1, cpu2
integer :: i
!***************************************************************************
call cpu_time(cpu1)
call get_command_argument(1,filename)
call get_data(prop, filename)
call allocate_propeller(prop)
call propeller_points(prop)
prop%G = 100._wp
call propeller_velocities(prop)

call calc_gammas(prop)

call calc_dF(prop)
write(*,*) 
write(*,*) 'Force'
write(*,*) prop%Force_Total

call calc_dM(prop)
write(*,*) 
write(*,*) 'Moment'
write(*,*) prop%Moment_Total
write(*,*) 
write(*,*) 'Advance Ratio'
write(*,*) prop%J


call cpu_time(cpu2)
write(*,*)
write(*,*) 'Computation time ',cpu2-cpu1, ' sec'
write(*,*) 'Ran helix function ',2*(prop%nblades*prop%ncontrolpoint)**2, ' times'
!***************************************************************************
!call display_graph(prop)
!call lift_dist(prop)
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
!subroutine display_graph(prop)
!use points_mod
!use dislin
!implicit none
!type(propeller),intent(in) :: prop
!! setup graph
!call metafl('xwin')
!!call metafl('pdf')
!call window(2000,10,1333,1000)
!call page(4000,3000)
!!call scrmod('reverse')
!call disini
!!call psfont('Times-Roman')
!call texmod('on')

!call axslen(1500,1500)
!call axspos(500,1750)
!call name('Y','y')
!call name('X','x')
!call labtyp('vert','x')

!call incmrk(-1)

!call graf(-1.1*real(prop%r_prop), 1.1*real(prop%r_prop), -1.1*real(prop%r_prop), real(prop%r_prop)/5. &
!		, -1.1*real(prop%r_prop), 1.1*real(prop%r_prop), -1.1*real(prop%r_prop), real(prop%r_prop)/5.)
!call color('half')
!call grid(1,1)
!call color('fore')

!call marker(2)
!call curve(real(prop%node(:,1)%x),real(prop%node(:,1)%y),prop%ncontrolpoint*prop%nblades)
!call marker(2)
!call curve(real(prop%node(:,2)%x),real(prop%node(:,2)%y),prop%ncontrolpoint*prop%nblades)

!call marker(0)
!call curve(real(prop%cp%x),real(prop%cp%y),prop%ncontrolpoint*prop%nblades)

!call disfin

!end subroutine display_graph
!!***************************************************************************
!subroutine lift_dist(prop)
!use points_mod
!use dislin
!implicit none
!type(propeller),intent(in) :: prop
!real :: max_thrust

!max_thrust = real(maxval(abs(prop%dF%z)))

!! setup graph
!!call metafl('xwin')
!call metafl('pdf')
!call window(2000,10,1333,1000)
!call page(4000,3000)
!!call scrmod('reverse')
!call disini
!call psfont('Times-Roman')
!call texmod('on')

!call axslen(1500,1500)
!call axspos(500,1750)
!call name('Force, lbf','y')
!call name('Radial position on blade, ft','x')
!call labtyp('vert','x')

!call graf(0., 1.1*real(prop%r_prop), 0., real(prop%r_prop)/10. &
!		, -.1*max_thrust, 1.1*max_thrust, -.1*max_thrust, max_thrust/10.)

!call color('half')
!call grid(1,1)
!call color('fore')

!call curve(real(prop%cp(1:prop%ncontrolpoint)%x),real(abs(prop%dF(1:prop%ncontrolpoint)%z)),&
!			prop%ncontrolpoint)
!call color('blue')
!call curve(real(prop%cp(1:prop%ncontrolpoint)%x),real(abs(prop%dF(1:prop%ncontrolpoint)%y)),&
!			prop%ncontrolpoint)
!call color('red')
!call curve(real(prop%cp(1:prop%ncontrolpoint)%x),real(abs(prop%dF(1:prop%ncontrolpoint)%x)),&
!			prop%ncontrolpoint)

!call disfin

!end subroutine lift_dist
!***************************************************************************
end program propeller_lifting_line



