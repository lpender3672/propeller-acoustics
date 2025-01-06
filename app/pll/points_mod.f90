! ZachMontgomery 2018
! A Propeller Model Based on a Modern Numerical Lifting-LineA Propeller Model Based on a Modern Numerical Lifting-Line
! Algorithm with an Iterative Semi-Free Wake SolverAlgorithm with an Iterative Semi-Free Wake Solver
! https://github.com/ZachMontgomery/PropellerLiftingLineMethod/tree/master
module points_mod
implicit none
!***************************************************************************
integer, parameter :: sp = selected_real_kind(p=6)
integer, parameter :: wp = selected_real_kind(p=16)
real(wp), parameter :: pi = 3.14159265358979323846264338327950288419716939937510582_wp
!***************************************************************************
type point
	real(wp) x
	real(wp) y
	real(wp) z
end type point

type vector
	real(wp) mag
	real(wp) x
	real(wp) y
	real(wp) z
end type vector
!***************************************************************************
interface operator (+)
	procedure add_points
	procedure add_vectors
end interface operator (+)

interface operator (-)
	procedure sub_points
	procedure sub_vectors
end interface operator (-)

interface operator (*)
	procedure scalar_times_vector
	procedure vector_times_scalar
end interface operator (*)

interface operator (/)
	procedure vector_over_scalar
end interface operator (/)

interface operator (.vec.)
	procedure create_vector
end interface operator (.vec.)

interface operator (.dot.)
	procedure dot_prod
end interface operator (.dot.)

interface operator (.cross.)
	procedure cross_product
end interface operator (.cross.)
!***************************************************************************
contains
!***************************************************************************
function add_points(p1,p2)
type(point), intent(in) :: p1, p2
type(point) :: add_points
add_points%x = p1%x + p2%x
add_points%y = p1%y + p2%y
add_points%z = p1%z + p2%z
end function add_points

function sub_points(p1,p2)
type(point), intent(in) :: p1, p2
type(point) :: sub_points
sub_points%x = p1%x - p2%x
sub_points%y = p1%y - p2%y
sub_points%z = p1%z - p2%z
end function sub_points

function add_vectors(v1,v2)
type(vector), intent(in) :: v1, v2
type(vector) add_vectors
add_vectors%x = v1%x + v2%x
add_vectors%y = v1%y + v2%y
add_vectors%z = v1%z + v2%z
add_vectors%mag = sqrt(add_vectors%x**2+add_vectors%y**2+add_vectors%z**2)
end function add_vectors

function sub_vectors(v1,v2)
type(vector), intent(in) :: v1, v2
type(vector) sub_vectors
sub_vectors%x = v1%x - v2%x
sub_vectors%y = v1%y - v2%y
sub_vectors%z = v1%z - v2%z
sub_vectors%mag = sqrt(sub_vectors%x**2+sub_vectors%y**2+sub_vectors%z**2)
end function sub_vectors

function scalar_times_vector(s,v)
real(wp), intent(in) :: s
type(vector), intent(in) :: v
type(vector) :: scalar_times_vector
scalar_times_vector%mag = s * v%mag
scalar_times_vector%x = s * v%x
scalar_times_vector%y = s * v%y
scalar_times_vector%z = s * v%z
end function scalar_times_vector

function vector_times_scalar(v,s)
real(wp), intent(in) :: s
type(vector), intent(in) :: v
type(vector) :: vector_times_scalar
vector_times_scalar = s * v
end function vector_times_scalar

function vector_over_scalar(v,s)
real(wp), intent(in) :: s
type(vector), intent(in) :: v
type(vector) :: vector_over_scalar
real(wp) :: sinv
sinv = 1._wp / s
vector_over_scalar = sinv * v
end function vector_over_scalar

function create_vector(p1,p2)
type(point), intent(in) :: p1, p2
type(vector) :: create_vector
create_vector%x = p2%x - p1%x
create_vector%y = p2%y - p1%y
create_vector%z = p2%z - p1%z
create_vector%mag = sqrt(create_vector%x**2 + create_vector%y**2 + &
						create_vector%z**2)
end function create_vector

function dot_prod(v1,v2)
type(vector), intent(in) :: v1, v2
real(wp) :: dot_prod
dot_prod = v1%x * v2%x + v1%y * v2%y + v1%z * v2%z
end function dot_prod

function cross_product(v1,v2)
type(vector), intent(in) :: v1, v2
type(vector) :: cross_product
cross_product%x = v1%y * v2%z - v1%z * v2%y
cross_product%y = v1%z * v2%x - v1%x * v2%z
cross_product%z = v1%x * v2%y - v1%y * v2%x
cross_product%mag = sqrt(cross_product%x**2 + cross_product%y**2 + &
						cross_product%z**2)
end function cross_product
!***************************************************************************
end module points_mod
!***************************************************************************
