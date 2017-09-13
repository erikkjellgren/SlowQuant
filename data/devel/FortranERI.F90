
! Connect runERI to F2PY
MODULE LOL

contains

SUBROUTINE runERI(basisidx, basisfloat, basisint, max_angular_moment, E1arr, E2arr, E3arr, ERI)
    IMPLICIT NONE
    ! INPUTS
    INTEGER, INTENT(in) :: max_angular_moment
    INTEGER, DIMENSION(:,:), INTENT(in) :: basisidx, basisint
    REAL(8), DIMENSION(:,:), INTENT(in) :: basisfloat
    REAL(8), DIMENSION(:,:,:,:,:), INTENT(in) :: E1arr, E2arr, E3arr
    
    ! OUTPUTS
    REAL(8), DIMENSION(size(basisidx,1),size(basisidx,1),size(basisidx,1),size(basisidx,1)), INTENT(out) :: ERI
    
    ! INTERNAL
    INTEGER :: i, j, k, l, mu, nu, lam, sig, munu, lamsig, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, ALSTAT
    REAL(8) :: a, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, Px, Py, Pz, Qx, Qy, Qz, p, q, alpha, Normalization1, &
    &          Normalization2, Normalization3, Normalization4, c1, c2, c3, c4, calc, outputvalue
    REAL(8), DIMENSION(:), ALLOCATABLE :: E1, E2, E3, E4, E5, E6
    REAL(8), DIMENSION(:,:,:), ALLOCATABLE :: R1buffer
    REAL(8), DIMENSION(:,:,:,:), ALLOCATABLE  :: Rbuffer
    
    ! ALLOCATE INTERNALS
    ALLOCATE(E1(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E2(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E3(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E4(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E5(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(E6(max_angular_moment*2+1), STAT=ALSTAT)
    ALLOCATE(R1buffer(4*max_angular_moment+1,4*max_angular_moment+1,4*max_angular_moment+1), STAT=ALSTAT)
    ALLOCATE(Rbuffer(4*max_angular_moment+1,4*max_angular_moment+1,4*max_angular_moment+1,12*max_angular_moment+1), STAT=ALSTAT)
    !if(ALSTAT /= 0) STOP "***NEED MORE PYLONS***" 
    
	ERI = 0.0d0
    open (unit = 1, file = "outputF.txt")  

    munu = 0
    lamsig = 0
    ! Loop over basisfunctions
    DO mu = 0, size(basisidx,1)-1
        DO nu = mu, size(basisidx,1)-1
            munu = mu*(mu+1)/2+nu
            DO lam = 0, size(basisidx,1)-1
                DO sig = lam, size(basisidx,1)-1
                    lamsig = lam*(lam+1)/2+sig
                    IF (munu >= lamsig) THEN
                        calc = 0.0d0
                        ! Loop over primitives
                        DO i = basisidx(mu+1,2), basisidx(mu+1,2)+basisidx(mu+1,1)-1
                            Normalization1 = basisfloat(i+1,1)
                            a  = basisfloat(i+1,2)
                            c1 = basisfloat(i+1,3)
                            Ax = basisfloat(i+1,4)
                            Ay = basisfloat(i+1,5)
                            Az = basisfloat(i+1,6)
                            l1 =   basisint(i+1,1)
                            m1 =   basisint(i+1,2)
                            n1 =   basisint(i+1,3)
                            DO j = basisidx(nu+1,2), basisidx(nu+1,2)+basisidx(nu+1,1)-1
                                Normalization2 = basisfloat(j+1,1)
                                b  = basisfloat(j+1,2)
                                c2 = basisfloat(j+1,3)
                                Bx = basisfloat(j+1,4)
                                By = basisfloat(j+1,5)
                                Bz = basisfloat(j+1,6)
                                l2 =   basisint(j+1,1)
                                m2 =   basisint(j+1,2)
                                n2 =   basisint(j+1,3)
                                p   = a+b
                                Px  = (a*Ax+b*Bx)/p
                                Py  = (a*Ay+b*By)/p
                                Pz  = (a*Az+b*Bz)/p
                                E1 = E1arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                E2 = E2arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                E3 = E3arr(mu+1,nu+1,i-basisidx(mu+1,2)+1,j-basisidx(nu+1,2)+1,:)
                                DO k = basisidx(lam+1,2), basisidx(lam+1,2)+basisidx(lam+1,1)-1
                                    Normalization3 = basisfloat(k+1,1)
                                    c  = basisfloat(k+1,2)
                                    c3 = basisfloat(k+1,3)
                                    Cx = basisfloat(k+1,4)
                                    Cy = basisfloat(k+1,5)
                                    Cz = basisfloat(k+1,6)
                                    l3 =   basisint(k+1,1)
                                    m3 =   basisint(k+1,2)
                                    n3 =   basisint(k+1,3)
                                    DO l = basisidx(sig+1,2), basisidx(sig+1,2)+basisidx(sig+1,1)-1
                                        Normalization4 = basisfloat(l+1,1)
                                        d  = basisfloat(l+1,2)
                                        c4 = basisfloat(l+1,3)
                                        Dx = basisfloat(l+1,4)
                                        Dy = basisfloat(l+1,5)
                                        Dz = basisfloat(l+1,6)
                                        l4 =   basisint(l+1,1)
                                        m4 =   basisint(l+1,2)
                                        n4 =   basisint(l+1,3)                                    
                                        q   = c+d
                                        Qx  = (c*Cx+d*Dx)/q
                                        Qy  = (c*Cy+d*Dy)/q
                                        Qz  = (c*Cz+d*Dz)/q
                                        E4 = E1arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        E5 = E2arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        E6 = E3arr(lam+1,sig+1,k-basisidx(lam+1,2)+1,l-basisidx(sig+1,2)+1,:)
                                        alpha = p*q/(p+q)
                                        
                                        call R(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Qx, Qy, Qz, Px,&
                                        &		Py, Pz, alpha, R1buffer, Rbuffer)
                                        call elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4,&
                                        &			Normalization1, Normalization2, Normalization3,&
                                        &			Normalization4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6,&
                                        &			R1buffer, outputvalue)
                                        calc = calc + outputvalue
                                    END DO
                                END DO
                            END DO
                        END DO
                        ERI(mu+1,nu+1,lam+1,sig+1) = calc
                        ERI(nu+1,mu+1,lam+1,sig+1) = calc
                        ERI(mu+1,nu+1,sig+1,lam+1) = calc
                        ERI(nu+1,mu+1,sig+1,lam+1) = calc
                        ERI(lam+1,sig+1,mu+1,nu+1) = calc
                        ERI(sig+1,lam+1,mu+1,nu+1) = calc
                        ERI(lam+1,sig+1,nu+1,mu+1) = calc
                        ERI(sig+1,lam+1,nu+1,mu+1) = calc
                    END IF
                END DO
            END DO
        END DO
    END DO
    close(1)
    !return ERI
END SUBROUTINE runERI
        
SUBROUTINE R(l1l2, m1m2, n1n2, Cx, Cy, Cz, Px, Py, Pz, p, R1, Rbuffer)
    IMPLICIT NONE
    !EXTERNAL Boys_func
    
    ! INPUTS
    INTEGER, INTENT(in) :: l1l2, m1m2, n1n2
    REAL(8) , INTENT(in):: Cx, Cy, Cz, Px, Py, Pz, p
    REAL(8), DIMENSION(:,:,:,:), INTENT(inout) :: Rbuffer
    
    ! OUTPUTS
    REAL(8), DIMENSION(:,:,:), INTENT(out) :: R1
    
    ! INTERNAL
    INTEGER :: t, u, v, n, exclude_from_n
    REAL(8) :: RPC, PCx, PCy, PCz, outputvalue, hg!, Boys_func
    
    PCx = Px-Cx
    PCy = Py-Cy
    PCz = Pz-Cz
    RPC = ((PCx)**2+(PCy)**2+(PCz)**2)**0.5
    DO t = 0, l1l2
        DO u = 0, m1m2
            DO v = 0, n1n2
                ! Check the range of n, to ensure no redundent n are calculated
                IF (t == 0 .AND. u == 0) THEN
                    exclude_from_n = v
                ELSEIF (t == 0) THEN
                    exclude_from_n = n1n2 + u
                ELSE
                    exclude_from_n = n1n2 + m1m2 + t
                END IF
                
                DO n = 0, l1l2+m1m2+n1n2 - exclude_from_n
                    outputvalue = 0.0d0
                    IF (t == 0 .AND. u == 0 .AND. v == 0) THEN
                        call chgm(n+0.5d0,n+1.5d0,-p*RPC*RPC,hg)
                        Rbuffer(t+1,u+1,v+1,n+1) = (-2.0d0*p)**n*hg/(2.0d0*n+1.0d0)
                        !Rbuffer(t+1,u+1,v+1,n+1) = (-2.0d0*p)**n*Boys_func(n,p*RPC*RPC)
                    ELSE
                        IF (t == 0 .AND. u == 0) THEN
                            IF (v > 1) THEN
                                outputvalue = outputvalue + (v-1.0d0)*Rbuffer(t+1,u+1,v+1-2,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCz*Rbuffer(t+1,u+1,v+1-1,n+1+1)  
                        ELSEIF (t == 0) THEN
                            IF (u > 1) THEN
                                outputvalue = outputvalue + (u-1.0d0)*Rbuffer(t+1,u+1-2,v+1,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCy*Rbuffer(t+1,u+1-1,v+1,n+1+1)
                        ELSE
                            IF (t > 1) THEN
                                outputvalue = outputvalue + (t-1.0d0)*Rbuffer(t+1-2,u+1,v+1,n+1+1)
                            END IF
                            outputvalue = outputvalue + PCx*Rbuffer(t+1-1,u+1,v+1,n+1+1)
                        END IF
                        Rbuffer(t+1,u+1,v+1,n+1) = outputvalue
                    END IF
                        
                    IF (n == 0) THEN
                        R1(t+1,u+1,v+1) = Rbuffer(t+1,u+1,v+1,n+1)
                    END IF
                END DO
            END DO
        END DO
    END DO
    !return R1
END SUBROUTINE R


SUBROUTINE elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Normalization1, Normalization2, &
&                        Normalization3, Normalization4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, Rpre, outputvalue)
    IMPLICIT NONE
    ! INPUT
    INTEGER, INTENT(in) :: l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4
    REAL(8), INTENT(in) :: p, q, Normalization1, Normalization2, Normalization3, Normalization4, c1, c2, c3, c4
    REAL(8), DIMENSION(:), INTENT(in) :: E1, E2, E3, E4, E5, E6
    REAL(8), DIMENSION(:,:,:), INTENT(in) :: Rpre
    
    ! OUTPUT
    REAL(8), INTENT(out) :: outputvalue
    
    ! INTERNAL
    INTEGER :: tau, nu, phi, t, u, v
    REAL(8) :: N, factor, pi
    pi = 3.141592653589793238462643383279d0
                        
    N = Normalization1*Normalization2*Normalization3*Normalization4*c1*c2*c3*c4
    outputvalue = 0.0d0
    DO tau = 0, l3+l4
        DO nu = 0, m3+m4
            DO phi = 0, n3+n4
                factor = (-1.0d0)**(tau+nu+phi)
                DO t = 0, l1+l2
                    DO u = 0, m1+m2
                        DO v = 0, n1+n2
                            outputvalue = outputvalue + E1(t+1)*E2(u+1)*E3(v+1)*&
                            &             E4(tau+1)*E5(nu+1)*E6(phi+1)*Rpre(t+tau+1,u+nu+1,v+phi+1)*factor
                        END DO
                    END DO
                END DO
            END DO
        END DO
    END DO
    outputvalue = outputvalue*2.0d0*pi**2.5/(p*q*(p+q)**0.5)*N
END SUBROUTINE elelrep


subroutine chgm ( a, b, x, hg )

!*****************************************************************************80
!
!! CHGM computes the confluent hypergeometric function M(a,b,x).
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However, 
!    they give permission to incorporate this routine into a user program 
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    27 July 2012
!
!  Author:
!
!    Shanjie Zhang, Jianming Jin
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) A, B, parameters.
!
!    Input, real ( kind = 8 ) X, the argument.
!
!    Output, real ( kind = 8 ) HG, the value of M(a,b,x).
!
  implicit none

  real ( kind = 8 ) a
  real ( kind = 8 ) a0
  real ( kind = 8 ) a1
  real ( kind = 8 ) aa
  real ( kind = 8 ) b
  real ( kind = 8 ) hg
  real ( kind = 8 ) hg1
  real ( kind = 8 ) hg2
  integer ( kind = 4 ) i
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k
  integer ( kind = 4 ) la
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  integer ( kind = 4 ) nl
  real ( kind = 8 ) pi
  real ( kind = 8 ) r
  real ( kind = 8 ) r1
  real ( kind = 8 ) r2
  real ( kind = 8 ) rg
  real ( kind = 8 ) sum1
  real ( kind = 8 ) sum2
  real ( kind = 8 ) ta
  real ( kind = 8 ) tb
  real ( kind = 8 ) tba
  real ( kind = 8 ) x
  real ( kind = 8 ) x0
  real ( kind = 8 ) xg
  real ( kind = 8 ) y0
  real ( kind = 8 ) y1

  pi = 3.141592653589793D+00
  a0 = a
  a1 = a
  x0 = x
  hg = 0.0D+00

  if ( b == 0.0D+00 .or. b == - abs ( int ( b ) ) ) then
    hg = 1.0D+300
  else if ( a == 0.0D+00 .or. x == 0.0D+00 ) then
    hg = 1.0D+00
  else if ( a == -1.0D+00 ) then
    hg = 1.0D+00 - x / b
  else if ( a == b ) then
    hg = exp ( x )
  else if ( a - b == 1.0D+00 ) then
    hg = ( 1.0D+00 + x / b ) * exp ( x )
  else if ( a == 1.0D+00 .and. b == 2.0D+00 ) then
    hg = ( exp ( x ) - 1.0D+00 ) / x
  else if ( a == int ( a ) .and. a < 0.0D+00 ) then
    m = int ( - a )
    r = 1.0D+00
    hg = 1.0D+00
    do k = 1, m
      r = r * ( a + k - 1.0D+00 ) / k / ( b + k - 1.0D+00 ) * x
      hg = hg + r
    end do
  end if

  if ( hg /= 0.0D+00 ) then
    return
  end if

  if ( x < 0.0D+00 ) then
    a = b - a
    a0 = a
    x = abs ( x )
  end if

  if ( a < 2.0D+00 ) then
    nl = 0
  end if

  if ( 2.0D+00 <= a ) then
    nl = 1
    la = int ( a )
    a = a - la - 1.0D+00
  end if

  do n = 0, nl

    if ( 2.0D+00 <= a0 ) then
      a = a + 1.0D+00
    end if

    if ( x <= 30.0D+00 + abs ( b ) .or. a < 0.0D+00 ) then

      hg = 1.0D+00
      rg = 1.0D+00
      do j = 1, 500
        rg = rg * ( a + j - 1.0D+00 ) &
          / ( j * ( b + j - 1.0D+00 ) ) * x
        hg = hg + rg
        if ( abs ( rg / hg ) < 1.0D-15 ) then
          exit
        end if
      end do

    else

      call gamma ( a, ta )
      call gamma ( b, tb )
      xg = b - a
      call gamma ( xg, tba )
      sum1 = 1.0D+00
      sum2 = 1.0D+00
      r1 = 1.0D+00
      r2 = 1.0D+00
      do i = 1, 8
        r1 = - r1 * ( a + i - 1.0D+00 ) * ( a - b + i ) / ( x * i )
        r2 = - r2 * ( b - a + i - 1.0D+00 ) * ( a - i ) / ( x * i )
        sum1 = sum1 + r1
        sum2 = sum2 + r2
      end do
      hg1 = tb / tba * x ** ( - a ) * cos ( pi * a ) * sum1
      hg2 = tb / ta * exp ( x ) * x ** ( a - b ) * sum2
      hg = hg1 + hg2

    end if

    if ( n == 0 ) then
      y0 = hg
    else if ( n == 1 ) then
      y1 = hg
    end if

  end do

  if ( 2.0D+00 <= a0 ) then
    do i = 1, la - 1
      hg = ( ( 2.0D+00 * a - b + x ) * y1 + ( b - a ) * y0 ) / a
      y0 = y1
      y1 = hg
      a = a + 1.0D+00
    end do
  end if

  if ( x0 < 0.0D+00 ) then
    hg = hg * exp ( x0 )
  end if

  a = a1
  x = x0

  return
end


subroutine gamma ( x, ga )

!*****************************************************************************80
!
!! GAMMA evaluates the Gamma function.
!
!  Licensing:
!
!    The original FORTRAN77 version of this routine is copyrighted by 
!    Shanjie Zhang and Jianming Jin.  However, they give permission to 
!    incorporate this routine into a user program that the copyright 
!    is acknowledged.
!
!  Modified:
!
!    08 September 2007
!
!  Author:
!
!    Original FORTRAN77 version by Shanjie Zhang, Jianming Jin.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument.
!    X must not be 0, or any negative integer.
!
!    Output, real ( kind = 8 ) GA, the value of the Gamma function.
!
  implicit none

  real ( kind = 8 ), dimension ( 26 ) :: g = (/ &
    1.0D+00, &
    0.5772156649015329D+00, &
   -0.6558780715202538D+00, &
   -0.420026350340952D-01, &
    0.1665386113822915D+00, &
   -0.421977345555443D-01, &
   -0.96219715278770D-02, &
    0.72189432466630D-02, &
   -0.11651675918591D-02, &
   -0.2152416741149D-03, &
    0.1280502823882D-03, & 
   -0.201348547807D-04, &
   -0.12504934821D-05, &
    0.11330272320D-05, &
   -0.2056338417D-06, & 
    0.61160950D-08, &
    0.50020075D-08, &
   -0.11812746D-08, &
    0.1043427D-09, & 
    0.77823D-11, &
   -0.36968D-11, &
    0.51D-12, &
   -0.206D-13, &
   -0.54D-14, &
    0.14D-14, &
    0.1D-15 /)
  real ( kind = 8 ) ga
  real ( kind = 8 ) gr
  integer ( kind = 4 ) k
  integer ( kind = 4 ) m
  integer ( kind = 4 ) m1
  real ( kind = 8 ), parameter :: pi = 3.141592653589793D+00
  real ( kind = 8 ) r
  real ( kind = 8 ) x
  real ( kind = 8 ) z

  if ( x == aint ( x ) ) then

    if ( 0.0D+00 < x ) then
      ga = 1.0D+00
      m1 = int ( x ) - 1
      do k = 2, m1
        ga = ga * k
      end do
    else
      ga = 1.0D+300
    end if

  else

    if ( 1.0D+00 < abs ( x ) ) then
      z = abs ( x )
      m = int ( z )
      r = 1.0D+00
      do k = 1, m
        r = r * ( z - real ( k, kind = 8 ) )
      end do
      z = z - real ( m, kind = 8 )
    else
      z = x
    end if

    gr = g(26)
    do k = 25, 1, -1
      gr = gr * z + g(k)
    end do

    ga = 1.0D+00 / ( gr * z )

    if ( 1.0D+00 < abs ( x ) ) then
      ga = ga * r
      if ( x < 0.0D+00 ) then
        ga = - pi / ( x* ga * sin ( pi * x ) )
      end if
    end if

  end if

  return
end


END MODULE LOL


