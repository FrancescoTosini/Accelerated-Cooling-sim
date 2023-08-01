!  
!  Final application of the Course "Parallel Computing using MPI and OpenMP"
!  Serial version 1.0
!
!  This program is meant to be used by course participants for demonstrating
!  the abilities acquired in optimizing and parallelising programs.
!
!  The techniques learnt at the course should be extensively used in order to
!  improve the program response times as much as possible, while gaining
!  the same or very closed results, i.e. the series of final produced images.
!
!  The code implemented herewith has been written for teaching purposes only,
!  therefore source code, algorithms and produced results must not be 
!  trusted nor used for anything different from their original purpose. 
!  
!  Description of the program:
!  in a squared grid a computed set of points is hit by a disturbing field whose 
!  distribution at an initial time can be  theoretically estimated, i.e. it can
!  be computed. The measured values for only a few points are known. 
!  After the initial "heating" effect, the perturbation evolves with a known law 
!  and results are computed for a few time steps.
!
!  Would you please send comments to assistenze_hpc@cilea.it 
!
!  Program outline:
!  1 - the program starts reading the measured values (InitGrid)
!  2 - the theoretical distribution of initial field values is computed (FieldDistribution)
!  3 - the set of struck points is computed (SensiblePoints)
!  4 - the distribution of values is estimated over the computed set of points (FieldInit)
!  5 - the evolution of the perturbation effects over time is computed (Cooling)
!
MODULE Environment
   IMPLICIT NONE
   INTEGER, PARAMETER :: Xdots=1400, Ydots=1400   ! Plate grid resolution in 2 dimensions
                                       ! May be changed to 1000x1000

   ! Parameters to compute point sensitiveness - values read from input file
   REAL(8):: Sreal, Simag, Rreal, Rimag
   INTEGER :: MaxIters
   ! Evolution time steps
   INTEGER :: TimeSteps
! 
   REAL(8), ALLOCATABLE, DIMENSION(:,:) :: MeasuredValues    ! Values read in input file
   INTEGER :: NumInputValues       ! Number of values read in input file
   REAL(8), ALLOCATABLE, DIMENSION(:,:) :: TheorSlope   ! Theoretical value distribution
   INTEGER :: TSlopeLength   ! TheorSlope grid dimensions
   INTEGER, ALLOCATABLE, DIMENSION(:,:) :: FieldWeight   ! Degree of sensitiveness to perturbing field 
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:) :: FieldCoord   ! X, Y coordinates in field
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:) :: FieldValues   ! X, Y values in field
   
CONTAINS
   !  Main functions
   
   SUBROUTINE InitGrid(InputFile)
      ! Output:
      ! MeasuredValues(:,3) - values read from input file
      ! Initialization of FieldWeight(Xdots,Ydots) and 
      !                  FieldCoord(Xdots,Ydots,2)
         USE  ISO_FORTRAN_ENV
         IMPLICIT NONE
         CHARACTER(*), INTENT(IN) :: InputFile
         
         INTEGER :: st, valrows
         CHARACTER(80) :: filerow, iomsg     
         
         WRITE(*,*) "Initializing grid, reading values from file ",TRIM(InputFile)
         
         OPEN(11,FILE=InputFile,ACTION="READ",STATUS="OLD", &
            & IOSTAT=st, IOMSG=iomsg)
         IF ( st /= 0 ) THEN
              WRITE(*,*) "Error accessing file ",TRIM(InputFile),": iomsg = ",TRIM(iomsg)
              STOP "Error halting"
         END IF
         ! Now read measured values
         NumInputValues = 0
         valrows = 0
         DO 
            READ(11,"(A)",IOSTAT=st,IOMSG=iomsg) filerow
            IF ( st /= 0 ) THEN
               IF ( st == IOSTAT_END ) THEN
                  WRITE(*,*) "Error reading file ",TRIM(InputFile),": unexpected EOF"
                  STOP "Error halting"
               ELSE
                  WRITE(*,*) "Error reading file ",TRIM(InputFile),": iomsg = ",TRIM(iomsg)
                  STOP "Error halting"
               END IF
            END IF
            IF ( filerow(1:1) == "#" ) CYCLE
            BACKSPACE(11)
            IF (  NumInputValues <= 0 ) THEN
               READ(11,*,IOSTAT=st,IOMSG=iomsg) NumInputValues
               IF ( st /= 0 .OR. NumInputValues <= 0 ) THEN
                  WRITE(*,*) "Error reading file ",TRIM(InputFile),": iomsg = ",TRIM(iomsg)
                  WRITE(*,*) "Error reading NumInputValues: ",NumInputValues
                  STOP "Error halting"
               ELSE
                  ALLOCATE(MeasuredValues(NumInputValues,3), STAT=st)
                  IF ( st /= 0 ) THEN
                     WRITE(*,*) "Error allocating MeasuredValues(",NumInputValues,",3)"
                     STOP "Error halting"
                  END IF
              END IF
            ELSE
               valrows = valrows + 1
               IF ( valrows > NumInputValues ) EXIT
               READ(11,*,IOSTAT=st,IOMSG=iomsg) MeasuredValues(valrows,1), &   ! X coord
               &               MeasuredValues(valrows,2), &    ! Y coord
               &            MeasuredValues(valrows,3)    ! Measured value 
               IF ( st /= 0 ) THEN
                  WRITE(*,*) "Error reading MeasuredValues(",valrows,",*): iomsg = ",TRIM(iomsg)
                  STOP "Error halting"
               END IF
            END IF
         END DO
         ALLOCATE(FieldWeight(Xdots,Ydots), STAT=st)
         IF ( st /= 0 ) THEN
              WRITE(*,*) "Error allocating FieldWeight(",Xdots,",",Ydots,")"
              STOP "Error halting"
         END IF
         FieldWeight = 0
         
         ALLOCATE(FieldCoord(Xdots,Ydots,2), STAT=st)
         IF ( st /= 0 ) THEN
              WRITE(*,*) "Error allocating FieldCoord(",Xdots,",",Ydots,",2)"
              STOP "Error halting"
         END IF
         FieldCoord = 0.0D0
         !
         ! Now read Sreal, Simag, Rreal, Rimag
         Sreal = 0.0D0; Simag = 0.0D0; Rreal = 0.0D0; Rimag = 0.0D0
         DO 
            READ(11,"(A)",IOSTAT=st,IOMSG=iomsg) filerow
            IF ( st /= 0 ) EXIT
            IF ( filerow(1:1) == "#" ) CYCLE
            BACKSPACE(11)
            READ(11,*,IOSTAT=st) Sreal
            IF ( st /= 0) THEN
              WRITE(*,*) "Error reading Sreal: iomsg = ",TRIM(iomsg)
              STOP "Error halting"
            END IF
            READ(11,*,IOSTAT=st) Simag
            IF ( st /= 0) THEN
              WRITE(*,*) "Error reading Simag: iomsg = ",TRIM(iomsg)
              STOP "Error halting"
            END IF
            READ(11,*,IOSTAT=st) Rreal
            IF ( st /= 0) THEN
              WRITE(*,*) "Error reading Rreal: iomsg = ",TRIM(iomsg)
              STOP "Error halting"
            END IF
            READ(11,*,IOSTAT=st) Rimag
            IF ( st /= 0) THEN
              WRITE(*,*) "Error reading Rimag: iomsg = ",TRIM(iomsg)
              STOP "Error halting"
            END IF
            
            EXIT           
         END DO
         !
         ! Now read MaxIters
         MaxIters = 0
         DO 
            READ(11,"(A)",IOSTAT=st,IOMSG=iomsg) filerow
            IF ( st /= 0) THEN
               WRITE(*,*) "Error reading file ",TRIM(InputFile),": iomsg = ",TRIM(iomsg)
               STOP "Error halting"
            END IF
            IF ( filerow(1:1) == "#" ) CYCLE
            BACKSPACE(11)
            READ(11,*,IOSTAT=st) MaxIters
            IF ( st /= 0) THEN
               WRITE(*,*) "Error reading MaxIters: iomsg = ",TRIM(iomsg)
               STOP "Error halting"
            END IF
            EXIT           
         END DO
         !
         ! Now read TimeSteps
         TimeSteps = 0
         DO 
            READ(11,"(A)",IOSTAT=st) filerow
            IF ( st /= 0) THEN
               WRITE(*,*) "Error reading file ",TRIM(InputFile),": iomsg = ",TRIM(iomsg)
               STOP "Error halting"
            END IF
            IF ( filerow(1:1) == "#" ) CYCLE
            BACKSPACE(11)
            READ(11,*,IOSTAT=st) TimeSteps
            IF ( st /= 0) THEN
               WRITE(*,*) "Error reading TimeSteps: iomsg = ",TRIM(iomsg)
               STOP "Error halting"
            END IF
            EXIT           
         END DO
   
         RETURN
      END SUBROUTINE InitGrid
   
      SUBROUTINE FieldInit()
         IMPLICIT NONE
   !     Initialize field values in the grid. Values are computed on the basis
   !     of the measured values read in subroutine InitGrid and the gross grid 
   !     values computed in subroutine FieldDistribution. Moreover sensitiveness
   !     to field effects as computed in subroutine SensiblePoints are taken into
   !     account.
   ! 
   !     Input:
   !       MeasuredValues(:,3)
   !       FieldWeight(Xdots,Ydots)
   !     Output:
   !       FieldValues(Xdots,Ydots,2)
         INTEGER :: rv, st
         REAL(8) :: xc, yc, ev, sv, sd, DiscrValue
         REAL(8), DIMENSION(:), ALLOCATABLE :: DiffValues
      
         WRITE(*,*) "Initializing entity of field effects ..."
         
         ALLOCATE(FieldValues(Xdots,Ydots,2),STAT=st)
         IF ( st /= 0 ) THEN
             WRITE(*,*) "Error allocating FieldValues(",Xdots,",",Ydots,",2)"
             STOP "Error halting"
         END IF
         ALLOCATE(DiffValues(NumInputValues),STAT=st)
         IF ( st /= 0 ) THEN
             WRITE(*,*) "Error allocating DiffValues(",NumInputValues,")"
             STOP "Error halting"
         END IF
         
   !     Compute discrepancy between Measured and Theoretical value
         DiffValues = 0.0D0
         DiscrValue = 0.0D0
         DO rv = 1, NumInputValues
            xc = MeasuredValues(rv,1)
            yc = MeasuredValues(rv,2)
            ! TheorSlope is computed on the basis of a coarser grid, so look for
            ! the best values near xc, yc coordinates
            sv = NearestValue(xc,yc,TSlopeLength,TheorSlope)
            ev = MeasuredValues(rv,3)
            DiffValues(rv) = ev - sv
            DiscrValue = DiscrValue + ev - sv
         END DO
         DiscrValue = DiscrValue / DBLE(NumInputValues)
         ! Compute statistics and best approximated value
         sd = 0.0D0   ! Standard Deviation
         DO rv = 1, NumInputValues
            sd = sd + ( DiffValues(rv) - DiscrValue )*( DiffValues(rv) - DiscrValue )
         END DO
         sd = SQRT(sd / DBLE(NumInputValues))
         ! Print statistics
         WRITE(*,*) 
         WRITE(*,"(A,I4,2(1X,F8.3))") "... Number of Points, Mean value, Standard deviation = ", &
         &         NumInputValues, DiscrValue, sd
         ! Compute FieldValues stage 1
         print*," DiscrValue = ",DiscrValue
         CALL FieldPoints(DiscrValue)
   
         RETURN
      END SUBROUTINE FieldInit
      
      
      SUBROUTINE FieldDistribution
      !  Compute theoretical value distribution of the perturbing field  
      !        on a grid Mm1 x Nm1 where M = SQRT(DBLE(Xdots)) and 
      !        N = SQRT(DBLE(Ydots)); Mm1 = M - 1 and Nm1 = N - 1
      ! Output:
      !    TheorSlope(TSlopeLength,3) - theoretical field distribution function
      IMPLICIT NONE
      !
      ! Let's consider the 2D-square x0 < X < x1, y0 < Y < y1
      ! Suppose we divide the square in N-1 x N-1 portions 
      ! Given a function F(x,y) suppose we know that in every point:
      !   d2F(x,x)/dx2 + d2F(x,y)/dy2 = x+y
      ! Let's define a system of equation so that we can compute the value
      ! of F(x,y) in every point of the 2D-square
      !
      REAL(8), ALLOCATABLE, DIMENSION(:,:) :: CoeffMatrix   
      REAL(8), ALLOCATABLE, DIMENSION(:) :: B
        
      REAL(8) :: x0, y0, x1, y1
   
      INTEGER :: M, Mm1, N, Nm1, LA, pos      
      INTEGER :: i, j, rc
      
      WRITE(*,*) "Computing theoretical perturbing field ..."
      x0 = Sreal; y0 = Simag; x1 = x0+Rreal; y1 = y0+Rimag
      ! How many intervals? It should be safe to use SQRT(Xdots)
      M = int(SQRT(DBLE(Xdots)))
      N = int(SQRT(DBLE(Ydots)))
      
      Nm1 = N-1  ! Grid points minus boundary
      Mm1 = M-1
      LA = Mm1*Nm1 ! unknown points
      TSlopeLength = LA
      ALLOCATE(CoeffMatrix(LA,LA), TheorSlope(TSlopeLength,3),STAT=rc)
      IF ( rc /= 0 ) THEN
            WRITE(*,*) "Error allocating coefficient matrix"
            STOP "Error halting"
      END IF   
   
      ALLOCATE(B(LA), STAT=rc)   ! Unknown and RHS vectors
      IF ( rc /= 0 ) THEN
            WRITE(*,*) "Error allocating RHS vector"
            STOP "Error halting"
      END IF
      
      CALL GridDef(x0,x1,y0,y1,N,TheorSlope)
      
      CALL EqsDef(x0,x1,y0,y1,N,LA,CoeffMatrix,B,TheorSlope,rc)
   
      CALL LinEquSolve(CoeffMatrix,LA,B,rc)
      IF ( rc /= 0 ) THEN
           STOP "Error halting"
      END IF
      
      TheorSlope(:,3) = B(:)
      
      RETURN
   CONTAINS
      FUNCTION Solution(x,y) RESULT (f)
         IMPLICIT NONE
         REAL(8), INTENT(IN) :: x, y   ! Point coordinates
         REAL(8) :: f
         
         f = (x*x*x+y*y*y)/6.0D0
         
         RETURN
      END FUNCTION Solution
            
      SUBROUTINE GridDef(x0,x1,y0,y1,N,Pts)
         IMPLICIT NONE
         REAL(8), INTENT(IN) :: x0,x1,y0,y1  ! Define grid
         INTEGER, INTENT(IN) :: N
         REAL(8), DIMENSION((N-1)*(N-1),2), INTENT(OUT) :: Pts   ! inner grid point Coordinates
   !
         REAL(8) :: x, y, dx, dy
         INTEGER :: np
         
         dx = (x1-x0)/DBLE(N); dy = (y1-y0)/DBLE(N)
         np = 0
         DO i = 1, Mm1
            DO j = 1, Nm1
               np = np + 1
               IF ( np > Mm1*Nm1 ) THEN
                   WRITE(*,*) " Error: NP = ",np," > M*N = ",Mm1*Nm1
                   STOP "Error halting"
               END IF
               x = x0 + dx * DBLE(i)
               y = y0 + dy * DBLE(j)
               Pts(np,1) = x
               Pts(np,2) = y
            END DO
         END DO

         RETURN
      END SUBROUTINE GridDef
   
      SUBROUTINE EqsDef(x0,x1,y0,y1,N,LA,A,Rhs,Pts,Info)
         IMPLICIT NONE
         REAL(8), INTENT(IN) :: x0,x1,y0,y1  ! Define grid
         INTEGER, INTENT(IN) :: N, LA   ! Extent of grid side and number of grid points
         REAL(8), DIMENSION(LA), INTENT(OUT) :: Rhs   ! Linear equation RHS
         REAL(8), DIMENSION(LA,LA), INTENT(OUT) :: A   ! Linear equation matrix
         REAL(8), DIMENSION(TSlopeLength,3), INTENT(IN) :: Pts
         INTEGER, INTENT(OUT) :: Info   ! Info < 0 if errors occur 
         !
         REAL(8) :: x, y, Eps, dx, dy
         INTEGER :: np
         !
         !  Define A matrix and RHS
         !
         dx = (x1-x0)/DBLE(N); dy = (y1-y0)/DBLE(N)
         
         A = 0.0D0; Rhs = 0.0D0
         Info = 0
         DO np = 1, LA
            x = Pts(np,1)
            y = Pts(np,2)
            A(np,np) = -4.0D0
            Rhs(np) = (x + y)*dx*dy
            ! define Eps, 1/10. part of grid cell
            Eps = (dx+dy)/20.0D0
            ! where is P(x-dx,y) ?
            IF ( ABS((x-dx)-x0) < Eps ) THEN ! boundary point
                  Rhs(np) = Rhs(np) - Solution(x0,y) 
            ELSE  ! inner point
                  ! Find pos = position of P(x-dx,y)
                  pos = np - Nm1
                  IF ( ABS(Pts(pos,1)-(x-dx)) > Eps ) THEN
                        WRITE(*,*) " Error x-dx: ",ABS(Pts(pos,1)-(x-dx))
                  END IF
                  A(np,pos) = 1.0D0
            END IF
            ! where is P(x+dx,y) ? 
            IF ( ABS((x+dx)-x1) < Eps ) THEN  ! boundary point
                  Rhs(np) = Rhs(np) - Solution(x1,y)
            ELSE  ! inner point
                  ! Find pos = position of P(x+dx,y)
                  pos = np + Nm1
                  IF ( ABS(Pts(pos,1)-(x+dx)) > Eps ) THEN
                        WRITE(*,*) " Error x+dx: ",ABS(Pts(pos,1)-(x+dx))
                  END IF
                  A(np,pos) = 1.0D0
            END IF
            ! where is P(x,y-dy) ? 
            IF ( ABS((y-dy)-y0) < Eps ) THEN  ! boundary point
                  Rhs(np) = Rhs(np) - Solution(x,y0)
            ELSE  ! inner point
                  ! Find pos = position of P(x,y-dy)
                  pos = np - 1
                  IF ( ABS(Pts(pos,2)-(y-dy)) > Eps ) THEN
                        WRITE(*,*) " Error y-dy: ",ABS(Pts(pos,2)-(y-dy))
                         Info = -1
                  END IF
                  A(np,pos) = 1.0D0
            END IF
            ! where is P(x,y+dy) ? 
            IF ( ABS((y+dy)-y1) < Eps ) THEN  ! boundary point
                  Rhs(np) = Rhs(np) - Solution(x,y1)
            ELSE  ! inner point
                  ! Find pos = position of P(x,y+dy)
                  pos = np + 1
                  IF ( ABS(Pts(pos,2)-(y+dy)) > Eps ) THEN
                        WRITE(*,*) " Error y+dy: ",ABS(Pts(pos,2)-(y+dy))
                         Info = -1
                  END IF
                  A(np,pos) = 1.0D0
            END IF
         END DO
         
         RETURN
      END SUBROUTINE EqsDef   
      
   END SUBROUTINE FieldDistribution
   
   
   SUBROUTINE LinEquSolve(a,n,b,errcode)
   !  Gauss-Jordan elimination algorithm
   !  Based on a code from "Numerical recipes in Fortran 77"
   !
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: n
      REAL(8), DIMENSION(n,n), INTENT(IN OUT) :: a    ! Coefficient matrix
      REAL(8), DIMENSION(n), INTENT(IN OUT) :: b      ! RHS (in) and solution (out)
      
      INTEGER :: i,j,k,l,icol,irow
      INTEGER, ALLOCATABLE, DIMENSION(:) :: indcol, indrow, ipiv 
      INTEGER :: rc, errcode
      REAL(8) :: bigger,temp 
      
      errcode = 0
      ALLOCATE(indcol(n), STAT=rc)
      IF ( rc /= 0 ) THEN
            WRITE(*,*) "In LinEquSolve error allocating INDCOL(",n,")."
            errcode = -1
            RETURN
      END IF
      
      ALLOCATE(indrow(n), STAT=rc)
      IF ( rc /= 0 ) THEN
            WRITE(*,*) "In LinEquSolve error allocating INDROW(",n,")."
            errcode = -1
            RETURN
      END IF
      
      ALLOCATE(ipiv(n), STAT=rc)
      IF ( rc /= 0 ) THEN
            WRITE(*,*) "In LinEquSolve error allocating IPIV(",n,")."
            errcode = -1
            RETURN
      END IF
      
      ipiv = 0
      DO i = 1, n 
         bigger = 0.0D0
         DO j = 1, n 
            IF (ipiv(j) /= 1) THEN
              DO k = 1, n
                 IF (ipiv(k) == 0 .AND. bigger <= abs(a(j,k))) THEN
                    bigger = abs(a(j,k))
                    irow = j
                    icol = k
                 ENDIF
              END DO 
           ENDIF
         END DO 
         ipiv(icol) = ipiv(icol) + 1
         IF (irow /= icol) THEN
            DO l = 1,n
               temp = a(irow,l)
               a(irow,l) = a(icol,l)
               a(icol,l) = temp
            END DO 
            temp = b(irow)
            b(irow) = b(icol)
            b(icol) = temp
         ENDIF
         indrow(i) = irow 
         indcol(i) = icol 
         IF (a(icol,icol) == 0.0D0) THEN
           WRITE(*,*) "In LinEquSolve a(",icol,",",icol,"): singular matrix!"
           errcode = -2
           return
         end IF
         temp = 1.0D0/a(icol,icol)
         a(icol,icol) = 1.0D0
         a(icol,:) = a(icol,:) * temp
         b(icol) = b(icol) * temp
         DO l = 1,n 
            IF (l /= icol) THEN 
                temp = a(l,icol)
                a(l,icol) = 0.
                a(l,:) = a(l,:)-a(icol,:) * temp
                b(l) = b(l)-b(icol) * temp
             ENDIF
         END DO 
      END DO  
      DO l = n,1,-1 
         IF (indrow(l) /= indcol(l)) THEN
            DO k = 1,n
               temp = a(k,indrow(l))
               a(k,indrow(l)) = a(k,indcol(l))
               a(k,indcol(l)) = temp
            END DO 
         ENDIF
       END DO 
       RETURN 
   END SUBROUTINE LinEquSolve
   
   SUBROUTINE FieldPoints(Diff)
      IMPLICIT NONE
      REAL(8), INTENT(IN) :: Diff
   
      INTEGER :: ix, iy     
      REAL(8) :: xc, yc, sv 
      REAL(8) :: rmin, rmax   
      
      integer :: stampaY, stampaX
   
      rmax = MAXVAL(FieldWeight)
      rmin = MINVAL(FieldWeight)
      print*,"In FieldPoints rmax, rmin = ",rmax,rmin
      stampaY = 0; stampaX = 0
      DO iy = 1, Ydots
        if ( iy < 4 .or. iy > YDOTS-4 ) then
         stampaY = 1
        else 
         stampaY = 0
        endif
        DO ix = 1, Xdots
           if ( iy < 4 .or. iy > YDOTS-4 ) then
              stampaX = 1
           else 
              stampaX = 0
           endif
            xc = FieldCoord(ix,iy,1)
            yc = FieldCoord(ix,iy,2)
            ! Compute effects of field in every point
            sv = NearestValue(xc,yc,TSlopeLength,TheorSlope)
            FieldValues(ix,iy,1) = 293.16 + 80 * ( Diff + sv ) * &
            &              (FieldWeight(ix,iy) - rmin) / (rmax - rmin)
            if ( stampaX == 1 .and. stampaY == 1 ) THEN
               print*," ix, iy = ",ix,iy
               print*,"xc, yc, sv, FieldWeight(ix,iy), FieldValues(ix,iy,1) = ", &
                  & xc, yc, sv, FieldWeight(ix,iy), FieldValues(ix,iy,1)
            endif
         END DO 
      END DO  
      ! Copy initial status
      FieldValues(:,:,2) = FieldValues(:,:,1)
      RETURN
   END SUBROUTINE FieldPoints
   
   
   SUBROUTINE SensiblePoints(Ir,Ii,Sr,Si,MaxIt)
      !  Compute "heated" points 
      ! Output:
      !    FieldCoord(Xdots,Ydots,2) 
      !    FieldWeight(Xdots,Ydots)
      IMPLICIT NONE
      REAL(8), INTENT(IN) :: Ir,Ii,Sr,Si
      INTEGER, INTENT(IN) :: MaxIt
   
      INTEGER :: ix, iy, iz     
      REAL(8) :: ca, cb, za, zb
      REAL(8) :: rad, zan, zbn 
      REAL(8) :: Xinc, Yinc  
      
      WRITE(*,*) "Computing sensitivity to field effects ..."
      
      Xinc = Sr / DBLE(Xdots)  
      Yinc = Si / DBLE(Ydots)  
   
      FieldWeight = 0
      DO iy = 1, Ydots
        DO ix = 1, Xdots
            ca = Xinc * (ix-1) + Ir 
            cb = Yinc * (iy-1) + Ii  
            FieldCoord(ix,iy,1) = ca 
            FieldCoord(ix,iy,2) = cb 
            rad = abs(ca * SQRT( 1.0D0 + (cb/ca)**2 ))
            zan = 0.0  
            zbn = 0.0  
            DO iz = 1, MaxIt
               IF ( rad > 2.0D0 ) EXIT  
               za = zan
               zb = zbn
               zan = ca + (za-zb)*(za+zb)  
               zbn = 2.0d0 * ( za*zb + cb/2.0d0 )   
               rad = abs(zan * SQRT( 1.0D0 + (zbn/zan)**2 ))
            END DO            
            FieldWeight(ix,iy) = iz
         END DO 
      END DO  
      RETURN
   END SUBROUTINE SensiblePoints
   
   subroutine update(xdots, ydots, u1, u2)
         implicit none
         integer, intent(in) :: xdots, ydots
         real(8), dimension(xdots,ydots), intent(in) :: u1
         real(8), dimension(xdots,ydots), intent(out) :: u2
   
         integer :: ix, iy
         real(8) :: CX, CY, dd, dx, dy, dgx, dgy
        
         dd=0.0000001;
         dx = 1.0/dble(xdots);
         dy = 1.0/dble(ydots);
         dgx = - 2.0 + dx*dx/(2*dd);
         dgy = - 2.0 + dy*dy/(2*dd);
         CX = dd/(dx*dx);
         CY = dd/(dy*dy);
         
         do iy=2, Ydots-1
           do ix=2, Xdots-1
               if ( ix <= 1 .OR. ix >= Xdots .OR. iy <= 1 .OR. iy >= Ydots ) then
                  u2(ix,iy) = u1(ix,iy)
                  cycle
               endif
               u2(ix,iy) = CX * ( u1(ix-1,iy) + u1(ix+1,iy) + dgx*u1(ix,iy) ) &
        &                +  CY * ( u1(ix,iy-1) + u1(ix,iy+1) + dgy*u1(ix,iy) ) 
           end do
         end do
         u2(1,:) = u2(2,:)
         u2(Xdots,:) = u2(Xdots-1,:)
         u2(:,1) = u2(:,2)
         u2(:,Ydots) = u2(:,Ydots-1)      
         
         return
   end subroutine update
   
    subroutine Cooling(steps)
      !  Compute evolution of the effects of the field 
      !  Input/Output:
      !    FieldValues(Xdots,Ydots,2)
         IMPLICIT NONE
   
         INTEGER, INTENT(IN) :: steps
         REAL(8) :: vmin, vmax
         INTEGER :: iz, it
         CHARACTER(80) :: fname
   
         WRITE(*,*) "Computing cooling of field effects ..."
         WRITE(*,*) "... ",STEPS," steps ..."
   !       Print and show initial state 
           fname = "FieldValuesf0000"
           vmin=0.0; vmax=0.0
           call RealVal2ppm(Xdots,Ydots,FieldValues(1,1,1),vmin,vmax,fname)
           call Statistics(Xdots,Ydots,FieldValues(1,1,1),0)
         iz=1
         do it=1, STEPS
   !       Update the value of grid points
           call update(Xdots,Ydots,FieldValues(1,1,iz),FieldValues(1,1,3-iz))
           iz=3-iz
   !       Print and show results 
           WRITE(fname,"(A,I4.4)") "FieldValuesf",it
           if ( mod(it,4) == 0 ) call RealVal2ppm(Xdots,Ydots,FieldValues(1,1,iz),vmin,vmax,fname)
           call Statistics(Xdots,Ydots,FieldValues(1,1,iz),it)
         end do
   
         return
       
   end subroutine cooling

!  Auxiliary functions     

   FUNCTION NearestValue(xc,yc,ld,Values) RESULT(v)
      IMPLICIT NONE
!     look for the best values near xc, yc coordinates
      REAL(8), INTENT(IN) :: xc, yc
      INTEGER, INTENT(IN) :: ld
      REAL(8), DIMENSION(ld,3), INTENT(IN) :: Values
      REAL(8) :: v
!
      REAL(8) :: mnd ! minimum distance
      REAL(8), ALLOCATABLE, DIMENSION(:) :: d ! distance vector
      INTEGER :: np ! number of nearest points
      INTEGER :: i, st
      
      ALLOCATE(d(ld),STAT=st)
      IF ( st /= 0 ) THEN
         WRITE(*,*) "Error in NearestValue: unable to allocate d(",ld,")!"
         STOP "Error halting"
      ENDIF
      mnd = (xc-Values(1,1))*(xc-Values(1,1)) + (yc-Values(1,2))*(yc-Values(1,2))
      ! Compute lowest distance
      DO i = 1, ld
         d(i) = (xc-Values(i,1))*(xc-Values(i,1)) + (yc-Values(i,2))*(yc-Values(i,2))
         IF ( mnd > d(i) ) then
            mnd = d(i)
         END IF
      END DO
      np = 0
      v = 0.0D0
      ! Compute nearest value
      DO i = 1, ld
         IF ( mnd == d(i) ) THEN
            ! add contributed value
            np = np + 1
            v = v + Values(i,3)
         END IF
      END DO
      ! mean value
      v = v / DBLE(np)      
      
      DEALLOCATE(d)
      RETURN
   END FUNCTION NearestValue
   
   SUBROUTINE get_date(time_date,time)
      IMPLICIT NONE
      CHARACTER(*), INTENT(OUT) :: time_date
      REAL(8), INTENT(OUT) :: time
      INTEGER, DIMENSION(8) :: time_data  ! year, month, day, UTC diff, hh, mm, ss, msec
      
      CALL DATE_AND_TIME(VALUES=time_data)
      WRITE(time_date,"(2(I2.2,':'),I2.2,' - ',2(I2.2,'/'),I4.4)") time_data(5), &
               & time_data(6),time_data(7), &
               & time_data(3), time_data(2),time_data(1)
      time = time_data(8)/1000.0D0 + time_data(7) + time_data(6)*60.0D0 + time_data(5)*3600.0D0
   END SUBROUTINE get_date
   
   SUBROUTINE RealVal2ppm(s1, s2, rdata, vmin, vmax, name) 
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: s1, s2
      REAL(8), DIMENSION(s1,s2), INTENT(IN) :: rdata
      REAL(8), INTENT(IN OUT) :: vmin, vmax
      CHARACTER(*), INTENT(IN) :: name
      !
      ! Simple subroutine to dump double data with fixed min & max values 
      !   in a PPM format
      !
   INTEGER ::  i, j, st
   INTEGER, DIMENSION(3,256) :: cm  ! R,G,B, Colour Map 
   INTEGER, PARAMETER :: ouni=33, ColMap=44
   INTEGER ::   vp, vs
   REAL(8) ::   rmin, rmax, value, k
   CHARACTER(80) ::  fname, jname, command
   !
   !  Define color map: 256 colours
   !
   OPEN(ColMap, FILE="ColorMap.txt", STATUS="OLD",IOSTAT=st)
   if ( st /= 0 ) THEN
           WRITE(*,'("Error read opening file ColorMap.txt")')
           STOP "Error halting"
   ENDIF
   DO i = 1, 256
      READ(ColMap,*, IOSTAT=st) cm(:,i) 
      IF ( st /= 0 ) THEN
         WRITE(*,'("Error reading colour map at line ",I3,": r, g, b =")',ADVANCE="NO") i
         WRITE(*,'(3(1X,I3))') cm(:,i)
         STOP "Error halting"
      ENDIF
   ENDDO
   CLOSE(ColMap)

   !
   !   Write on unit ouni with  PPM format
   !
   fname = TRIM(name)//".ppm"
   OPEN(ouni, FILE=TRIM(fname), STATUS="UNKNOWN",IOSTAT=st)
   if ( st /= 0 ) THEN
           WRITE(*,'("!!!! Error write access to file ",A)') TRIM(fname)
           STOP "Error halting"
   ENDIF

   !   Magic code 
   WRITE(ouni,'("P3")')
   !   Dimensions 
   WRITE(ouni,'(2(I6,1X))') s1, s2
   !   Maximum value 
   WRITE(ouni,'("255")')
   !   Values from 0 to 255 
   rmin = MINVAL(rdata); rmax = MAXVAL(rdata)
   if ( (vmin == vmax) .AND. (vmin == 0.0D0) ) THEN
      vmin = rmin; vmax = rmax
   ELSE
      rmin = vmin; rmax = vmax
   ENDIF
   k = 255.0D0 / DBLE(rmax - rmin)
   vs = 0
   DO i = 1, s1
      DO j = 1, s2
         value = rdata(i,j)
         vp =  int(1.0 + ( (value - rmin) * k ))
         vp = max(1,vp); vp = min(256,vp)
         vs = vs + 1
         WRITE(ouni,'(3(1X,I3))', ADVANCE="NO") cm(:,vp)
         if ( vs >= 10 ) THEN
            WRITE(ouni,'(1X)')
            vs = 0
         ENDIF
      ENDDO
      WRITE(ouni,'(1X)')
      vs = 0
   ENDDO
   CLOSE(ouni)
   ! the following instructions require ImageMagick tool: comment out if not
   ! available
   !!jname = TRIM(name)//".jpg"
   !!command = "convert "//TRIM(fname)//" "//TRIM(jname)
   !!CALL system(command)

   END SUBROUTINE RealVal2ppm


   SUBROUTINE Statistics(s1, s2, rdata, step) 
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: s1, s2, step
      REAL(8), DIMENSION(s1,s2), INTENT(IN) :: rdata

      REAL(8) :: mnv, mv, mxv, sd
      INTEGER :: i, j
      !  Compute statistics 
      print*,"In Statistics: rdata(1:3,1:3) = ",rdata(1:3,1:3)
      print*,"In Statistics: rdata(1:3,1+350:3+350) = ",rdata(1:3,1+350:3+350)
      print*,"In Statistics: rdata(1:3,1+700:3+700) = ",rdata(1:3,1+700:3+700)
      print*,"In Statistics: rdata(s1-3:s1,s2-3:s2) = ",rdata(s1-3:s1,s2-3:s2)
      !  Compute statistics 
      !!mv = 0.0;   ! Min, mean, max value
      !!mnv = rdata(1,1)
      mnv = MINVAL(rdata)
      !!mxv = rdata(1,1)
      mxv = MAXVAL(rdata)
      !!DO i = 1, s1
      !!   DO j = 1, s2
      !!      mv = mv + rdata(i,j)
      !!      if ( mnv > rdata(i,j) ) mnv = rdata(i,j)
      !!      if ( mxv < rdata(i,j) ) mxv = rdata(i,j)
      !!   ENDDO
      !!ENDDO
      !!mv = mv / DBLE(s1*s2)
      mv = SUM(rdata) / DBLE(s1*s2)

     sd = 0.0;   ! Standard Deviation
     DO i = 1, s1
         DO j = 1, s2
            sd = sd + ( rdata(i,j) - mv ) * ( rdata(i,j) - mv )
         ENDDO
     ENDDO
     sd = SQRT( sd / DBLE(s1*s2) )
     
     WRITE(*,"('Step ',I4,': min, mean, max, std.dev. =',4(1X,F8.3))") step,mnv,mv,mxv,sd
   END SUBROUTINE Statistics

END MODULE Environment


PROGRAM Cooling_grid
   USE ISO_FORTRAN_ENV
   USE Environment
   IMPLICIT NONE
   
   INTEGER :: narg, length, st
   CHARACTER(132) :: exename, filename
   CHARACTER(80) :: asctime
   REAL(8) :: time0, time1, time2, time3
   
   CALL GET_COMMAND_ARGUMENT(0, exename, length, st)
   IF ( st /= 0 ) THEN
      WRITE(*,*) "Error in reading program name from execution line"
      exename = "Cooling_grid"
   END IF
   narg = COMMAND_ARGUMENT_COUNT()
   IF ( narg < 1 ) THEN
      filename = "Cooling.inp"
   ELSE
      CALL GET_COMMAND_ARGUMENT(1, filename, length, st)
      IF ( st /= 0 ) THEN
         WRITE(*,*) "Error in reading filename from execution line"
         STOP "Error in program!"
      END IF
   END IF

   CALL get_date(asctime,time0)
   WRITE(*,'("Starting ",a," at: ",a)') TRIM(exename),TRIM(asctime)
   
   CALL InitGrid(filename)
      CALL get_date(asctime,time2)
      WRITE(*,'("InitGrid ended in ",F10.3," seconds")') time2-time0

   CALL FieldDistribution()
      CALL get_date(asctime,time3)
      WRITE(*,'("FieldDistribution ended in ",F10.3," seconds")') time3-time2

   CALL SensiblePoints(Sreal,Simag,Rreal,Rimag,MaxIters)
      CALL get_date(asctime,time2)
      WRITE(*,'("SensiblePoints ended in ",F10.3," seconds")') time2-time3
   !!print*,"In main: FieldWeight(1:3,1:3) = ",FieldWeight(1:3,1:3)
   !!print*,"In main: FieldWeight(Xdots-3:Xdots,Ydots-3:Ydots) = ",FieldWeight(Xdots-3:Xdots,Ydots-3:Ydots)
   !!print*,"In main: FieldCoord(1:3,1:3,1) = ",FieldCoord(1:3,1:3,1)
   !!print*,"In main: FieldCoord(Xdots-3:Xdots,Ydots-3:Ydots,1) = ",FieldCoord(Xdots-3:Xdots,Ydots-3:Ydots,1)
   !!print*,"In main: FieldCoord(1:3,1:3,2) = ",FieldCoord(1:3,1:3,2)
   !!print*,"In main: FieldCoord(Xdots-3:Xdots,Ydots-3:Ydots,2) = ",FieldCoord(Xdots-3:Xdots,Ydots-3:Ydots,2)

   print*," TheorSlope(1:3,1:3) = ",TheorSlope(1:3,1:3)
   CALL FieldInit()
      CALL get_date(asctime,time3)
      WRITE(*,'("FieldInit ended in ",F10.3," seconds")') time3-time2
   print*,"In main: FieldValues(1:3,1:3,1) = ",FieldValues(1:3,1:3,1)
   print*,"In main: FieldValues(Xdots-3:Xdots,Ydots-3:Ydots,1) = ",FieldValues(Xdots-3:Xdots,Ydots-3:Ydots,1)
   print*,"In main: FieldValues(1:3,1:3,2) = ",FieldValues(1:3,1:3,2)
   print*,"In main: FieldValues(Xdots-3:Xdots,Ydots-3:Ydots,2) = ",FieldValues(Xdots-3:Xdots,Ydots-3:Ydots,2)

   CALL Cooling(TimeSteps)
      CALL get_date(asctime,time2)
      WRITE(*,'("Cooling ended in ",F10.3," seconds")') time2-time3
   
   CALL get_date(asctime,time1)
   WRITE(*,'("Ending   at: ",a)') TRIM(asctime)
   WRITE(*,'("Computations ended in ",F10.3," seconds")') time1-time0

   WRITE(*,'("End of program!")')

   STOP "Program exit"
END PROGRAM Cooling_grid



