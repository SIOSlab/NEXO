subroutine eval_z(ns, nt, lam, eta, xi, t, z, ok)

    implicit none

    ! Inputs
    integer, intent(in) :: ns, nt
    real(8), intent(in), dimension(ns) :: lam
    real(8), intent(in), dimension(2, ns) :: eta
    real(8), intent(in), dimension(2, 2, ns) :: xi
    real(8), intent(in), dimension(nt) :: t

    ! Outputs
    real(8), intent(out), dimension(2, ns, nt) :: z
    logical, intent(out), dimension(ns, nt) :: ok

    ! Local variables
    integer :: j, k
    real(8), dimension(2, ns, nt) :: zeta

    ! Evaluate zeta
    call eval_zeta(spread(t, 1, ns), spread(lam, 2, nt), &
        spread(eta(1, :), 2, nt), spread(eta(2, :), 2, nt), &
        zeta(1, :, :), zeta(2, :, :), ok)

    ! Measurements
    do concurrent(j = 1:ns, k = 1:nt)
        z(:, j, k) = matmul(xi(:, :, j), zeta(:, j, k)) 
    end do

    ! Final check for NaN & infinite values
    ok = ok .and. (abs(z(1, :, :)) < huge(z)) &
            .and. (abs(z(2, :, :)) < huge(z)) 

    contains

        elemental subroutine eval_zeta(t, lam, eta1, eta2, zeta1, zeta2, ok)

            implicit none

            ! Inputs
            real(8), intent(in) :: t, lam, eta1, eta2

            ! Outputs
            real(8), intent(out) :: zeta1, zeta2
            logical, intent(out) :: ok

            ! Constants
            real(8), parameter :: twopi = 8 * atan(1.0D0)
                    
            ! Local variables
            real(8) :: n, es, ex, ey, f, phi, cos_phi, sin_phi, s 

            ! Mean motion
            n = twopi * exp(-lam)

            ! Eccentricity-related values
            es = hypot(1.0D0, hypot(eta1, eta2))
            ex = eta1 / es
            ey = eta2 / es
            f = sqrt(1.0D+0 - ex**2 - ey**2)

            ! Evaluate phi
            call eval_phi(t, n, ex, ey, phi, ok)

            ! Cosine & sine of phi
            cos_phi = cos(phi)
            sin_phi = sin(phi)

            ! Positions in Q-frame scaled by 1/a
            s = (ex * sin_phi - ey * cos_phi) / (1 + f)
            zeta1 = cos_phi - ex + ey * s
            zeta2 = sin_phi - ey - ex * s 

        end subroutine

        elemental subroutine eval_phi(t, n, ex, ey, phi, ok)

            implicit none

            ! Inputs
            real(8), intent(in) :: t, n, ex, ey

            ! Outputs
            real(8), intent(out) :: phi
            logical, intent(out) :: ok
    
            ! Parameters
            integer, parameter :: maxit = 100
            real(8), parameter :: phitol = 1.0D-9 

            ! Local variables
            integer :: it
            real(8) :: dm, cos_phi, sin_phi, dphi, f, fd

            ! Relative mean anomaly
            dm = n * t

            ! Initial guess
            phi = dm - ey * cos(dm) + ex * sin(dm)
            dphi = huge(phitol)

            ! Newton-Raphson iteration
            ok = .false.
            it = 0
            do while ((.not. ok) .and. (it < maxit))

                cos_phi = cos(phi)
                sin_phi = sin(phi)

                f = phi - dm - ex * sin_phi + ey * cos_phi
        
                fd = 1 - ex * cos_phi - ey * sin_phi

                dphi = f / fd

                phi = phi - dphi

                it = it + 1

                ok = abs(dphi) < phitol

            end do

        end subroutine 

end subroutine

subroutine eval_z_pm(ns, lam, xi, z, ok)

    implicit none

    ! Inputs
    integer, intent(in) :: ns
    real(8), intent(in), dimension(ns) :: lam
    real(8), intent(in), dimension(2, 2, ns) :: xi

    ! Outputs
    real(8), intent(out), dimension(ns) :: z
    logical, intent(out), dimension(ns) :: ok

    ! Local variables
    real(8), dimension(ns) :: u, v, sma2

    ! Half of squared Frobenius norm of xi
    u = 0.5D0 * (xi(1, 1, :)**2 + xi(1, 2, :)**2 &
               + xi(2, 1, :)**2 + xi(2, 2, :)**2)

    ! Determinant of xi
    v = xi(1, 1, :) * xi(2, 2, :) - xi(2, 1, :) * xi(1, 2, :)

    ! Semi-major axis squared
    sma2 = u + sqrt((u + v) * (u - v))

    ! Parallax & mass measurement value
    z = 1.5D0 * log(sma2) - 2.0D0 * lam

    ! Check for infinite & NaN values
    ok = abs(z) < huge(z)

end subroutine

subroutine run_filter(ns, nm, xs, w, t, z, cov_ww, &
        xm_p, l_xx_p, loglik_p, xm, l_xx, loglik, info)

    implicit none

    ! Inputs
    integer, intent(in) :: ns, nm
    real(8), intent(in), dimension(7, ns) :: xs 
    real(8), intent(in), dimension(ns) :: w
    real(8), intent(in), dimension(nm) :: t
    real(8), intent(in), dimension(2, nm) :: z
    real(8), intent(in), dimension(2, 2, nm) :: cov_ww
    real(8), intent(in), dimension(7) :: xm_p
    real(8), intent(in), dimension(7, 7) :: l_xx_p
    real(8), intent(in) :: loglik_p

    ! Outputs
    real(8), intent(out), dimension(7) :: xm
    real(8), intent(out), dimension(7, 7) :: l_xx
    real(8), intent(out) :: loglik
    integer, intent(out) :: info

    ! Local variables
    integer :: k
    real(8), dimension(2, nm, 2, nm) :: cov_ww_b

    ! Combined batch covariance
    cov_ww_b = 0.0D0
    do k = 1, nm
        cov_ww_b(:, k, :, k) = cov_ww(:, :, k)
    end do

    ! Priors
    xm     = xm_p
    l_xx   = l_xx_p
    loglik = loglik_p
    
    ! Filter update
    call srspf_update(h, 7, 2*nm, ns, xs, w, z, cov_ww_b, xm, l_xx, &
        loglik, info)

    contains

        subroutine h(nx, nz, ns, x, z, ok)
        
            implicit none

            ! Inputs
            integer, intent(in) :: nx, nz, ns
            real(8), intent(in), dimension(nx, ns) :: x

            ! Outputs
            real(8), intent(out), dimension(nz, ns) :: z
            logical, intent(out), dimension(ns) :: ok

            ! Local variables
            real(8), dimension(ns) :: lam
            real(8), dimension(2, ns) :: eta
            real(8), dimension(4, ns) :: xi
            real(8), dimension(2, ns, nm) :: z_k
            logical, dimension(ns, nm) :: ok_k

            ! Nonsingular parameters
            lam = x(1,   :)
            eta = x(2:3, :)
            xi  = x(4:7, :)
            
            ! Compute z values
            call eval_z(ns, nm, lam, eta, xi, t, z_k, ok_k)
            
            ! Combine batch measurements
            z(1:(2*nm):2, :) = transpose(z_k(1, :, :))
            z(2:(2*nm):2, :) = transpose(z_k(2, :, :))

            ! Check combined measurements
            ok = all(ok_k, 2)

        end subroutine

end subroutine

subroutine mix_filter(nm, nq, wgt_p, xm_p, l_xx_p, t, z, cov_ww, xm, l_xx)

    implicit none

    ! Inputs
    integer, intent(in) :: nm, nq
    real(8), intent(in), dimension(nq) :: wgt_p
    real(8), intent(in), dimension(7, nq) :: xm_p
    real(8), intent(in), dimension(7, 7, nq) :: l_xx_p
    real(8), intent(in), dimension(nm) :: t
    real(8), intent(in), dimension(2, nm) :: z
    real(8), intent(in), dimension(2, 2, nm) :: cov_ww

    ! Outputs
    real(8), intent(out), dimension(7) :: xm
    real(8), intent(out), dimension(7, 7) :: l_xx

    ! Local variables
    integer :: ns, j, info
    real(8), dimension(nq) :: loglik_p, loglik, w_mix, v
    real(8), dimension(7, nq) :: xm_q
    real(8), dimension(7, 7, nq) :: l_xx_q
    real(8), dimension(7, 8, nq) :: ps, work
    real(8), dimension(7) :: tau
    logical, dimension(nq) :: ok
    real(8), allocatable :: xs(:, :), w(:)

    ! Number of sample points for filter
    call en_r2_05_3_size(7, ns)
 
    ! Allocate arrays for points & weights
    allocate(xs(7, ns), w(ns))

    ! Cubature rule
    call en_r2_05_3(7, ns, xs, w)
 
    ! Scale cubature points
    xs = sqrt(2.0D0) * xs

    ! Normalize cubature weights
    w = w / sum(w)

    ! Prior log-likelihoods
    loglik_p = log(wgt_p)

    ! Iterate over mixture components
    do j = 1, nq

        ! Run filter
        call run_filter(ns, nm, xs, w, &
            t, z, cov_ww, xm_p(1, j), l_xx_p(1, 1, j), loglik_p(j), &
            xm_q(1, j), l_xx_q(1, 1, j), loglik(j), info)

        ! Check whether filter was successful
        ok(j) = info == 0

        ! Set posteriors to zero if unsuccessful
        if (.not. ok(j)) then
            xm_q  (:,    j) = 0
            l_xx_q(:, :, j) = 0
        end if

    end do

    ! Regularize log-likelihoods
    loglik = loglik - maxval(loglik, ok)

    ! Mixture weights
    where (ok)
        w_mix = exp(loglik)
    else where
        w_mix = 0
    end where

    ! Normalize mixture weights
    w_mix = w_mix / sum(w_mix)

    ! Square roots of weights
    v = sqrt(w_mix)

    ! Overall mean
    xm = matmul(xm_q, w_mix)

    ! Square-root weighted posterior sample points
    ps(:, 1:7, :) = l_xx_q * spread(spread(v, 1, 7), 1, 7)
    ps(:,   8, :) = (xm_q - spread(xm, 2, nq)) * spread(v, 1, 7)

    ! LQ decomposition
    call dgelqf(7, 8*nq, ps, 7, tau, work, 56*nq, info)
    if (info /= 0) stop 'mix_filter: LQ decomposition failed!'

    ! Posterior square root of covariance
    l_xx = 0
    call dlacpy('L', 7, 7, ps, 7, l_xx, 7)

end subroutine

subroutine mix_filter_pop(nm, std_a, std_eta, mm, std_m, px, std_px, alpha, &
        t, z, cov_ww, xm, l_xx)

    implicit none

    ! Inputs
    integer, intent(in) :: nm
    real(8), intent(in) :: std_a, std_eta, mm, std_m, px, std_px, alpha
    real(8), intent(in), dimension(nm) :: t
    real(8), intent(in), dimension(2, nm) :: z
    real(8), intent(in), dimension(2, 2, nm) :: cov_ww

    ! Outputs
    real(8), intent(out), dimension(7) :: xm
    real(8), intent(out), dimension(7, 7) :: l_xx

    ! Local variables
    integer :: nq, j, lwork, info
    real(8), dimension(8) :: ym
    real(8), dimension(8, 8) :: l_yy
    real(8), dimension(7) :: tau
    real(8), allocatable :: wc(:), wgt_p(:), yc(:, :), y(:, :), &
        u(:), v(:), sma(:), lam(:), x(:, :), work(:, :), &
        xm_p(:, :), l_xx_p(:, :, :)
    logical, allocatable :: ok(:)

    ! Constants
    real(8), parameter :: twopi = 8 * atan(1.0D0)
    
    ! Mean of y
    ym = 0
    ym(1) = mm
    ym(2) = px
        
    ! Square root of covariance of y
    l_yy = 0
    l_yy(1, 1) = std_m
    l_yy(2, 2) = std_px
    l_yy(3, 3) = std_eta
    l_yy(4, 4) = std_eta
    l_yy(5, 5) = std_a
    l_yy(6, 6) = std_a
    l_yy(7, 7) = std_a
    l_yy(8, 8) = std_a

    ! Number of mixture components
    call en_r2_05_3_size(8, nq)

    ! Allocate arrays
    allocate(wc(nq), wgt_p(nq), yc(8, nq), y(8, nq), &
        u(nq), v(nq), sma(nq), lam(nq), x(7, nq), work(7, nq), &
        xm_p(7, nq), l_xx_p(7, 7, nq))
    allocate(ok(nq))

    ! Cubature rule
    call en_r2_05_3(8, nq, yc, wgt_p)
 
    ! Scale cubature points
    call dtrmm('L', 'L', 'N', 'N', 8, nq, sqrt(2.0D0), l_yy, 8, yc, 8)

    ! Normalize cubature weights
    wgt_p = wgt_p / sum(wgt_p)

    ! Initalize covariances of x to zero
    l_xx_p = 0

    ! Length of work array
    lwork = 7 * nq

    ! Compute mixture components
    do j = 1, nq
        
        ! Sample points for y
        y = alpha * yc + spread((1 - alpha) * yc(:, j) + ym, 2, nq)

        ! Half of squared Frobenius norm of xi-prime
        u = 0.5D0 * (y(5, :)**2 + y(6, :)**2 + &
                     y(7, :)**2 + y(8, :)**2)

        ! Determinant of xi-prime
        v = y(5, :) * y(8, :) - y(6, :) * y(7, :)

        ! Semi-major axis
        sma = sqrt(u + sqrt((u + v) * (u - v)))

        ! Log-period
        lam = 1.5D0 * log(sma) - 0.5D0 * log(y(1, :))

        ! Check log periods
        ok = (lam > -huge(lam)) .and. (lam < huge(lam))

        ! Weights
        wc = wgt_p

        ! Set bad log-periods & weights to zero
        where (.not. ok)
           lam = 0
           wc  = 0
        end where

        ! Re-normalize weights
        wc = wc / sum(wc)

        ! Convert to nonsingular orbital elements
        x(1,   :) = lam
        x(2:3, :) = y(3:4, :)
        x(4:7, :) = y(5:8, :) * spread(y(2, :), 1, 4)

        ! Mean of x
        xm_p(:, j) = matmul(x, wgt_p)

        ! Centralized & square-root weighted x values
        x = (x - spread(xm_p(:, j), 2, nq)) * spread(sqrt(wgt_p), 1, 7)

        ! LQ factorization
        call dgelqf(7, nq, x, 7, tau, work, lwork, info)
        if (info /= 0) stop 'mix_filter_pop: LQ decomposition failed!'

        ! Copy square root of covariance
        call dlacpy('L', 7, 7, x, 7, l_xx_p(1, 1, j), 7) 

    end do

    ! Run mixture filter
    call mix_filter(nm, nq, wgt_p, xm_p, l_xx_p, t, z, cov_ww, xm, l_xx)

end

subroutine coe2nse(n, px, sma, ecc, inc, lan, aop, mae, per, lam, eta, xi)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: px, sma, ecc, inc, lan, aop, mae, per

    ! Outputs
    real(8), intent(out), dimension(n) :: lam
    real(8), intent(out), dimension(2, n) :: eta
    real(8), intent(out), dimension(2, 2, n) :: xi

    ! Constants
    real(8), parameter :: deg2rad = atan(1.0D0) / 45
    
    ! Local variables
    real(8), dimension(n) :: m0, eta_norm, u, w, cos_w, sin_w, &
        cos_u, sin_u, cos_i, s

    lam = log(per)

    m0 = deg2rad * mae

    eta_norm = ecc / sqrt(1 - ecc**2)

    eta(1, :) =  cos(m0) * eta_norm
    eta(2, :) = -sin(m0) * eta_norm

    w = deg2rad * lan
    u = deg2rad * (aop + mae)

    cos_w = cos(w)
    sin_w = sin(w)
    cos_u = cos(u)
    sin_u = sin(u)

    cos_i = cos(deg2rad * inc)

    s = sma * px

    xi(1, 1, :) = s * ( cos_u*cos_w - sin_u*sin_w*cos_i)
    xi(2, 1, :) = s * ( cos_u*sin_w + sin_u*cos_w*cos_i)
    xi(1, 2, :) = s * (-sin_u*cos_w - cos_u*sin_w*cos_i)
    xi(2, 2, :) = s * (-sin_u*sin_w + cos_u*cos_w*cos_i)

end subroutine

subroutine nse2coe(n, px, lam, eta, xi, sma, ecc, inc, lan, aop, mae, per)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: px, lam
    real(8), intent(in), dimension(2, n) :: eta
    real(8), intent(in), dimension(2, 2, n) :: xi

    ! Outputs
    real(8), intent(out), dimension(n) :: sma, ecc, inc, lan, aop, mae, per

    ! Constants
    real(8), parameter :: rad2deg = 45 / atan(1.0D0)

    ! Local variables
    real(8), dimension(n) :: eta_norm, xi_norm2, xi_det, c, cos_i, y1, y2

    per = exp(lam)

    eta_norm = hypot(eta(1, :), eta(2, :))

    ecc = eta_norm / hypot(1.0D0, eta_norm)

    mae = rad2deg * atan2(-eta(2, :), eta(1, :))

    y1 = rad2deg * atan2(xi(2, 1, :) - xi(1, 2, :), xi(1, 1, :) + xi(2, 2, :))
    y2 = rad2deg * atan2(xi(2, 1, :) + xi(1, 2, :), xi(1, 1, :) - xi(2, 2, :))

    lan = 0.5D0 * (y1 + y2)
    
    where (lan < 0) lan = lan + 180.0D0

    aop = y1 - lan - mae

    where (aop < 0) aop = aop + 360.0D0
    where (mae < 0) mae = mae + 360.0D0

    xi_norm2 = (xi(1, 1, :)**2 + xi(2, 1, :)**2 &
              + xi(1, 2, :)**2 + xi(2, 2, :)**2)

    xi_det = xi(1, 1, :) * xi(2, 2, :) - xi(2, 1, :) * xi(1, 2, :)

    c = 0.5 * xi_norm2 / xi_det

    cos_i = c - sign(sqrt(c**2 - 1), c)

    inc = rad2deg * acos(cos_i)

    sma = sqrt(xi_norm2) / (px * hypot(1.0D0, cos_i)) 

end subroutine

subroutine radec2z(n, raoff, decoff, std_ra, std_dec, corr_radec, z, cov_zz)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: raoff, decoff, std_ra, std_dec
    real(8), intent(in), dimension(n) :: corr_radec
    
    ! Outputs
    real(8), intent(out), dimension (2, n) :: z
    real(8), intent(out), dimension (2, 2, n) :: cov_zz

    z(1, :) = decoff
    z(2, :) = raoff

    cov_zz(1, 1, :) = std_dec**2
    cov_zz(2, 2, :) = std_ra**2

    cov_zz(2, 1, :) = corr_radec * std_dec * std_ra

    cov_zz(1, 2, :) = cov_zz(2, 1, :)

end subroutine

subroutine z2radec(n, z, cov_zz, raoff, decoff, std_ra, std_dec, corr_radec)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension (2, n) :: z
    real(8), intent(in), dimension (2, 2, n) :: cov_zz
    
    ! Outputs
    real(8), intent(out), dimension(n) :: raoff, decoff, std_ra, std_dec
    real(8), intent(out), dimension(n) :: corr_radec

    raoff  = z(2, :)
    decoff = z(1, :)

    std_ra  = sqrt(cov_zz(2, 2, :))
    std_dec = sqrt(cov_zz(1, 1, :))

    corr_radec = cov_zz(2, 1, :) / (std_ra * std_dec)

end subroutine

subroutine seppa2z(n, sep, pa, std_sep, std_pa, corr_seppa, z, cov_zz)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: sep, pa, std_sep, std_pa, corr_seppa

    ! Outputs
    real(8), intent(out), dimension(2, n) :: z 
    real(8), intent(out), dimension(2, 2, n) :: cov_zz

    ! Constants
    real(8), parameter :: deg2rad = atan(1.0D0) / 45
    
    ! Local variables
    integer :: j
    real(8), dimension(n) :: theta, std_arc, cos_theta, sin_theta
    real(8), dimension(2, 2, n) :: cov, rot

    theta = deg2rad * pa

    std_arc = deg2rad * std_pa * sep

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    z(1, :) = sep * cos_theta
    z(2, :) = sep * sin_theta

    rot(1, 1, :) =  cos_theta
    rot(2, 1, :) =  sin_theta
    rot(1, 2, :) = -sin_theta
    rot(2, 2, :) =  cos_theta

    cov(1, 1, :) = std_sep**2
    cov(2, 2, :) = std_arc**2
    cov(2, 1, :) = std_sep * std_arc * corr_seppa
    cov(1, 2, :) = cov(2, 1, :) 

    do concurrent(j = 1:n)
        cov_zz(:, :, j) = matmul(matmul(rot(:, :, j), cov(:, : ,j)), &
                transpose(rot(:, :, j)))
    end do

end subroutine

subroutine z2seppa(n, z, cov_zz, sep, pa, std_sep, std_pa, corr_seppa)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(2, n) :: z 
    real(8), intent(in), dimension(2, 2, n) :: cov_zz

    ! Outputs
    real(8), intent(out), dimension(n) :: sep, pa, std_sep, std_pa, corr_seppa

    ! Constants
    real(8), parameter :: rad2deg = 45 / atan(1.0D0)

    ! Local variables
    integer :: j
    real(8), dimension(n) :: theta, cos_theta, sin_theta
    real(8), dimension(2, 2, n) :: cov, rot

    sep = hypot(z(1, :), z(2, :))

    theta = atan2(z(2, :), z(1, :))

    pa = rad2deg * theta

    where (pa < 0) pa = pa + 360.0D0

    cos_theta = z(1, :) / sep
    sin_theta = z(2, :) / sep

    rot(1, 1, :) =  cos_theta
    rot(2, 1, :) =  sin_theta
    rot(1, 2, :) = -sin_theta
    rot(2, 2, :) =  cos_theta

    do concurrent(j = 1:n)
        cov(:, :, j) = matmul(matmul(transpose(rot(:, :, j)), &
                cov_zz(:, :, j)), rot(:, :, j))
    end do

    std_sep = sqrt(cov(1, 1, :))

    std_pa = rad2deg * sqrt(cov(2, 2, :)) / sep

    corr_seppa = cov(2, 1, :) / sqrt(cov(1, 1, :) * cov(2, 2, :))

end subroutine

subroutine rando(n, xm, l_xx, pxm, std_px, seed_in, lam, eta, xi, px, seed_out)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in) :: pxm, std_px
    integer, intent(in), dimension(4) :: seed_in

    ! Outputs
    real(8), intent(out), dimension(n) :: lam
    real(8), intent(out), dimension(2, n) :: eta
    real(8), intent(out), dimension(2, 2, n) :: xi
    real(8), intent(out), dimension(n) :: px
    integer, intent(out), dimension(4) :: seed_out

    ! Constants
    integer, parameter :: dist = 3

    ! Local variables 
    real(8), dimension(7, n) :: x

    ! Set seed
    seed_out = seed_in

    ! Standard normal random values
    call dlarnv(dist, seed_out, 7*n, x)

    ! Centralized values
    call dtrmm('L', 'L', 'N', 'N', 7, n, 1.0D0, l_xx, 7, x, 7)
        
    ! Non-central values
    x = x + spread(xm, 2, n)

    ! Components
    call dlacpy('A', 1, n, x(1, 1), 7, lam, 1)
    call dlacpy('A', 2, n, x(2, 1), 7, eta, 2)
    call dlacpy('A', 4, n, x(4, 1), 7, xi,  4)

    ! Standard normal random values
    call dlarnv(dist, seed_out, n, px)

    ! Parallax values
    px = std_px * px + pxm

end subroutine

subroutine gen_prior(std_lam, std_eta, std_xi, xm, l_xx)

    implicit none

    ! Inputs
    real(8), intent(in) :: std_lam, std_eta, std_xi

    ! Outputs
    real(8), intent(out), dimension(7) :: xm
    real(8), intent(out), dimension(7, 7) :: l_xx

    ! Zero mean
    xm = 0

    ! Diagonal covariance
    l_xx = 0
    l_xx(1, 1) = std_lam
    l_xx(2, 2) = std_eta
    l_xx(3, 3) = l_xx(2, 2)
    l_xx(4, 4) = std_xi
    l_xx(5, 5) = l_xx(4, 4)
    l_xx(6, 6) = l_xx(4, 4)
    l_xx(7, 7) = l_xx(4, 4)

end subroutine

subroutine nse_ci(xm, l_xx, z, ci_lam, ci_eta, ci_xi)

    implicit none

    ! Inputs
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in) :: z

    ! Outputs
    real(8), intent(out), dimension(3) :: ci_lam
    real(8), intent(out), dimension(2, 3) :: ci_eta
    real(8), intent(out), dimension(2, 2, 3) :: ci_xi

    ! Local variables
    integer :: i, info
    real(8), dimension(7, 7) :: cov_xx
    real(8), dimension(7) :: std_x
    real(8), dimension(7, 3) :: ci_x

    ! Covariance of x (lower triangular only)
    cov_xx = l_xx
    call dlauum('L', 7, cov_xx, 7, info)
    if (info /= 0) stop 'Covariance calculation failed!'

    ! Standard deviation of each component of x
    do concurrent (i = 1:7)
        std_x(i) = sqrt(cov_xx(i, i))
    end do

    ! Credible intervals for x
    ci_x(:, 1) = xm - std_x * z
    ci_x(:, 2) = xm
    ci_x(:, 3) = xm + std_x * z

    ! Credible intervals for lambda & eta
    ci_lam = ci_x(1,   :)
    ci_eta = ci_x(2:3, :)

    ! Credible intervals for xi
    ci_xi = reshape(ci_x(4:7, :), [2, 2, 3])

end subroutine

subroutine predict_z(nm, xm, l_xx, t, zm, cov_zz)

    implicit none

    ! Inputs
    integer, intent(in) :: nm
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in), dimension(nm) :: t

    ! Outputs
    real(8), intent(out), dimension(2, nm) :: zm
    real(8), intent(out), dimension(2, 2, nm) :: cov_zz

    ! Local variables
    integer :: nq, k
    real(8), allocatable, dimension(:) :: lam, wa, wq
    real(8), allocatable, dimension(:, :) :: x, xq, eta, xi, zc
    real(8), allocatable, dimension(:, :, :) :: z
    logical, allocatable, dimension(:, :) :: ok

    ! Number of quadrature points
    call en_r2_05_3_size(7, nq)

    ! Allocate arrays
    allocate(lam(nq), wa(nq), wq(nq), x(7, nq), xq(7, nq), eta(2, nq), &
        xi(4, nq), zc(2, nq), z(2, nq, nm), ok(nq, nm))  

    ! Cubature rule
    call en_r2_05_3(7, nq, xq, wq)
   
    ! Scale cubature points
    xq = sqrt(2.0D0) * xq

    ! Values of x
    x = matmul(l_xx, xq) + spread(xm, 2, nq)

    ! Nonsingular elements
    lam = x(1,   :)
    eta = x(2:3, :)
    xi  = x(4:7, :)

    ! Predicted measurements
    call eval_z(nq, nm, lam, eta, xi, t, z, ok)

    ! Set bad measurement to zero
    where (.not. ok)
        z(1, :, :) = 0
        z(2, :, :) = 0
    end where

    ! Iterate over times
    do k = 1, nm

        ! Set bad weights to zero
        where (ok(:, k))
            wa = wq
        else where
            wa = 0
        end where

        ! Re-normalize weights
        wa = wa / sum(wa)

        ! Measurement mean
        zm(:, k) = matmul(z(:, :, k), wa)

        ! Centralized measurements
        zc = z(:, :, k) - spread(zm(:, k), 2, nq)

        ! Measurement covariance
        cov_zz(:, :, k) = matmul(zc * spread(wa, 1, 2), transpose(zc))  

    end do

end subroutine

subroutine lin_ci(n, x, w, conf, ci)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: x, w
    real(8), intent(in) :: conf

    ! Output
    real(8), intent(out), dimension(3) :: ci

    ! Local variables
    real(8) :: xm, var_x, eps

    ! Mean of x
    xm = dot_product(w, x)

    ! Variance of x
    var_x = dot_product(w, (x - xm)**2)

    ! Chebyshev factor
    eps = sqrt(var_x / (1 - conf))

    ! Confidence interval
    ci(1) = xm - eps
    ci(2) = xm
    ci(3) = xm + eps

end subroutine

subroutine cir_ci(n, x, w, conf, ci)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: x, w
    real(8), intent(in) :: conf

    ! Output
    real(8), intent(out), dimension(3) :: ci

    ! Constants
    real(8), parameter :: deg2rad = atan(1.0D0) / 45
    real(8), parameter :: rad2deg = 45 / atan(1.0D0)

    ! Local variables
    real(8) :: cxm, sxm, xm, r, dx

    ! Means of cosine & sine of x
    cxm = dot_product(w, cos(deg2rad * x))
    sxm = dot_product(w, sin(deg2rad * x))

    ! Circular mean of x
    xm = rad2deg * atan2(sxm, cxm)

    ! Shift cirular mean to [0, 360]
    if (xm < 0) xm = xm + 360

    ! Mean resultant length
    r = hypot(cxm, sxm)

    ! Discrepancy bound
    dx = rad2deg * 2 * asin(min(1.0D0, sqrt((1 - r) / (2 * (1 - conf)))))

    ! Confidence interval
    ci(1) = xm - dx
    ci(2) = xm
    ci(3) = xm + dx

end subroutine

subroutine lin_ci_eqw(n, x, conf, ci)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: x
    real(8), intent(in) :: conf

    ! Output
    real(8), intent(out), dimension(3) :: ci

    ! Local variables
    real(8), dimension(n) :: w

    ! Weights
    w = 1.0D0 / n

    ! Weighted confidence interval
    call lin_ci(n, x, w, conf, ci)

end subroutine

subroutine cir_ci_eqw(n, x, conf, ci)

    implicit none

    ! Inputs
    integer, intent(in) :: n
    real(8), intent(in), dimension(n) :: x
    real(8), intent(in) :: conf

    ! Output
    real(8), intent(out), dimension(3) :: ci

    ! Local variables
    real(8), dimension(n) :: w

    ! Weights
    w = 1.0D0 / n

    ! Weighted confidence interval
    call cir_ci(n, x, w, conf, ci)

end subroutine

subroutine coe_ci(ns, xs, w, xm, l_xx, pxm, std_px, conf, ci_sma, ci_ecc, &
        ci_inc, ci_lan, ci_aop, ci_mae, ci_per, ci_tp)

    ! Inputs
    integer, intent(in) :: ns
    real(8), intent(in), dimension(8, ns) :: xs
    real(8), intent(in), dimension(ns) :: w
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in) :: pxm, std_px, conf

    ! Outputs
    real(8), intent(out), dimension(3) :: ci_sma, ci_ecc, ci_inc, ci_lan, &
                ci_aop, ci_mae, ci_per, ci_tp

    ! Local variables
    real(8), dimension(7, ns) :: x
    real(8), dimension(ns) :: px, lam, sma, ecc, inc, lan, aop, mae, per, tp
    real(8), dimension(2, ns) :: eta
    real(8), dimension(4, ns) :: xi

    ! Centralized x values
    x = xs(1:7, :)
    call dtrmm('L', 'L', 'N', 'N', 7, ns, 1.0D0, l_xx, 7, x, 7)

    ! Non-central x values
    x = x + spread(xm, 2, ns)

    ! Nonsingular elements
    lam = x(1,   :)
    eta = x(2:3, :)
    xi  = x(4:7, :)

    ! Parallax values
    px = pxm + std_px * xs(8, :)

    ! Convert to classical elements
    call nse2coe(ns, px, lam, eta, xi, sma, ecc, inc, lan, aop, mae, per)

    ! Times of periapse passage
    tp = per * (1 - mae / 360.0D0)

    ! Confidence intervals
    call lin_ci(ns, sma, w, conf, ci_sma)
    call lin_ci(ns, ecc, w, conf, ci_ecc)
    call cir_ci(ns, inc, w, conf, ci_inc)
    call cir_ci(ns, lan, w, conf, ci_lan)
    call cir_ci(ns, aop, w, conf, ci_aop)
    call cir_ci(ns, mae, w, conf, ci_mae)
    call lin_ci(ns, per, w, conf, ci_per)
    call lin_ci(ns, tp,  w, conf, ci_tp )

end subroutine

subroutine coe_ci_stroud(xm, l_xx, pxm, std_px, conf, ci_sma, ci_ecc, &
        ci_inc, ci_lan, ci_aop, ci_mae, ci_per, ci_tp)

    ! Inputs
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in) :: pxm, std_px, conf

    ! Outputs
    real(8), intent(out), dimension(3) :: ci_sma, ci_ecc, ci_inc, ci_lan, &
                ci_aop, ci_mae, ci_per, ci_tp

    ! Local variables
    integer :: ns
    real(8), allocatable :: xs(:, :), w(:)

    ! Number of sample points for filter
    call en_r2_05_3_size(8, ns)
 
    ! Allocate arrays for points & weights
    allocate(xs(8, ns), w(ns))

    ! Cubature rule
    call en_r2_05_3(8, ns, xs, w)
 
    ! Scale cubature points
    xs = sqrt(2.0D0) * xs

    ! Normalize cubature weights
    w = w / sum(w)

    ! Compute confidence intervals
    call coe_ci(ns, xs, w, xm, l_xx, pxm, std_px, conf, ci_sma, ci_ecc, &
            ci_inc, ci_lan, ci_aop, ci_mae, ci_per, ci_tp)

end subroutine

subroutine eval_err(ns, lam_tru, eta_tru, xi_tru, lam, eta, xi, w, ti, tf, &
        rmse, chi2m, ok)

    implicit none

    ! Inputs
    integer, intent(in) :: ns
    real(8), intent(in) :: lam_tru
    real(8), intent(in), dimension(2) :: eta_tru
    real(8), intent(in), dimension(2, 2) :: xi_tru
    real(8), intent(in), dimension(ns) :: lam
    real(8), intent(in), dimension(2, ns) :: eta
    real(8), intent(in), dimension(2, 2, ns) :: xi
    real(8), intent(in), dimension(ns) :: w
    real(8), intent(in) :: ti, tf

    ! Outputs
    real(8), intent(out) :: rmse, chi2m
    logical, intent(out) :: ok

    ! Parameters
    integer, parameter :: key   = 3
    integer, parameter :: limit = 1000
    integer, parameter :: lenw  = limit * 4
    real(8), parameter :: atol  = 1.0D-2
    real(8), parameter :: rtol  = 1.0D-2

    ! Local variables
    integer :: neval, ier_mse, ier_chi2m, last
    integer, dimension(limit) :: iwork
    real(8) :: res, abserr
    real(8), dimension(lenw) :: work
    logical :: eval_chi2

    ! Mean-square error
    eval_chi2 = .false.
    call dqag(f, ti, tf, atol, rtol, key, res, abserr, neval, ier_mse, &
        limit, lenw, last, iwork, work)
    rmse = sqrt(res)

    ! Compute mean chi-square value
    eval_chi2 = .true.
    call dqag(f, ti, tf, atol, rtol, key, res, abserr, neval, ier_chi2m, &
        limit, lenw, last, iwork, work)
    chi2m = res

    ! Check if integrations were sucessful
    ok = (ier_mse == 0) .and. (ier_chi2m == 0)

    if (.not. ok) print*, 'ier: ', ier_mse, ier_chi2m

    contains

        real(8) function f(t)

            implicit none

            ! Input
            real(8) :: t

            ! Local variables
            integer :: info
            real(8), dimension(2) :: z_tru, zm, z_err, tau
            real(8), dimension(2, ns) :: z, work
            real(8), dimension(ns) :: wa
            logical, dimension(1) :: ok_tru
            logical, dimension(ns) :: ok

            ! Evaluate true measurement
            call eval_z(1, 1, [lam_tru], eta_tru, xi_tru, [t], z_tru, ok_tru)
            if (.not. ok_tru(1)) stop 'Estimator error calculation failed' 

            ! Evaluate sample measurements
            call eval_z(ns, 1, lam, eta, xi, [t], z, ok)
            if (all(.not. ok)) stop 'Estimator error calculation failed'
                
            ! Set bad weights & measurements to zero
            where (ok)
                wa = w
            else where
                wa = 0
                z(1, :) = 0
                z(2, :) = 0
            end where

            ! Re-normalize weights
            wa = wa / sum(wa)

            ! Measurement mean
            zm = matmul(z, wa)

            ! Estimation error
            z_err = zm - z_tru

            ! Evaluate error
            if (eval_chi2) then

                ! Centralize & scale sample measurements
                z = (z - spread(zm, 2, ns)) * spread(sqrt(wa), 1, 2)

                ! LQ decomposition
                call dgelqf(2, ns, z, 2, tau, work, 2*ns, info)
                if (info /= 0) stop 'LQ decomposition failed!' 

                ! Normalize error
                call dtrsv('L', 'N', 'N', 2, z, 2, z_err, 1) 

            end if

            ! Time-scaled squared estimation error
            f = norm2(z_err)**2 / (tf - ti)

        end function

end subroutine

subroutine eval_err_srspf(lam_tru, eta_tru, xi_tru, xm, l_xx, ti, tf, &
        rmse, chi2m, ok)

    implicit none

    ! Inputs
    real(8), intent(in) :: lam_tru
    real(8), intent(in), dimension(2) :: eta_tru
    real(8), intent(in), dimension(2, 2) :: xi_tru
    real(8), intent(in), dimension(7) :: xm
    real(8), intent(in), dimension(7, 7) :: l_xx
    real(8), intent(in) :: ti, tf

    ! Outputs
    real(8), intent(out) :: rmse, chi2m
    logical, intent(out) :: ok
    
    ! Local variables
    integer :: ns
    real(8), allocatable :: x(:, :), w(:), lam(:), eta(:, :), xi(:, :, :)

    ! Number of sample points for filter
    call en_r2_05_3_size(7, ns)
 
    ! Allocate arrays for points & weights
    allocate(x(7, ns), w(ns), lam(ns), eta(2, ns), xi(2, 2, ns))

    ! Cubature rule
    call en_r2_05_3(7, ns, x, w)
 
    ! Centralized sample points
    call dtrmm('L', 'L', 'N', 'N', 7, ns, sqrt(2.0D0), l_xx, 7, x, 7)

    ! Non-central sample points
    x = x + spread(xm, 2, ns)

    ! Nonsingular elements
    lam = x(1,   :)
    eta = x(2:3, :)
    xi  = reshape(x(4:7, :), [2, 2, ns])

    ! Compute errors
    call eval_err(ns, lam_tru, eta_tru, xi_tru, lam, eta, xi, w, ti, tf, &
        rmse, chi2m, ok)

end subroutine

subroutine scale_mix(nq, xm_0, cov_xx_0, mstar, std_mstar, px, std_px, &
        xm, l_xx)

    implicit none

    ! Inputs
    integer, intent(in) :: nq
    real(8), intent(in), dimension(7, nq) :: xm_0
    real(8), intent(in), dimension(7, 7, nq) :: cov_xx_0
    real(8), intent(in) :: mstar, std_mstar, px, std_px

    ! Outputs
    real(8), intent(out), dimension(7, nq) :: xm
    real(8), intent(out), dimension(7, 7, nq) :: l_xx

    ! Local variables
    integer :: j, info
    real(8) :: var_px, px2m 
    real(8), dimension(7, 7, nq) :: cov_xx

    ! Means of lambda
    xm(1, :) = xm_0(1, :) - log(mstar) / 2

    ! Means of eta
    xm(2:3, :) = xm_0(2:3, :)

    ! Means of xi
    xm(4:7, :) = xm_0(4:7, :) * px

    ! Variance of lambda
    cov_xx(1, 1, :) = cov_xx_0(1, 1, :) + (std_mstar / mstar)**2 / 4

    ! Covariances & cross-covariances of eta
    cov_xx(2:3, 1:3, :) = cov_xx_0(2:3, 1:3, :)
    cov_xx(1:3, 2:3, :) = cov_xx_0(1:3, 2:3, :)

    ! Cross-covariances of xi
    cov_xx(1:3, 4:7, :) = cov_xx_0(1:3, 4:7, :) * px
    cov_xx(4:7, 1:3, :) = cov_xx_0(4:7, 1:3, :) * px

    ! Variance & mean-square of parallax
    var_px = std_px**2
    px2m   = var_px + px**2 

    ! Covariances of xi
    do j = 1, nq
        cov_xx(4:7, 4:7, j) = cov_xx_0(4:7, 4:7, j) * px2m &
            + spread(xm_0(4:7, j), 2, 4) * spread(xm_0(4:7, j), 1, 4)  * var_px
    end do

    ! Square roots of covariances
    l_xx = 0.0D0
    do j = 1, nq

        call dlacpy('L', 7, 7, cov_xx(1, 1, j), 7, l_xx(1, 1, j), 7)

        call dpotrf('L', 7, l_xx(1, 1, j), 7, info) 

        if (info /= 0) stop 'scale_mix: Cholesky decomposition failed' 

    end do

end
