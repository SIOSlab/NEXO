subroutine srspf_update(h, nx, nz, ns, xs, w, t, z_tru, cov_ww, &
        xm, l_xx, loglik, info)

    implicit none

    ! Interface for measurement function
    interface
        subroutine h(nx, nz, ns, t, x, z, ok)
            integer, intent(in) :: nx, nz, ns
            real(8), intent(in) :: t
            real(8), intent(in), dimension(nx, ns) :: x
            real(8), intent(out), dimension(nz, ns) :: z
            logical, intent(out), dimension(ns) :: ok
        end subroutine
    end interface

    ! Inputs
    integer, intent(in) :: nx, nz, ns
    real(8), intent(in), dimension(nx, ns) :: xs
    real(8), intent(in), dimension(ns) :: w
    real(8), intent(in) :: t
    real(8), intent(in), dimension(nz) :: z_tru
    real(8), intent(in), dimension(nz, nz) :: cov_ww

    ! Inputs/Outputs
    real(8), intent(inout), dimension(nx) :: xm
    real(8), intent(inout), dimension(nx, nx) :: l_xx
    real(8), intent(inout) :: loglik

    ! Outputs
    integer, intent(out) :: info

    ! Local variables
    integer :: i
    real(8), dimension(nx, ns) :: x, xc, s, work
    real(8), dimension(nz, ns) :: z, zc, zs, zcw
    logical, dimension(ns) :: ok
    real(8), dimension(ns) :: wa
    real(8), dimension(nz) :: zm, zs_tru
    real(8), dimension(nz, nz) :: cov_zz, l_zz
    real(8), dimension(nx, nz) :: cov_xz, gain
    real(8), dimension(nx) :: tau
    real(8) :: log_det_l_zz

    ! Centralized x values
    xc = xs
    call dtrmm('L', 'L', 'N', 'N', nx, ns, 1.0D+0, l_xx, nx, xc, nx)

    ! Non-central x values
    x = xc + spread(xm, 2, ns)

    ! Predicted measurements
    call h(nx, nz, ns, t, x, z, ok)
        
    ! Stop if too few good points
    if (count(ok) < nx + 1) then
        info = 1
        return
    end if 

    ! Adjust weights
    where (ok)
        wa = w
    else where
        wa = 0
    end where

    ! Normalize adjusted weights
    wa = wa / sum(wa)

    ! Set bad z values to zero
    do concurrent(i = 1:nz)
        where (.not. ok) z(i, :) = 0
    end do

    ! Mean of z
    zm = matmul(z, wa)

    ! Centralized z values
    zc = z - spread(zm, 2, ns)

    ! Centralized & weighted z values
    zcw = zc * spread(wa, 1, nz)

    ! Covariance of z
    cov_zz = matmul(zc, transpose(zcw)) + cov_ww

    ! Cross-covariance of x & z
    cov_xz = matmul(xc, transpose(zcw))

    ! Cholesky decomposition of covariance of z (lower triangular)
    l_zz = cov_zz
    info = 0
    call dpotrf('L', nz, l_zz, nz, info)
    if (info /= 0) return

    ! Log-determinant of Cholesky decomposition of z
    log_det_l_zz = 0
    do i = 1, nz
        log_det_l_zz = log_det_l_zz + log(l_zz(i, i))
    end do

    ! Standardized true measurement
    zs_tru = z_tru - zm 
    call dtrsv('L', 'N', 'N', nz, l_zz, nz, zs_tru, 1)

    ! Standardized z values
    zs = zc
    call dtrsm('L', 'L', 'N', 'N', nz, ns, 1.0D+0, l_zz, nz, zs, nz)

    ! Kalman gain
    gain = cov_xz
    call dtrsm('R', 'L', 'T', 'N', nx, nz, 1.0D+0, l_zz, nz, gain, nx)

    ! Updated mean
    xm = xm + matmul(gain, zs_tru)

    ! Updated, centralized & square-root weighted x values
    s = (xc -  matmul(gain, zs)) * spread(sqrt(wa), 1, nx)

    ! LQ decomposition
    call dgelqf(nx, ns, s, nx, tau, work, nx*ns, info)
    if (info /= 0) return

    ! Updated Cholesky decomposition of covariance
    call dlacpy('L', nx, nx, s, nx, l_xx, nx) 

    ! Updated log-likelihood
    loglik = loglik - 0.5D0 * norm2(zs_tru)**2 - log_det_l_zz

end subroutine
