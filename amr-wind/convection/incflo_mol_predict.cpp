#include "amr-wind/convection/incflo_convection_K.H"
#include "amr-wind/convection/MOL.H"
#include "amr-wind/utilities/bc_ops.H"

using namespace amrex;

void mol::predict_vels_on_faces(
    int lev,
    Box const& ubx,
    Box const& vbx,
    Box const& wbx,
    Array4<Real> const& u,
    Array4<Real> const& v,
    Array4<Real> const& w,
    Array4<Real const> const& vcc,
    Vector<BCRec> const& h_bcrec,
    BCRec const* d_bcrec,
    Vector<Geometry> geom)
{
    BL_PROFILE("amr-wind::mol::predict_vels_on_faces");

    const int ncomp =
        AMREX_SPACEDIM; // This is only used because h_bcrec and d_bcrec hold
                        // the bc's for all three velocity components

    const Box& domain_box = geom[lev].Domain();
    const int domain_ilo = domain_box.smallEnd(0);
    const int domain_ihi = domain_box.bigEnd(0);
    const int domain_jlo = domain_box.smallEnd(1);
    const int domain_jhi = domain_box.bigEnd(1);
    const int domain_klo = domain_box.smallEnd(2);
    const int domain_khi = domain_box.bigEnd(2);

    // At an ext_dir boundary, the boundary value is on the face, not cell
    // center.

    auto extdir_lohi = amr_wind::utils::has_extdir_or_ho(
        h_bcrec.data(), ncomp, static_cast<int>(Direction::x));
    bool has_extdir_lo = extdir_lohi.first;
    bool has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_ilo >= ubx.smallEnd(0) - 1) or
        (has_extdir_hi and domain_ihi <= ubx.bigEnd(0))) {
        amrex::ParallelFor(
            ubx, [vcc, domain_ilo, domain_ihi, u,
                  d_bcrec] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Real u_val = c1 * (vcc(i - 1, j, k, 0) + vcc(i, j, k, 0)) +
                             c2 * (vcc(i - 2, j, k, 0) + vcc(i + 1, j, k, 0));

                if (i == domain_ilo && (d_bcrec[0].lo(0) == BCType::ext_dir)) {
                    u_val = vcc(i - 1, j, k, 0);
                } else if (
                    i == domain_ihi + 1 &&
                    (d_bcrec[0].hi(0) == BCType::ext_dir)) {
                    u_val = vcc(i, j, k, 0);
                }

                u(i, j, k) = u_val;
            });
    } else {
        amrex::ParallelFor(
            ubx, [vcc, u] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                u(i, j, k) = c1 * (vcc(i - 1, j, k, 0) + vcc(i, j, k, 0)) +
                             c2 * (vcc(i - 2, j, k, 0) + vcc(i + 1, j, k, 0));
            });
    }

    extdir_lohi = amr_wind::utils::has_extdir_or_ho(
        h_bcrec.data(), ncomp, static_cast<int>(Direction::y));
    has_extdir_lo = extdir_lohi.first;
    has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_jlo >= vbx.smallEnd(1) - 1) or
        (has_extdir_hi and domain_jhi <= vbx.bigEnd(1))) {
        amrex::ParallelFor(
            vbx, [vcc, domain_jlo, domain_jhi, v,
                  d_bcrec] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Real v_val = c1 * (vcc(i, j - 1, k, 1) + vcc(i, j, k, 1)) +
                             c2 * (vcc(i, j - 2, k, 1) + vcc(i, j + 1, k, 1));

                if (j == domain_jlo && (d_bcrec[1].lo(1) == BCType::ext_dir)) {
                    v_val = vcc(i, j - 1, k, 1);
                } else if (
                    j == domain_jhi + 1 &&
                    (d_bcrec[1].hi(1) == BCType::ext_dir)) {
                    v_val = vcc(i, j, k, 1);
                }

                v(i, j, k) = v_val;
            });
    } else {
        amrex::ParallelFor(
            vbx, [vcc, v] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                v(i, j, k) = c1 * (vcc(i, j - 1, k, 1) + vcc(i, j, k, 1)) +
                             c2 * (vcc(i, j - 2, k, 1) + vcc(i, j + 1, k, 1));
            });
    }

    extdir_lohi = amr_wind::utils::has_extdir_or_ho(
        h_bcrec.data(), ncomp, static_cast<int>(Direction::z));
    has_extdir_lo = extdir_lohi.first;
    has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_klo >= wbx.smallEnd(2) - 1) or
        (has_extdir_hi and domain_khi <= wbx.bigEnd(2))) {
        amrex::ParallelFor(
            wbx, [vcc, domain_klo, domain_khi, w,
                  d_bcrec] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Real w_val = c1 * (vcc(i, j, k - 1, 2) + vcc(i, j, k, 2)) +
                             c2 * (vcc(i, j, k - 2, 2) + vcc(i, j, k + 1, 2));

                if (k == domain_klo && (d_bcrec[2].lo(2) == BCType::ext_dir)) {
                    w_val = vcc(i, j, k - 1, 2);
                } else if (
                    k == domain_khi + 1 &&
                    (d_bcrec[2].hi(2) == BCType::ext_dir)) {
                    w_val = vcc(i, j, k, 2);
                }

                w(i, j, k) = w_val;
            });
    } else {
        amrex::ParallelFor(
            wbx, [vcc, w] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                w(i, j, k) = c1 * (vcc(i, j, k - 1, 2) + vcc(i, j, k, 2)) +
                             c2 * (vcc(i, j, k - 2, 2) + vcc(i, j, k + 1, 2));
            });
    }
}
