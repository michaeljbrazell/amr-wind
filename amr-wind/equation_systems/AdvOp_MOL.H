#ifndef ADVOP_MOL_H
#define ADVOP_MOL_H

#include <type_traits>

#include "amr-wind/convection/MOL.H"
#include "amr-wind/equation_systems/SchemeTraits.H"
#include "amr-wind/equation_systems/PDETraits.H"
#include "amr-wind/equation_systems/PDEOps.H"

#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace pde {

/** MOL advection operator for scalar transport equations
 *  \ingroup pdeop
 */
template <typename PDE>
struct AdvectionOp<
    PDE,
    fvm::MOL,
    typename std::enable_if<std::is_base_of<ScalarTransport, PDE>::value>::type>
{
    AdvectionOp(
        PDEFields& fields_in,
        bool /* has_overset */,
        bool /* variable density */)
        : fields(fields_in)
        , density(fields_in.repo.get_field("density"))
        , u_mac(fields_in.repo.get_field("u_mac"))
        , v_mac(fields_in.repo.get_field("v_mac"))
        , w_mac(fields_in.repo.get_field("w_mac"))
    {}

    void operator()(const FieldState fstate, const amrex::Real)
    {
        static_assert(
            PDE::ndim == 1, "Invalid number of components for scalar");

        auto& repo = fields.repo;
        auto& geom = repo.mesh().Geom();

        auto& conv_term = fields.conv_term.state(fstate);
        auto& dof_field = fields.field.state(fstate);
        const auto& den = density.state(fstate);

        auto flux_x =
            repo.create_scratch_field(PDE::ndim, 0, amr_wind::FieldLoc::XFACE);
        auto flux_y =
            repo.create_scratch_field(PDE::ndim, 0, amr_wind::FieldLoc::YFACE);
        auto flux_z =
            repo.create_scratch_field(PDE::ndim, 0, amr_wind::FieldLoc::ZFACE);

        for (int lev = 0; lev < repo.num_active_levels(); ++lev) {
            amrex::MFItInfo mfi_info;
            // if (amrex::Gpu::notInLaunchRegion())
            // mfi_info.EnableTiling(amrex::IntVect(1024,16,16)).SetDynamic(true);
            if (amrex::Gpu::notInLaunchRegion())
                mfi_info.EnableTiling(amrex::IntVect(1024, 1024, 1024))
                    .SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(density(lev), mfi_info); mfi.isValid();
                 ++mfi) {

                amrex::Box const& bx = mfi.tilebox();
                auto rho_arr = den(lev).const_array(mfi);
                auto tra_arr = dof_field(lev).const_array(mfi);
                amrex::FArrayBox rhotracfab;
                amrex::Elixir eli_rt;
                amrex::Array4<amrex::Real> rhotrac;

                if (PDE::multiply_rho) {

                    amrex::Box rhotrac_box =
                        amrex::grow(bx, fvm::MOL::nghost_state);
                    rhotracfab.resize(rhotrac_box, PDE::ndim);
                    eli_rt = rhotracfab.elixir();
                    rhotrac = rhotracfab.array();

                    amrex::ParallelFor(
                        rhotrac_box, PDE::ndim,
                        [=] AMREX_GPU_DEVICE(
                            int i, int j, int k, int n) noexcept {
                            rhotrac(i, j, k, n) =
                                rho_arr(i, j, k) * tra_arr(i, j, k, n);
                        });
                }

                {

                    mol::compute_convective_fluxes(
                        lev, bx, PDE::ndim, (*flux_x)(lev).array(mfi),
                        (*flux_y)(lev).array(mfi), (*flux_z)(lev).array(mfi),
                        (PDE::multiply_rho ? rhotrac : tra_arr),
                        u_mac(lev).const_array(mfi),
                        v_mac(lev).const_array(mfi),
                        w_mac(lev).const_array(mfi), dof_field.bcrec().data(),
                        dof_field.bcrec_device().data(), geom);
                }
            }
        }

        amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fluxes(
            repo.num_active_levels());
        for (int lev = 0; lev < repo.num_active_levels(); ++lev) {
            fluxes[lev][0] = &(*flux_x)(lev);
            fluxes[lev][1] = &(*flux_y)(lev);
            fluxes[lev][2] = &(*flux_z)(lev);
        }

        // In order to enforce conservation across coarse-fine boundaries we
        // must be sure to average down the fluxes before we use them
        for (int lev = repo.num_active_levels() - 1; lev > 0; --lev) {
            amrex::IntVect rr =
                geom[lev].Domain().size() / geom[lev - 1].Domain().size();
            amrex::average_down_faces(
                GetArrOfConstPtrs(fluxes[lev]), fluxes[lev - 1], rr,
                geom[lev - 1]);
        }

        for (int lev = 0; lev < repo.num_active_levels(); ++lev) {
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(dof_field(lev), amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                mol::compute_convective_rate(
                    bx, PDE::ndim, conv_term(lev).array(mfi),
                    (*flux_x)(lev).array(mfi), (*flux_y)(lev).array(mfi),
                    (*flux_z)(lev).array(mfi), geom[lev].InvCellSizeArray());
            }
        }
    }

    PDEFields& fields;
    Field& density;
    Field& u_mac;
    Field& v_mac;
    Field& w_mac;
};

} // namespace pde
} // namespace amr_wind

#endif /* ADVOP_MOL_H */
