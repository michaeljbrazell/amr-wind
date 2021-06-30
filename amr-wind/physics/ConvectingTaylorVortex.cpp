#include "amr-wind/physics/ConvectingTaylorVortex.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "amr-wind/utilities/trig_ops.H"
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_FillPatchUtil.H>

namespace amr_wind {
namespace ctv {

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real UExact::operator()(
    const amrex::Real u0,
    const amrex::Real v0,
    const amrex::Real omega,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real t) const
{
    return u0 - std::cos(utils::pi() * (x - u0 * t)) *
                    std::sin(utils::pi() * (y - v0 * t)) *
                    std::exp(-2.0 * omega * t);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real VExact::operator()(
    const amrex::Real u0,
    const amrex::Real v0,
    const amrex::Real omega,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real t) const
{
    return v0 + std::sin(utils::pi() * (x - u0 * t)) *
                    std::cos(utils::pi() * (y - v0 * t)) *
                    std::exp(-2.0 * omega * t);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real WExact::operator()(
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real) const
{
    return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpxExact::operator()(
    const amrex::Real u0,
    const amrex::Real,
    const amrex::Real omega,
    const amrex::Real x,
    const amrex::Real,
    const amrex::Real t) const
{
    return 0.5 * amr_wind::utils::pi() *
           std::sin(2.0 * amr_wind::utils::pi() * (x - u0 * t)) *
           std::exp(-4.0 * omega * t);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpyExact::operator()(
    const amrex::Real,
    const amrex::Real v0,
    const amrex::Real omega,
    const amrex::Real,
    const amrex::Real y,
    const amrex::Real t) const
{
    return 0.5 * amr_wind::utils::pi() *
           std::sin(2.0 * amr_wind::utils::pi() * (y - v0 * t)) *
           std::exp(-4.0 * omega * t);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpzExact::operator()(
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real,
    const amrex::Real) const
{
    return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real VorticityExact::operator()(
    const amrex::Real u0,
    const amrex::Real v0,
    const amrex::Real omega,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real t) const
{

    const amrex::Real dvdx =   utils::pi() * std::cos(utils::pi() * (x - u0 * t)) * std::cos(utils::pi() * (y - v0 * t)) * std::exp(-2.0 * omega * t);
    const amrex::Real dudy = - utils::pi() * std::cos(utils::pi() * (x - u0 * t)) * std::cos(utils::pi() * (y - v0 * t)) * std::exp(-2.0 * omega * t);
    return dvdx - dudy;

}
} // namespace

ConvectingTaylorVortex::ConvectingTaylorVortex(const CFDSim& sim)
    : m_time(sim.time())
    , m_sim(sim)
    , m_repo(sim.repo())
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_gradp(sim.repo().get_field("gp"))
    , m_density(sim.repo().get_field("density"))
{
    sim.repo().declare_nd_field("vorticity", 3, 1, 1);
    sim.repo().declare_nd_field("streamfunction", 3, 1, 1);

    {
        amrex::ParmParse pp("CTV");
        pp.query("density", m_rho);
        pp.query("u0", m_u0);
        pp.query("v0", m_v0);
        pp.query("activate_pressure", m_activate_pressure);
        pp.query("error_log_file", m_output_fname);
    }
    {
        amrex::Real nu;
        amrex::ParmParse pp("transport");
        pp.query("viscosity", nu);
        m_omega = utils::pi() * utils::pi() * nu;
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f;
        f.open(m_output_fname.c_str());
        f << std::setw(m_w) << "time" << std::setw(m_w) << "L2_u"
          << std::setw(m_w) << "L2_v" << std::setw(m_w) << "L2_w"
          << std::setw(m_w) << "L2_gpx" << std::setw(m_w) << "L2_gpy"
          << std::setw(m_w) << "L2_gpz" << std::endl;
        f.close();
    }
}

/** Initialize the velocity and density fields at the beginning of the
 *  simulation.
 */
void ConvectingTaylorVortex::initialize_fields(
    int level, const amrex::Geometry& geom)
{
    using namespace utils;

    const auto u0 = m_u0;
    const auto v0 = m_v0;
    const auto omega = m_omega;
    const bool activate_pressure = m_activate_pressure;

    auto& velocity = m_velocity(level);
    auto& density = m_density(level);
    auto& pressure = m_repo.get_field("p")(level);
    auto& gradp = m_repo.get_field("gp")(level);

    density.setVal(m_rho);

    UExact u_exact;
    VExact v_exact;
    WExact w_exact;
    GpxExact gpx_exact;
    GpyExact gpy_exact;
    GpzExact gpz_exact;
    VorticityExact vorticity_exact;

    auto& vorticity = m_repo.get_field("vorticity");
    auto& streamfunction = m_repo.get_field("streamfunction");

    for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();

        const auto& nbx = mfi.nodaltilebox();
        auto vort = vorticity(level).array(mfi);

        amrex::ParallelFor(
            nbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + i * dx[0];
                const amrex::Real y = problo[1] + j * dx[1];
//
//            const amrex::Real v_omega = sin(x);
//            const amrex::Real r = sqrt(std::pow(x-0.0,2) + std::pow(y-0.0,2));
                vort(i, j, k, 0) = 0.0;
                vort(i, j, k, 1) = 0.0;
                vort(i, j, k, 2) = vorticity_exact(u0, v0, omega, x, y, 0.0);
            });

    }

    amrex::LPInfo info;
    auto& mesh = m_velocity.repo().mesh();
    amrex::MLNodeLaplacian linop({mesh.Geom(level)}, {mesh.boxArray(level)}, {mesh.DistributionMap(level)}, info, {}, 1.0);
//    linop.setDomainBC({amrex::LinOpBCType::Dirichlet,amrex::LinOpBCType::Dirichlet,amrex::LinOpBCType::Dirichlet},
//                      {amrex::LinOpBCType::Dirichlet,amrex::LinOpBCType::Dirichlet,amrex::LinOpBCType::Dirichlet});
    linop.setDomainBC({amrex::LinOpBCType::Periodic,amrex::LinOpBCType::Periodic,amrex::LinOpBCType::Periodic},
                      {amrex::LinOpBCType::Periodic,amrex::LinOpBCType::Periodic,amrex::LinOpBCType::Periodic});

    amrex::MLMG mlmg(linop);

    if(level == 0) {
        streamfunction(level).setVal(0.0,0,3,1);
    } else {
        amrex::PhysBCFunctNoOp bcnoop;
        amrex::Vector<amrex::BCRec> bcrec(1);
        amrex::InterpFromCoarseLevel(streamfunction(level), 0.0,
                                   streamfunction(level-1), 0, 0, 3,
                                   mesh.Geom(level-1), mesh.Geom(level),
                                   bcnoop, 0, bcnoop, 0,
                                   amrex::IntVect{2},
                                   & amrex::node_bilinear_interp,
                                   bcrec, 0);
    }

    for(int i=0;i<AMREX_SPACEDIM;++i){
        auto stream = streamfunction.subview(i);
        auto vort = vorticity.subview(i);
        mlmg.solve({&stream(level)}, {&vort(level)}, 1.0e-6, 0.0);
    }

    for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dxinv = geom.InvCellSizeArray();
        auto vel = velocity.array(mfi);
        auto psi = streamfunction(level).array(mfi);
        const amrex::Real facx = amrex::Real(0.25)*dxinv[0];
        const amrex::Real facy = amrex::Real(0.25)*dxinv[1];
        const amrex::Real facz = amrex::Real(0.25)*dxinv[2];

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                //const amrex::Real dpsix_dx = facx * (-psi(i,j,k,0)+psi(i+1,j,k,0)-psi(i,j+1,k,0)+psi(i+1,j+1,k,0)-psi(i,j,k+1,0)+psi(i+1,j,k+1,0)-psi(i,j+1,k+1,0)+psi(i+1,j+1,k+1,0));
                const amrex::Real dpsix_dy = facy * (-psi(i,j,k,0)-psi(i+1,j,k,0)+psi(i,j+1,k,0)+psi(i+1,j+1,k,0)-psi(i,j,k+1,0)-psi(i+1,j,k+1,0)+psi(i,j+1,k+1,0)+psi(i+1,j+1,k+1,0));
                const amrex::Real dpsix_dz = facz * (-psi(i,j,k,0)-psi(i+1,j,k,0)-psi(i,j+1,k,0)-psi(i+1,j+1,k,0)+psi(i,j,k+1,0)+psi(i+1,j,k+1,0)+psi(i,j+1,k+1,0)+psi(i+1,j+1,k+1,0));
                const amrex::Real dpsiy_dx = facx * (-psi(i,j,k,1)+psi(i+1,j,k,1)-psi(i,j+1,k,1)+psi(i+1,j+1,k,1)-psi(i,j,k+1,1)+psi(i+1,j,k+1,1)-psi(i,j+1,k+1,1)+psi(i+1,j+1,k+1,1));
                //const amrex::Real dpsiy_dy = facy * (-psi(i,j,k,1)-psi(i+1,j,k,1)+psi(i,j+1,k,1)+psi(i+1,j+1,k,1)-psi(i,j,k+1,1)-psi(i+1,j,k+1,1)+psi(i,j+1,k+1,1)+psi(i+1,j+1,k+1,1));
                const amrex::Real dpsiy_dz = facz * (-psi(i,j,k,1)-psi(i+1,j,k,1)-psi(i,j+1,k,1)-psi(i+1,j+1,k,1)+psi(i,j,k+1,1)+psi(i+1,j,k+1,1)+psi(i,j+1,k+1,1)+psi(i+1,j+1,k+1,1));
                const amrex::Real dpsiz_dx = facx * (-psi(i,j,k,2)+psi(i+1,j,k,2)-psi(i,j+1,k,2)+psi(i+1,j+1,k,2)-psi(i,j,k+1,2)+psi(i+1,j,k+1,2)-psi(i,j+1,k+1,2)+psi(i+1,j+1,k+1,2));
                const amrex::Real dpsiz_dy = facy * (-psi(i,j,k,2)-psi(i+1,j,k,2)+psi(i,j+1,k,2)+psi(i+1,j+1,k,2)-psi(i,j,k+1,2)-psi(i+1,j,k+1,2)+psi(i,j+1,k+1,2)+psi(i+1,j+1,k+1,2));
                //const amrex::Real dpsiz_dz = facz * (-psi(i,j,k,2)-psi(i+1,j,k,2)-psi(i,j+1,k,2)-psi(i+1,j+1,k,2)+psi(i,j,k+1,2)+psi(i+1,j,k+1,2)+psi(i,j+1,k+1,2)+psi(i+1,j+1,k+1,2));

                vel(i, j, k, 0) = u0 - (dpsiz_dy - dpsiy_dz);
                vel(i, j, k, 1) = v0 - (dpsix_dz - dpsiz_dx);
                vel(i, j, k, 2) =    - (dpsiy_dx - dpsix_dy);

            });
    }

}

template <typename T>
amrex::Real ConvectingTaylorVortex::compute_error(const Field& field)
{

    amrex::Real error = 0.0;
    const amrex::Real time = m_time.new_time();
    const auto u0 = m_u0;
    const auto v0 = m_v0;
    const auto omega = m_omega;
    T f_exact;
    const auto comp = f_exact.m_comp;

    const int nlevels = m_repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {

        amrex::iMultiFab level_mask;
        if (lev < nlevels - 1) {
            level_mask = makeFineMask(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev),
                m_mesh.boxArray(lev + 1), amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev), 1, 0,
                amrex::MFInfo());
            level_mask.setVal(1);
        }

        if (m_sim.has_overset()) {
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& vbx = mfi.validbox();

                const auto& iblank_arr =
                    m_repo.get_int_field("iblank_cell")(lev).array(mfi);
                const auto& imask_arr = level_mask.array(mfi);
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        if (iblank_arr(i, j, k) < 1) imask_arr(i, j, k) = 0;
                    });
            }
        }

        const auto& dx = m_mesh.Geom(lev).CellSizeArray();
        const auto& problo = m_mesh.Geom(lev).ProbLoArray();
        const amrex::Real cell_vol = dx[0] * dx[1] * dx[2];

        const auto& fld = field(lev);
        error += amrex::ReduceSum(
            fld, level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& fld_arr,
                amrex::Array4<int const> const& mask_arr) -> amrex::Real {
                amrex::Real err_fab = 0.0;

                amrex::Loop(bx, [=, &err_fab](int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                    const amrex::Real u = fld_arr(i, j, k, comp);
                    const amrex::Real u_exact =
                        f_exact(u0, v0, omega, x, y, time);
                    err_fab += cell_vol * mask_arr(i, j, k) * (u - u_exact) *
                               (u - u_exact);
                });
                return err_fab;
            });
    }

    amrex::ParallelDescriptor::ReduceRealSum(error);

    const amrex::Real total_vol = m_mesh.Geom(0).ProbDomain().volume();
    return std::sqrt(error / total_vol);
}

void ConvectingTaylorVortex::output_error()
{
    const amrex::Real u_err = compute_error<UExact>(m_velocity);
    const amrex::Real v_err = compute_error<VExact>(m_velocity);
    const amrex::Real w_err = compute_error<WExact>(m_velocity);
    const amrex::Real gpx_err = compute_error<GpxExact>(m_gradp);
    const amrex::Real gpy_err = compute_error<GpyExact>(m_gradp);
    const amrex::Real gpz_err = compute_error<GpzExact>(m_gradp);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f;
        f.open(m_output_fname.c_str(), std::ios_base::app);
        f << std::setprecision(12) << std::setw(m_w) << m_time.new_time()
          << std::setw(m_w) << u_err << std::setw(m_w) << v_err
          << std::setw(m_w) << w_err << std::setw(m_w) << gpx_err
          << std::setw(m_w) << gpy_err << std::setw(m_w) << gpz_err
          << std::endl;
        f.close();
    }
}

void ConvectingTaylorVortex::post_init_actions() { output_error(); }

void ConvectingTaylorVortex::post_advance_work() { output_error(); }

} // namespace ctv
} // namespace amr_wind
