#include <cmath>

#include "amr-wind/incflo.H"
#include "amr-wind/core/Physics.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/convection/Godunov.H"
#include "amr-wind/convection/MOL.H"
#include "amr-wind/convection/mac_projection.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/diffusion/diffusion.H"
#include "amr-wind/turbulence/TurbulenceModel.H"
#include "amr-wind/utilities/console_io.H"
#include "amr-wind/utilities/PostProcessing.H"
#include "amr-wind/core/field_ops.H"
#include "amr-wind/utilities/trig_ops.H"

using namespace amrex;


void incflo::update_velocity(amrex::Real u0, amrex::Real v0, amrex::Real omega, amrex::Real t, amr_wind::FieldState fstate){

    if (!m_sim.has_overset()) return;

    auto& vel = velocity().state(fstate);
    auto& iblank_cell = m_repo.get_int_field("iblank_cell");

    amrex::Real dummy = 0.0;
    // amr_wind::ctv::UExact uexact;

    for (int lev = 0; lev < m_repo.num_active_levels(); ++lev) {
        const auto& dx = Geom(lev).CellSizeArray();
        const auto& problo = Geom(lev).ProbLoArray();

        for (amrex::MFIter mfi(vel(lev)); mfi.isValid(); ++mfi)
        {
            amrex::Box bx = mfi.validbox();

            auto varr = vel(lev).array(mfi);
            auto ibcarr = iblank_cell(lev).array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                if(ibcarr(i,j,k) == 0)
                {
                    varr(i,j,k,0) = dummy;
                    varr(i,j,k,1) = dummy;
                    varr(i,j,k,2) = dummy;
                }

                if(ibcarr(i,j,k) < 1) // to override above
//                  if(ibcarr(i,j,k) == -1)
                {

                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5) * dx[1];

    //                varr(i,j,k,0) = uexact(u0, v0, omega, x, y, t);
                    varr(i,j,k,0) = u0 - std::cos(amr_wind::utils::pi() * (x - u0 * t)) * std::sin(amr_wind::utils::pi() * (y - v0 * t)) * std::exp(-2.0 * omega * t);
                    varr(i,j,k,1) = v0 + std::sin(amr_wind::utils::pi() * (x - u0 * t)) * std::cos(amr_wind::utils::pi() * (y - v0 * t)) * std::exp(-2.0 * omega * t);

                }
            });

        }
    }
}


void incflo::update_pr(amrex::Real u0, amrex::Real v0, amrex::Real p0, amrex::Real omega, amrex::Real t){

    if (!m_sim.has_overset()) return;

    auto& pr = pressure();
    auto& iblank_node = m_repo.get_int_field("iblank_node");

    amrex::Real dummy = 0.0;

    for (int lev = 0; lev < m_repo.num_active_levels(); ++lev) {
        const auto& dx = Geom(lev).CellSizeArray();
        const auto& problo = Geom(lev).ProbLoArray();


        for (amrex::MFIter mfi(pr(lev)); mfi.isValid(); ++mfi)
        {
            amrex::Box bx = mfi.validbox();
            auto parr = pr(lev).array(mfi);
            auto ibnarr = iblank_node(lev).array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if(ibnarr(i,j,k) == 0)
                {
                    parr(i,j,k) = dummy;
                }

                if(ibnarr(i,j,k) < 1)
//              if(ibnarr(i,j,k) == -1)
                {
                    const amrex::Real x = problo[0] + i * dx[0];
                    const amrex::Real y = problo[1] + j * dx[1];

                    parr(i,j,k) = - 0.25*p0*(std::cos(2.0*amr_wind::utils::pi() * (x - u0 * t)) + std::cos(2.0*amr_wind::utils::pi() * (y - v0 * t))) * std::exp(-4.0 * omega * t);
                }
            });

        }
    }
}



/** Advance simulation state by one timestep
 *
 *  Performs the following actions at a given timestep
 *  - Compute \f$\Delta t\f$
 *  - Advance all computational fields to new timestate in preparation for time integration
 *  - Call pre-advance work for all registered physics modules
 *  - For Godunov scheme, advance to new time state
 *  - For MOL scheme, call predictor corrector steps
 *  - Perform any post-advance work
 *
 *  Much of the heavy-lifting is done by incflo::ApplyPredictor and
 *  incflo::ApplyCorrector. Please refer to the documentation of those methods
 *  for detailed information on the various equations being solved.
 *
 * \callgraph
 */
void incflo::advance()
{
    BL_PROFILE("amr-wind::incflo::Advance");

    if(m_sim.has_overset()){
        amrex::Real t = m_time.new_time();
        amrex::Real u0 = 1.0;
        amrex::Real v0 = 1.0;
        amrex::Real p0 = 1.0;

        amrex::Print() << "time: " << t << std::endl;

        {
            amrex::ParmParse pp("CTV");
            pp.query("u0", u0);
            pp.query("v0", v0);
        }
        amrex::Real omega = 0.0;
        {
            amrex::ParmParse pp("transport");
            amrex::Real nu;
            pp.query("viscosity", nu);
            omega = amr_wind::utils::pi() * amr_wind::utils::pi() * nu;
        }

        update_pr(u0, v0, p0, omega, t);
//        update_pr(u0, v0, p0, omega, t+0.5*m_time.deltaT());
        update_velocity(u0, v0, omega, t, amr_wind::FieldState::New);

    }

    // Compute time step size
    bool explicit_diffusion = (m_diff_type == DiffusionType::Explicit);
    ComputeDt(explicit_diffusion);

    if (m_constant_density) {
        density().advance_states();
        density().state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());
    }

    auto& vel = icns().fields().field;
    vel.advance_states();
    vel.state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());
    for (auto& eqn: scalar_eqns()) {
        auto& field = eqn->fields().field;
        field.advance_states();
        field.state(amr_wind::FieldState::Old).fillpatch(m_time.current_time());
    }

    for (auto& pp: m_sim.physics())
        pp->pre_advance_work();

    ApplyPredictor();

    if (!m_use_godunov) {
        vel.state(amr_wind::FieldState::New).fillpatch(m_time.current_time());
        for (auto& eqn: scalar_eqns()) {
            auto& field = eqn->fields().field;
            field.state(amr_wind::FieldState::New).fillpatch(m_time.current_time());
        }

        ApplyCorrector();
    }

    for (auto& pp: m_sim.physics())
        pp->post_advance_work();

    m_sim.post_manager().post_advance_work();
    if (m_verbose > 1) PrintMaxValues("end of timestep");
}

// Apply predictor step
//
//  For Godunov, this completes the timestep. For MOL, this is the first part of
//  the predictor/corrector within a timestep.
//
//  <ol>
//  <li> Use u = vel_old to compute
//
//     \code{.cpp}
//     conv_u  = - u grad u
//     conv_r  = - div( u rho  )
//     conv_t  = - div( u trac )
//     eta_old     = visosity at m_time.current_time()
//     if (m_diff_type == DiffusionType::Explicit)
//        divtau _old = div( eta ( (grad u) + (grad u)^T ) ) / rho^n
//        rhs = u + dt * ( conv + divtau_old )
//     else
//        divtau_old  = 0.0
//        rhs = u + dt * conv
//
//     eta     = eta at new_time
//     \endcode
//
//  <li> Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//     \code{.cpp}
//     rhs += dt * ( g - grad(p + p0) / rho^nph )
//     \endcode
//
//  Note that in order to add the pressure gradient terms divided by rho,
//  we convert the velocity to momentum before adding and then convert them back.
//
//  <li> A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//  \code{.cpp}
//  ( 1 - dt / rho^nph * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                               + dt * ( g - grad(p + p0) / rho^nph )
//  \endcode
//
//  B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//     solve semi-implicit diffusion equation for u*
//
//     \code{.cpp}
//     ( 1 - (dt/2) / rho^nph * div ( eta_old grad ) ) u* = u^n +
//            dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//          + dt * ( g - grad(p + p0) / rho^nph )
//     \endcode
//
//  <li> Apply projection (see incflo::ApplyProjection)
//
//     Add pressure gradient term back to u*:
//
//      \code{.cpp}
//      u** = u* + dt * grad p / rho^nph
//      \endcode
//
//     Solve Poisson equation for phi:
//
//     \code{.cpp}
//     div( grad(phi) / rho^nph ) = div( u** )
//     \endcode
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho^nph
//  </ol>
//
// It is assumed that the ghost cels of the old data have been filled and
// the old and new data are the same in valid region.
//

/** Apply predictor step
 *
 *  For Godunov, this completes the timestep. For MOL, this is the first part of
 *  the predictor/corrector within a timestep.
 *
 *  <ol>
 *  <li> Solve transport equation for momentum and scalars
 *
 *  \f{align}
 *  \left[1 - \kappa \frac{\Delta t}{\rho^{n+1/2}} \nabla \cdot \left( \mu
 *  \nabla \right)\right] u^{*} &= u^n - \Delta t (u \cdot \nabla) u + (1 - \kappa) \frac{\Delta t}{\rho^n} \nabla \cdot \left( \mu^{n} \nabla\right) u^{n} + \frac{\Delta t}{\rho^{n+1/2}} \left( S_u - \nabla(p + p_0)\right) \\
 *  \f}
 *
 *  where
 *  \f{align}
 *  \kappa = \begin{cases}
 *  0 & \text{Explicit} \\
 *  0.5 & \text{Crank-Nicholson} \\
 *  1 & \text{Implicit}
 *  \end{cases}
 *  \f}
 *
 *  <li> \ref incflo::ApplyProjection "Apply projection"
 *  </ol>
 */
void incflo::ApplyPredictor (bool incremental_projection)
{
    BL_PROFILE("amr-wind::incflo::ApplyPredictor");

    // We use the new time value for things computed on the "*" state
    Real new_time = m_time.new_time();

    if (m_verbose > 2) PrintMaxValues("before predictor step");

    if (m_use_godunov)
        amr_wind::io::print_mlmg_header("Godunov:");
    else
        amr_wind::io::print_mlmg_header("Predictor:");

    auto& icns_fields = icns().fields();
    auto& velocity_new = icns_fields.field;
    auto& velocity_old = velocity_new.state(amr_wind::FieldState::Old);
    auto& density_new = density();
    auto& density_old = density_new.state(amr_wind::FieldState::Old);
    auto& density_nph = density_new.state(amr_wind::FieldState::NPH);

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    m_sim.turbulence_model().update_turbulent_viscosity(amr_wind::FieldState::Old);
    icns().compute_mueff(amr_wind::FieldState::Old);
    for (auto& eqns: scalar_eqns())
        eqns->compute_mueff(amr_wind::FieldState::Old);
    
    // *************************************************************************************
    // Define the forcing terms to use in the Godunov prediction
    // *************************************************************************************
    if (m_use_godunov)
    {
        icns().compute_source_term(amr_wind::FieldState::Old);
        for (auto& seqn: scalar_eqns()) {
            seqn->compute_source_term(amr_wind::FieldState::Old);
        }
    }

    // *************************************************************************************
    // Compute explicit viscous term
    // *************************************************************************************
    if (need_divtau()) {
        // Reuse existing buffer to avoid creating new multifabs
        amr_wind::field_ops::copy(velocity_new, velocity_old, 0, 0, velocity_new.num_comp(), 1);
        icns().compute_diffusion_term(amr_wind::FieldState::Old);
        if (m_use_godunov) {
            auto& velocity_forces = icns_fields.src_term;
            // only the old states are used in predictor
            auto& divtau = m_use_godunov
                       ? icns_fields.diff_term
                       : icns_fields.diff_term.state(amr_wind::FieldState::Old);

            amr_wind::field_ops::add(velocity_forces, divtau, 0, 0, AMREX_SPACEDIM, 0);
        }
    }



    // *************************************************************************************
    // Compute explicit diffusive terms
    // *************************************************************************************
    if (need_divtau()) {
        for (auto& eqn: scalar_eqns()) {
            auto& field = eqn->fields().field;
            // Reuse existing buffer to avoid creating new multifabs
            amr_wind::field_ops::copy(field, field.state(amr_wind::FieldState::Old),
                                      0, 0, field.num_comp(), 1);

            eqn->compute_diffusion_term(amr_wind::FieldState::Old);

            if (m_use_godunov)
                amr_wind::field_ops::add(
                    eqn->fields().src_term,
                    eqn->fields().diff_term, 0, 0,
                    field.num_comp(), 0);
        }
    }

    if (m_use_godunov) {
        const int nghost_force = 1;
        IntVect ng(nghost_force);
        icns().fields().src_term.fillpatch(m_time.current_time(), ng);

        for (auto& eqn: scalar_eqns()) {
            eqn->fields().src_term.fillpatch(m_time.current_time(), ng);
        }
    }

    // *************************************************************************************
    // if ( m_use_godunov) Compute the explicit advective terms
    //                     R_u^(n+1/2), R_s^(n+1/2) and R_t^(n+1/2)
    // if (!m_use_godunov) Compute the explicit advective terms
    //                     R_u^n      , R_s^n       and R_t^n
    // *************************************************************************************
    icns().compute_advection_term(amr_wind::FieldState::Old);
    for (auto& seqn: scalar_eqns()) {
        seqn->compute_advection_term(amr_wind::FieldState::Old);
    }

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (m_constant_density)
    {
        amr_wind::field_ops::copy(density_nph, density_old, 0, 0, 1, 1);
    }

    // Perform scalar update one at a time. This is to allow an updated density
    // at `n+1/2` to be computed before other scalars use it when computing
    // their source terms.
    for (auto& eqn: scalar_eqns()) {
        // Compute (recompute for Godunov) the scalar forcing terms
        eqn->compute_source_term(amr_wind::FieldState::NPH);

        // Update the scalar (if explicit), or the RHS for implicit/CN
        eqn->compute_predictor_rhs(m_diff_type);

        auto& field = eqn->fields().field;
        if (m_diff_type != DiffusionType::Explicit) {
            amrex::Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                ? m_time.deltaT() : 0.5 * m_time.deltaT();

            // Solve diffusion eqn. and update of the scalar field
            eqn->solve(dt_diff);

            // Post-processing actions after a PDE solve
        }
        eqn->post_solve_actions();

        // Update scalar at n+1/2
        amr_wind::field_ops::lincomb(
            field.state(amr_wind::FieldState::NPH),
            0.5, field.state(amr_wind::FieldState::Old), 0,
            0.5, field, 0, 0, field.num_comp(), 1);
    }

    // *************************************************************************************
    // Define (or if use_godunov, re-define) the forcing terms, without the viscous terms
    //    and using the half-time density
    // *************************************************************************************
    icns().compute_source_term(amr_wind::FieldState::New);

    // *************************************************************************************
    // Update the velocity
    // *************************************************************************************
    icns().compute_predictor_rhs(m_diff_type);

    // *************************************************************************************
    // Solve diffusion equation for u* but using eta_old at old time
    // *************************************************************************************
    if (m_diff_type == DiffusionType::Crank_Nicolson ||
        m_diff_type == DiffusionType::Implicit) {
        Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                           ? m_time.deltaT()
                           : 0.5 * m_time.deltaT();
        icns().solve(dt_diff);
    }
    icns().post_solve_actions();

    // ************************************************************************************
    //
    // Project velocity field, update pressure
    //
    // ************************************************************************************
    ApplyProjection(
        (density_nph).vec_const_ptrs(), new_time, m_time.deltaT(),
        incremental_projection);
}


//
// Apply corrector:
//
//  Output variables from the predictor are labelled _pred
//
//  1. Use u = vel_pred to compute
//
//      conv_u  = - u grad u
//      conv_r  = - u grad rho
//      conv_t  = - u grad trac
//      eta     = viscosity
//      divtau  = div( eta ( (grad u) + (grad u)^T ) ) / rho
//
//      conv_u  = 0.5 (conv_u + conv_u_pred)
//      conv_r  = 0.5 (conv_r + conv_r_pred)
//      conv_t  = 0.5 (conv_t + conv_t_pred)
//      if (m_diff_type == DiffusionType::Explicit)
//         divtau  = divtau at new_time using (*) state
//      else
//         divtau  = 0.0
//      eta     = eta at new_time
//
//     rhs = u + dt * ( conv + divtau )
//
//  2. Add explicit forcing term i.e. gravity + lagged pressure gradient
//
//      rhs += dt * ( g - grad(p + p0) / rho )
//
//      Note that in order to add the pressure gradient terms divided by rho,
//      we convert the velocity to momentum before adding and then convert them back.
//
//  3. A. If (m_diff_type == DiffusionType::Implicit)
//        solve implicit diffusion equation for u*
//
//     ( 1 - dt / rho * div ( eta grad ) ) u* = u^n + dt * conv_u
//                                                  + dt * ( g - grad(p + p0) / rho )
//
//     B. If (m_diff_type == DiffusionType::Crank-Nicolson)
//        solve semi-implicit diffusion equation for u*
//
//     ( 1 - (dt/2) / rho * div ( eta grad ) ) u* = u^n + dt * conv_u + (dt/2) / rho * div (eta_old grad) u^n
//                                                      + dt * ( g - grad(p + p0) / rho )
//
//  4. Apply projection
//
//     Add pressure gradient term back to u*:
//
//      u** = u* + dt * grad p / rho
//
//     Solve Poisson equation for phi:
//
//     div( grad(phi) / rho ) = div( u** )
//
//     Update pressure:
//
//     p = phi / dt
//
//     Update velocity, now divergence free
//
//     vel = u** - dt * grad p / rho
//

/** Corrector step for MOL scheme
 *
 *  <ol>
 *  <li> Solve transport equation for momentum and scalars
 *
 *  \f{align}
 *  \left[1 - \kappa \frac{\Delta t}{\rho} \nabla \cdot \left( \mu
 *  \nabla \right)\right] u^{*} &= u^n - \Delta t C_u + (1 - \kappa) \frac{\Delta t}{\rho} \nabla \cdot \left( \mu \nabla\right) u^{n} + \frac{\Delta t}{\rho} \left( S_u - \nabla(p + p_0)\right) \\
 *  \f}
 *
 *  where
 *  \f{align}
 *  \kappa = \begin{cases}
 *  0 & \text{Explicit} \\
 *  0.5 & \text{Crank-Nicholson} \\
 *  1 & \text{Implicit}
 *  \end{cases}
 *  \f}
 *
 *  <li> \ref incflo::ApplyProjection "Apply projection"
 *  </ol>
 */
void incflo::ApplyCorrector()
{
    BL_PROFILE("amr-wind::incflo::ApplyCorrector");

    // We use the new time value for things computed on the "*" state
    Real new_time = m_time.new_time();

    if (m_verbose > 2) PrintMaxValues("before corrector step");

    amr_wind::io::print_mlmg_header("Corrector:");

    auto& density_new = density();
    auto& density_old = density_new.state(amr_wind::FieldState::Old);
    auto& density_nph = density_new.state(amr_wind::FieldState::NPH);

    // *************************************************************************************
    // Compute the explicit "new" advective terms R_u^(n+1,*), R_r^(n+1,*) and R_t^(n+1,*)
    // We only reach the corrector if !m_use_godunov which means we don't use the forces
    // in constructing the advection term
    // *************************************************************************************
    icns().compute_advection_term(amr_wind::FieldState::New);
    for (auto& seqn: scalar_eqns()) {
        seqn->compute_advection_term(amr_wind::FieldState::New);
    }

    // *************************************************************************************
    // Compute viscosity / diffusive coefficients
    // *************************************************************************************
    m_sim.turbulence_model().update_turbulent_viscosity(amr_wind::FieldState::New);
    icns().compute_mueff(amr_wind::FieldState::New);
    for (auto& eqns: scalar_eqns())
        eqns->compute_mueff(amr_wind::FieldState::New);

    // Here we create divtau of the (n+1,*) state that was computed in the predictor;
    //      we use this laps only if DiffusionType::Explicit
    if (m_diff_type == DiffusionType::Explicit) {
        icns().compute_diffusion_term(amr_wind::FieldState::New);

        for (auto& eqns: scalar_eqns()) {
            eqns->compute_diffusion_term(amr_wind::FieldState::New);
        }
    }

    // *************************************************************************************
    // Update density first
    // *************************************************************************************
    if (m_constant_density) {
        amr_wind::field_ops::copy(density_nph, density_old, 0, 0, 1, 1);
    }

    // Perform scalar update one at a time. This is to allow an updated density
    // at `n+1/2` to be computed before other scalars use it when computing
    // their source terms.
    for (auto& eqn: scalar_eqns()) {
        // Compute (recompute for Godunov) the scalar forcing terms
        // Note this is (rho * scalar) and not just scalar
        eqn->compute_source_term(amr_wind::FieldState::New);

        // Update (note that dtdt already has rho in it)
        // (rho trac)^new = (rho trac)^old + dt * (
        //                   div(rho trac u) + div (mu grad trac) + rho * f_t
        eqn->compute_corrector_rhs(m_diff_type);

        auto& field = eqn->fields().field;
        if (m_diff_type != DiffusionType::Explicit) {
            amrex::Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                ? m_time.deltaT() : 0.5 * m_time.deltaT();

            // Solve diffusion eqn. and update of the scalar field
            eqn->solve(dt_diff);
        }
        eqn->post_solve_actions();

        // Update scalar at n+1/2
        amr_wind::field_ops::lincomb(
            field.state(amr_wind::FieldState::NPH),
            0.5, field.state(amr_wind::FieldState::Old), 0,
            0.5, field, 0, 0, field.num_comp(), 1);
    }

    // *************************************************************************************
    // Define the forcing terms to use in the final update (using half-time density)
    // *************************************************************************************
    icns().compute_source_term(amr_wind::FieldState::New);

    // *************************************************************************************
    // Update velocity
    // *************************************************************************************
    icns().compute_corrector_rhs(m_diff_type);

    // *************************************************************************************
    //
    // Solve diffusion equation for u* at t^{n+1} but using eta at predicted new time
    //
    // *************************************************************************************

    if (m_diff_type == DiffusionType::Crank_Nicolson ||
        m_diff_type == DiffusionType::Implicit) {
        Real dt_diff = (m_diff_type == DiffusionType::Implicit)
                           ? m_time.deltaT()
                           : 0.5 * m_time.deltaT();
        icns().solve(dt_diff);
    }
    icns().post_solve_actions();

    // *************************************************************************************
    // Project velocity field, update pressure
    // *************************************************************************************
    bool incremental = false;
    ApplyProjection((density_nph).vec_const_ptrs(),new_time, m_time.deltaT(), incremental);

}
