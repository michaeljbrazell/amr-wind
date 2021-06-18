#ifndef POWERLAWPROFILE_H
#define POWERLAWPROFILE_H

#include "AMReX_Geometry.H"
#include "AMReX_Gpu.H"
#include "amr-wind/utilities/trig_ops.H"

namespace amr_wind {

class Field;

namespace udf {

struct PowerLawProfile
{
    struct DeviceOp
    {
        int idir{2};
        amrex::Real zoffset{0.0};
        amrex::Real zref;
        amrex::Real uref;
        amrex::Real shear_exp;
        amrex::Real umin{0.0};
        amrex::Real umax{100.0};
        amrex::Real uvec[AMREX_SPACEDIM];

        AMREX_GPU_DEVICE
        inline void operator()(
            const amrex::IntVect& iv,
            amrex::Array4<amrex::Real> const& field,
            amrex::GeometryData const& geom,
            const amrex::Real,
            amrex::Orientation,
            const int comp) const
        {
            const auto* problo = geom.ProbLo();
            const auto* dx = geom.CellSize();
            const auto zco = problo[idir] + (iv[idir] + 0.5) * dx[idir];

            const amrex::Real zeff = zco - zoffset;
            amrex::Real pfac =
                (zeff > 0.0) ? std::pow((zeff / zref), shear_exp) : 0.0;
            pfac = amrex::min(amrex::max(pfac, umin), umax);

            field(iv[0], iv[1], iv[2], comp) = uvec[comp] * pfac;
        }
    };

    using DeviceType = DeviceOp;

    static std::string identifier() { return "PowerLawProfile"; }

    explicit PowerLawProfile(const Field& fld);

    DeviceType device_instance() const { return m_op; }

    DeviceOp m_op;
};


struct CTVProfile
{
    struct DeviceOp
    {
        amrex::Real u0{1.0};
        amrex::Real v0{1.0};
        amrex::Real omega{1.0};

        AMREX_GPU_DEVICE
        inline void operator()(
            const amrex::IntVect& iv,
            amrex::Array4<amrex::Real> const& field,
            amrex::GeometryData const& geom,
            const amrex::Real t,
            amrex::Orientation,
            const int comp) const
        {

            const auto* problo = geom.ProbLo();
            const auto* dx = geom.CellSize();

            const auto x = problo[0] + (iv[0] + 0.5) * dx[0];
            const auto y = problo[1] + (iv[1] + 0.5) * dx[1];
            const auto z = problo[2] + (iv[2] + 0.5) * dx[2];

            amrex::Real vel[3];

            vel[0] = u0 - std::cos(utils::pi() * (x - u0 * t)) *
            std::sin(utils::pi() * (y - v0 * t)) *
            std::exp(-2.0 * omega * t);

            vel[1] = v0 + std::sin(utils::pi() * (x - u0 * t)) *
            std::cos(utils::pi() * (y - v0 * t)) *
            std::exp(-2.0 * omega * t);

            vel[2] = 0.0;

            field(iv[0], iv[1], iv[2], comp) = vel[comp];
        }
    };

    using DeviceType = DeviceOp;

    static std::string identifier() { return "CTVProfile"; }

    explicit CTVProfile(const Field& fld);

    DeviceType device_instance() const { return m_op; }

    DeviceOp m_op;
};

} // namespace udf
} // namespace amr_wind

#endif /* POWERLAWPROFILE_H */
