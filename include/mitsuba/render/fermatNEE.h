#pragma once

#include "fwd.h"
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

// EmitterInteraction structure (from SpecularManifoldSampling/manifold.h)
template <typename Float_, typename Spectrum_> struct EmitterInteraction {
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_RENDER_BASIC_TYPES()
    using EmitterPtr = typename RenderAliases::EmitterPtr;

    Point3f p; // Emitter position (for area / point)
    Normal3f n;
    Vector3f d; // Emitter direction (for infinite / directional )

    Spectrum weight; // Samping weight (already divided by positional
                     // sampling pdf)
    Float pdf;       // Sampling pdf

    EmitterPtr emitter = nullptr;

    bool is_point() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaPosition);
    }

    bool is_directional() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaDirection) ||
               has_flag(emitter->flags(), EmitterFlags::Infinite);
    }

    bool is_area() const {
        return has_flag(emitter->flags(), EmitterFlags::Surface);
    }

    bool is_delta() const {
        return has_flag(emitter->flags(), EmitterFlags::DeltaPosition) ||
               has_flag(emitter->flags(), EmitterFlags::DeltaDirection);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "EmitterInteraction[" << std::endl
            << "  p = " << p << "," << std::endl
            << "  n = " << n << "," << std::endl
            << "  d = " << d << "," << std::endl
            << "  weight = " << weight << "," << std::endl
            << "  pdf    = " << pdf << "," << std::endl
            << "]";
        return oss.str();
    }
};

template <typename Float_, typename Spectrum_>
class MTS_EXPORT_RENDER FermatNEE {
public:
    using Float    = Float_;
    using Spectrum = Spectrum_;
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter);
    using EmitterPtr         = typename RenderAliases::EmitterPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using EmitterInteraction = EmitterInteraction<Float, Spectrum>;

    // config structure
    struct FNEE_config {
        size_t max_bernouilli_trial;
        ScalarFloat solution_identical_threshold;
        ScalarFloat solver_gradient_threshold;
        ScalarFloat64 alpha;
        bool caustic_bleed;
    };
    struct L_data {
        ShapePtr pH1, pH2;
        Vector3f dimensions; // H L W
        Float e;
        Point3f O, S;
        Point2f ior; // eta inside, eta outside
    };
    FermatNEE() {}
    FermatNEE(FNEE_config &config) : m_config(config) {}

    void init(const Scene *scene, Sampler *sampler) { m_scene = scene; }

    Vector4d newton_solver_double(L_data const &data, Vector4f *X_out,
                              Float *f_out) const;

    Vector4d solve_4x4(const Matrix4d &m, const Vector4d &b) const {
        using Vector = Vector4d;
        return Vector4d(0);
        Vector col0 = m.coeff(0), col1 = m.coeff(1), col2 = m.coeff(2),
               col3 = m.coeff(3);

        col1 = shuffle<2, 3, 0, 1>(col1);
        col3 = shuffle<2, 3, 0, 1>(col3);

        Vector tmp, row0, row1, row2, row3;

        tmp  = shuffle<1, 0, 3, 2>(col2 * col3);
        row0 = col1 * tmp;
        row1 = col0 * tmp;
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row0 = fmsub(col1, tmp, row0);
        row1 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row1));

        tmp  = shuffle<1, 0, 3, 2>(col1 * col2);
        row0 = fmadd(col3, tmp, row0);
        row3 = col0 * tmp;
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row0 = fnmadd(col3, tmp, row0);
        row3 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row3));

        tmp  = shuffle<1, 0, 3, 2>(shuffle<2, 3, 0, 1>(col1) * col3);
        col2 = shuffle<2, 3, 0, 1>(col2);
        row0 = fmadd(col2, tmp, row0);
        row2 = col0 * tmp;
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row0 = fnmadd(col2, tmp, row0);
        row2 = shuffle<2, 3, 0, 1>(fmsub(col0, tmp, row2));

        tmp  = shuffle<1, 0, 3, 2>(col0 * col1);
        row2 = fmadd(col3, tmp, row2);
        row3 = fmsub(col2, tmp, row3);
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row2 = fmsub(col3, tmp, row2);
        row3 = fnmadd(col2, tmp, row3);

        tmp  = shuffle<1, 0, 3, 2>(col0 * col3);
        row1 = fnmadd(col2, tmp, row1);
        row2 = fmadd(col1, tmp, row2);
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row1 = fmadd(col2, tmp, row1);
        row2 = fnmadd(col1, tmp, row2);

        tmp  = shuffle<1, 0, 3, 2>(col0 * col2);
        row1 = fmadd(col3, tmp, row1);
        row3 = fnmadd(col1, tmp, row3);
        tmp  = shuffle<2, 3, 0, 1>(tmp);
        row1 = fnmadd(col3, tmp, row1);
        row3 = fmadd(col1, tmp, row3);

        Matrix4d inv_times_det = Matrix4d(row0, row1, row2, row3);

        Float64 inv_det = (1.0/(dot(col0, row0)));

        // return (inv_times_det * b);
        return -inv_det * (inv_times_det * b);
    }

    Float L(Vector4d x_in, Vector4d *grad_out,
                           Matrix4d *hess_out, L_data const &data, Float *f_out, Mask active=true) const;
    bool fermat_connection(L_data const &data, Vector4f const &sample,
                           Vector3f *result, Mask active = true) const;

private:
    const Scene *m_scene = nullptr;

    FNEE_config m_config;
};
MTS_EXTERN_CLASS_RENDER(FermatNEE)
NAMESPACE_END(mitsuba)
