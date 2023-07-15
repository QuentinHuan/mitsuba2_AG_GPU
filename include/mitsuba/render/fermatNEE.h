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
    MTS_IMPORT_BASE(Shape);
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
    FermatNEE() {}
    FermatNEE(FNEE_config &config) : m_config(config) {}


    void init(const Scene *scene, Sampler *sampler) {
        m_scene   = scene;
        m_sampler = sampler;
    }

private:
    const Scene *m_scene = nullptr;
    Sampler *m_sampler   = nullptr;

    FNEE_config m_config;
};
MTS_EXTERN_CLASS_RENDER(FermatNEE)
NAMESPACE_END(mitsuba)
