#include <cstddef>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <vector>

// #include <mitsuba/ui/viewer.h>
// #include <mitsuba/ui/texture.h>
#include <mitsuba/core/appender.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/formatter.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>

#include <enoki/dynamic.h>
#include <enoki/autodiff.h>
#include <enoki/special.h> // for erf()

// #include <optim.hpp>

#if defined(MTS_ENABLE_OPTIX)
#include "optix/heightfield.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-heightfield:

Heightfield (:monosp:`heightfield`)
-------------------------------------------------

.. pluginparameters::

 * - origin
   - |point|
   - Origin of the heightfield (Default: (0, 0, 0))
* - L
   - |float|
   - length (|) of the heightfield (Default: 0)
* - W
   - |float|
   - width (---) of the heightfield (Default: 0)
 * - H
   - |float|
   - maximum elevation of the heightfield (Default: 0). Extends in normals
direction
 * - flip_normals
   - |bool|
   - Is the heightfield inverted, i.e. should the normal vectors be flipped?
(Default:|false|, i.e. the normals point outside)
 * - to_world
   - |transform|
   -  Specifies an optional linear object-to-world transformation.
      Note that non-uniform scales and shears are not permitted!
      (Default: none, i.e. object space = world space)

This shape plugin describes a heightfield intersection primitive: given an .exr
elevation profile, a surface is reconstructed using bicubic interpolation

A heightfield can either be configured using a linear :monosp:`to_world`
transformation or the :monosp:`center` and :monosp:`radius` parameters (or
both). The two declarations below are equivalent.


.. warning:: This plugin is currently CPU only.

 */

template <typename Float, typename Spectrum>
class Heightfield final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES(Texture)

    using typename Base::ScalarSize;
    using InputFloat    = float;
    using FloatStorage  = DynamicBuffer<replace_scalar_t<Float, InputFloat>>;
    using InputPoint4f  = Point<InputFloat, 4>;
    using InputPoint16f = Point<InputFloat, 16>;

    Heightfield(const Properties &props) : Base(props) {
        FileResolver *fs          = Thread::thread()->file_resolver();
        fs::path heightfield_path = fs->resolve(props.string("heightmap_path"));
        ref<Bitmap> bitmap        = new Bitmap(heightfield_path);
        N                         = (size_t)(bitmap->size().x());
        scale                     = rcp((ScalarFloat) N);

        // ifdef CUDA_CC
        m_bicubic_buf = zero<FloatStorage>(N * N * 16);
        m_bicubic_buf.managed();

        if constexpr (is_cuda_array_v<Float>) {
            cuda_sync();
        }

        InputFloat *bicubic_ptr = m_bicubic_buf.data();

        ScalarFloat *ptr = (ScalarFloat *) bitmap->data();

        // // test
        // DiffArray<float> T = 1.f;
        // set_requires_gradient(T);
        // backward(T);
        // DiffArray<float> B = enoki::erf(T);
        // backward(B);
        // set_label(T, "T");
        // set_label(B, "B");
        // std::cout << "compute gradient test " << gradient(T) << std::endl;

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                size_t X = i, Y = j;
                X = min(N - 3, max(1, X));
                Y = min(N - 3, max(1, Y));
                ScalarFloat F00 =
                    load_unaligned<ScalarFloat>(ptr + (X - 1) + (Y - 1) * N);
                ScalarFloat F01 =
                    load_unaligned<ScalarFloat>(ptr + (X - 1) + (Y - 0) * N);
                ScalarFloat F02 =
                    load_unaligned<ScalarFloat>(ptr + (X - 1) + (Y + 1) * N);
                ScalarFloat F03 =
                    load_unaligned<ScalarFloat>(ptr + (X - 1) + (Y + 2) * N);
                ScalarFloat F10 =
                    load_unaligned<ScalarFloat>(ptr + (X + 0) + (Y - 1) * N);
                ScalarFloat F11 =
                    load_unaligned<ScalarFloat>(ptr + (X + 0) + (Y - 0) * N);
                ScalarFloat F12 =
                    load_unaligned<ScalarFloat>(ptr + (X + 0) + (Y + 1) * N);
                ScalarFloat F13 =
                    load_unaligned<ScalarFloat>(ptr + (X + 0) + (Y + 2) * N);
                ScalarFloat F20 =
                    load_unaligned<ScalarFloat>(ptr + (X + 1) + (Y - 1) * N);
                ScalarFloat F21 =
                    load_unaligned<ScalarFloat>(ptr + (X + 1) + (Y - 0) * N);
                ScalarFloat F22 =
                    load_unaligned<ScalarFloat>(ptr + (X + 1) + (Y + 1) * N);
                ScalarFloat F23 =
                    load_unaligned<ScalarFloat>(ptr + (X + 1) + (Y + 2) * N);
                ScalarFloat F30 =
                    load_unaligned<ScalarFloat>(ptr + (X + 2) + (Y - 1) * N);
                ScalarFloat F31 =
                    load_unaligned<ScalarFloat>(ptr + (X + 2) + (Y - 0) * N);
                ScalarFloat F32 =
                    load_unaligned<ScalarFloat>(ptr + (X + 2) + (Y + 1) * N);
                ScalarFloat F33 =
                    load_unaligned<ScalarFloat>(ptr + (X + 2) + (Y + 2) * N);
                InputPoint16f P(
                    1.0F * F00,
                    -1.83333333F * F00 + 3.0F * F01 - 1.5F * F02 +
                        0.333333333F * F03,
                    1.0F * F00 - 2.5F * F01 + 2.0F * F02 - 0.5F * F03,
                    -0.166666667F * F00 + 0.5F * F01 - 0.5F * F02 +
                        0.166666667F * F03,
                    -1.83333333F * F00 + 3.0F * F10 - 1.5F * F20 +
                        0.333333333F * F30,
                    3.36111111F * F00 - 5.5F * F01 + 2.75F * F02 -
                        0.611111111F * F03 - 5.5F * F10 + 9.0F * F11 -
                        4.5F * F12 + 1.0F * F13 + 2.75F * F20 - 4.5F * F21 +
                        2.25F * F22 - 0.5F * F23 - 0.611111111F * F30 +
                        1.0F * F31 - 0.5F * F32 + 0.111111111F * F33,
                    -1.83333333F * F00 + 4.58333333F * F01 - 3.66666667F * F02 +
                        0.916666667F * F03 + 3.0F * F10 - 7.5F * F11 +
                        6.0F * F12 - 1.5F * F13 - 1.5F * F20 + 3.75F * F21 -
                        3.0F * F22 + 0.75F * F23 + 0.333333333F * F30 -
                        0.833333333F * F31 + 0.666666667F * F32 -
                        0.166666667F * F33,
                    0.305555556F * F00 - 0.916666667F * F01 +
                        0.916666667F * F02 - 0.305555556F * F03 - 0.5F * F10 +
                        1.5F * F11 - 1.5F * F12 + 0.5F * F13 + 0.25F * F20 -
                        0.75F * F21 + 0.75F * F22 - 0.25F * F23 -
                        0.0555555556F * F30 + 0.166666667F * F31 -
                        0.166666667F * F32 + 0.0555555556F * F33,
                    1.0F * F00 - 2.5F * F10 + 2.0F * F20 - 0.5F * F30,
                    -1.83333333F * F00 + 3.0F * F01 - 1.5F * F02 +
                        0.333333333F * F03 + 4.58333333F * F10 - 7.5F * F11 +
                        3.75F * F12 - 0.833333333F * F13 - 3.66666667F * F20 +
                        6.0F * F21 - 3.0F * F22 + 0.666666667F * F23 +
                        0.916666667F * F30 - 1.5F * F31 + 0.75F * F32 -
                        0.166666667F * F33,
                    1.0F * F00 - 2.5F * F01 + 2.0F * F02 - 0.5F * F03 -
                        2.5F * F10 + 6.25F * F11 - 5.0F * F12 + 1.25F * F13 +
                        2.0F * F20 - 5.0F * F21 + 4.0F * F22 - 1.0F * F23 -
                        0.5F * F30 + 1.25F * F31 - 1.0F * F32 + 0.25F * F33,
                    -0.166666667F * F00 + 0.5F * F01 - 0.5F * F02 +
                        0.166666667F * F03 + 0.416666667F * F10 - 1.25F * F11 +
                        1.25F * F12 - 0.416666667F * F13 - 0.333333333F * F20 +
                        1.0F * F21 - 1.0F * F22 + 0.333333333F * F23 +
                        0.0833333333F * F30 - 0.25F * F31 + 0.25F * F32 -
                        0.0833333333F * F33,
                    -0.166666667F * F00 + 0.5F * F10 - 0.5F * F20 +
                        0.166666667F * F30,
                    0.305555556F * F00 - 0.5F * F01 + 0.25F * F02 -
                        0.0555555556F * F03 - 0.916666667F * F10 + 1.5F * F11 -
                        0.75F * F12 + 0.166666667F * F13 + 0.916666667F * F20 -
                        1.5F * F21 + 0.75F * F22 - 0.166666667F * F23 -
                        0.305555556F * F30 + 0.5F * F31 - 0.25F * F32 +
                        0.0555555556F * F33,
                    -0.166666667F * F00 + 0.416666667F * F01 -
                        0.333333333F * F02 + 0.0833333333F * F03 + 0.5F * F10 -
                        1.25F * F11 + 1.0F * F12 - 0.25F * F13 - 0.5F * F20 +
                        1.25F * F21 - 1.0F * F22 + 0.25F * F23 +
                        0.166666667F * F30 - 0.416666667F * F31 +
                        0.333333333F * F32 - 0.0833333333F * F33,
                    0.0277777778F * F00 - 0.0833333333F * F01 +
                        0.0833333333F * F02 - 0.0277777778F * F03 -
                        0.0833333333F * F10 + 0.25F * F11 - 0.25F * F12 +
                        0.0833333333F * F13 + 0.0833333333F * F20 -
                        0.25F * F21 + 0.25F * F22 - 0.0833333333F * F23 -
                        0.0277777778F * F30 + 0.0833333333F * F31 -
                        0.0833333333F * F32 + 0.0277777778F * F33);
                store_unaligned(bicubic_ptr, P);
                bicubic_ptr += 16;
            }
        }
        // dimensions
        L    = props.float_("L", 2.0f);
        W    = props.float_("W", 2.0f);
        H    = props.float_("H", 0.04);
        flip = select(props.bool_("flip_normals", false), ScalarFloat(-1.0f),
                      ScalarFloat(1.0f));

        invW     = rcp(W);
        invL     = rcp(L);
        du_scale = L * rcp(ScalarFloat(N));
        dv_scale = W * rcp(ScalarFloat(N));

        update();
        set_children();
    }

    Matrix4f load_coef(Int32 i, Int32 j) const {
        Matrix4f A =
            gather<Matrix4f>(m_bicubic_buf, (Int32) i * (Int32)(N) + (Int32) j);
        return transpose(A);
    }

    Float textureBicubic(Vector2f p, Mask active) const {
        MTS_MASK_ARGUMENT(active);

        // eval point is outside of heightfield definition domain
        active = any(p > 0.0f || p < 1.0f);

        Float x  = p[0];
        Float y  = p[1];
        Float x_ = x * Float(N);
        Float y_ = y * Float(N);
        Int32 X  = Int32(floor(x_));
        Int32 Y  = Int32(floor(y_));
        Float dx = (x_ - X + 1);
        Float dy = (y_ - Y + 1);

        Float d2y = dy * dy, d3y = d2y * dy, d2x = dx * dx, d3x = d2x * dx;

        X = min(max(X, 1), N - 3);
        Y = min(max(Y, 1), N - 3);

        Vector4f vY(1.f, dy, d2y, d3y);
        Vector4f vX(1.f, dx, d2x, d3x);

        Matrix4f A = load_coef(X, Y);

        // if (partial == 0) {
        //     // nothing to do
        // } else if (partial == 1) {
        //     vX = optix::optix::Vector4f(0.f, 1.f, 2.f * dx, 3.f * d2x);
        // } else if (partial == 2) {
        //     vY = optix::optix::Vector4f(0.f, 1.f, 2.f * dy, 3.f * d2y);
        // } else if (partial == 3) {
        //     vX = optix::optix::Vector4f(0.f, 0.f, 2.f, 6.f * dx);
        // } else if (partial == 4) {
        //     vY = optix::optix::Vector4f(0.f, 0.f, 2.f, 6.f * dy);
        // } else if (partial == 5) {
        //     vX = optix::optix::Vector4f(0.f, 1.f, 2.f * dx, 3.f * d2x);
        //     vY = optix::optix::Vector4f(0.f, 1.f, 2.f * dy, 3.f * d2y);
        // } else {
        //     std::cout << "error, Heighfield bicubic() <partial> " << partial
        //     << "is not a valid parameter" << std::endl; return 0;
        // }
        return dot(vX, A * vY);
    }

    Float textureBicubic_du(Vector2f p, Mask active) const {
        MTS_MASK_ARGUMENT(active);

        Float x  = p[0];
        Float y  = p[1];
        Float x_ = x * Float(N);
        Float y_ = y * Float(N);
        Int32 X  = Int32(floor(x_));
        Int32 Y  = Int32(floor(y_));
        Float dx = (x_ - X + 1);
        Float dy = (y_ - Y + 1);

        Float d2y = dy * dy, d3y = d2y * dy, d2x = dx * dx;

        X = min(max(X, 1), N - 3);
        Y = min(max(Y, 1), N - 3);

        Vector4f vY(1.f, dy, d2y, d3y);
        Vector4f vX(0.f, 1.f, 2.f * dx, 3.f * d2x);

        Matrix4f A = load_coef(X, Y);
        return dot(vX, A * vY);
    }

    Float textureBicubic_dv(Vector2f p, Mask active) const {
        MTS_MASK_ARGUMENT(active);

        Float x  = p[0];
        Float y  = p[1];
        Float x_ = x * Float(N);
        Float y_ = y * Float(N);
        Int32 X  = Int32(floor(x_));
        Int32 Y  = Int32(floor(y_));
        Float dx = (x_ - X + 1);
        Float dy = (y_ - Y + 1);

        Float d2y = dy * dy, d2x = dx * dx, d3x = d2x * dx;

        Vector4f vY(0.f, 1.f, 2.f * dy, 3.f * d2y);
        Vector4f vX(1.f, dx, d2x, d3x);

        Matrix4f A = load_coef(X, Y);
        return dot(vX, A * vY);
    }

    void update() {
        m_to_object = m_to_world.inverse();

        ScalarVector3f dp_du = m_to_world * ScalarVector3f(2.f, 0.f, 0.f);
        ScalarVector3f dp_dv = m_to_world * ScalarVector3f(0.f, 2.f, 0.f);
        ScalarNormal3f normal =
            normalize(m_to_world * ScalarNormal3f(0.f, 0.f, 1.f));
        m_frame = ScalarFrame3f(dp_du, dp_dv, normal);

        m_inv_surface_area = rcp(surface_area());
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(0, L, 0)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(0, L, W)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(0, 0, W)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(0, 0, 0)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-H, L, 0)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-H, L, W)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-H, 0, W)));
        bbox.expand(m_to_world.transform_affine(ScalarPoint3f(-H, 0, 0)));
        return bbox;
    }

    ScalarFloat surface_area() const override {
        return norm(cross(m_frame.s, m_frame.t));
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        PositionSample3f ps;
        ps.p = m_to_world.transform_affine(
            Point3f(sample.x() * 2.f - 1.f, sample.y() * 2.f - 1.f, 0.f));
        ps.n     = m_frame.n;
        ps.pdf   = m_inv_surface_area;
        ps.uv    = sample;
        ps.time  = time;
        ps.delta = false;

        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/,
                       Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_inv_surface_area;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    // Heighfield distance field description
    Float h(Point3f P, Mask active) const {
        MTS_MASK_ARGUMENT(active);
        Float x = P.x();
        Point2f p(P.y() * invL, P.z() * invW);
        return h_f(p, active) - x;
    }

    // Heighfield function description
    Float h_f(Point2f p, Mask active) const {
        MTS_MASK_ARGUMENT(active);
        return -H * textureBicubic(p, active);
    }

    // gradient computation
    Normal3f grad_h(Point2f P, Mask active) const {
        MTS_MASK_ARGUMENT(active);
        if (H != 0) {
            Float df_du    = -H * textureBicubic_du(P, active);
            Float df_dv    = -H * textureBicubic_dv(P, active);
            Normal3f dp_du = Normal3f(df_du, du_scale, 0);
            Normal3f dp_dv = Normal3f(df_dv, 0, du_scale);

            return normalize(cross(dp_du, dp_dv));
        } else {
            return Normal3f(1.0f, 0, 0);
        }
    }

    // Intersect ray_ (local space) and Heightfield surface by sphere tracing
    Float sphereTrace(const Ray3f &ray_, Mask active) const {
        MTS_MASK_ARGUMENT(active);

        Float epsilon = 1e-4f;
        Float t       = ray_.mint;
        Float d       = h(ray_(t), active);

        // prevent unwanted backface culling
        Float minus = select(d >= 0.0f, Float(1.0f), Float(-1.0f));
        d           = minus * d;

        // main loop
        for (std::size_t c = 0; c < 20; c++) {
            t = t + d;
            d = minus * h(ray_(t), active);
            active = d > epsilon && t < ray_.maxt;

            if (none(active))
                break;
        }
        return select(t > ray_.maxt || t < 0,
                      std::numeric_limits<ScalarFloat>::infinity(), t);
    }

    PreliminaryIntersection3f
    ray_intersect_preliminary(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray     = m_to_object.transform_affine(ray_);
        Float t       = sphereTrace(ray, active);
        Point3f local = ray(t);

        // Is intersection within ray segment and surface?
        active = active && t >= ray.mint && t <= ray.maxt && local.z() <= W &&
                 local.y() <= L && local.y() >= 0.f && local.z() >= 0.f;

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t                         = select(active, t, math::Infinity<Float>);
        pi.prim_uv = Point2f(local.y() * invL, local.z() * invW);
        pi.shape   = this;

        return pi;
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray = m_to_object.transform_affine(ray_);
        Float t   = sphereTrace(ray, active);

        Point3f local = ray(t);

        // Is intersection within ray segment and surface?
        return active && t >= ray.mint && t <= ray.maxt && local.z() <= W &&
               local.y() <= L && local.y() >= 0.f && local.z() >= 0.f;
    }

    SurfaceInteraction3f
    compute_surface_interaction(const Ray3f &ray, PreliminaryIntersection3f pi,
                                HitComputeFlags flags,
                                Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable &&
            !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = select(active, pi.t, math::Infinity<Float>);

        si.p          = ray(pi.t);
        Point3f local = m_to_object.transform_affine(si.p);
        si.uv = Point2f(local.y() * invL, local.z() * invW);

        // partial derivatives
        Normal3f grad_local = grad_h(si.uv, active);

        // Fill surface interaction
        si.n          = m_to_world * normalize(flip * grad_local);
        si.sh_frame.n = si.n;
        si.dp_du = m_to_world *
                   (Normal3f(-H * (Float) N * textureBicubic_du(si.uv, active),
                             (Float) N * du_scale, 0));
        si.dp_dv = m_to_world *
                   (Normal3f(-H * (Float) N * textureBicubic_dv(si.uv, active),
                             0, (Float) N * dv_scale));

        si.dn_du = si.dn_dv = zero<Vector3f>();

        return si;
    }

    Float eval_attribute_1(const std::string &name, const SurfaceInteraction3f &si, Mask active = true) const override {
        MTS_MASK_ARGUMENT(active);
        if (name == "H")
            return H;
        else if (name == "L")
            return L;
        else if (name == "W")
            return W;
        else if (name == "flip")
            return flip ? -1.0f : 1.0f;
        else if (name == "N")
            return Float(N);
        else
            return 0.0f;
    }
    Color3f eval_attribute_3(const std::string &name, const SurfaceInteraction3f &si, Mask active = true) const override {
        MTS_MASK_ARGUMENT(active);
        if (name == "dimensions")
            return Color3f(H,L,W);
        else
            return Color3f(0.f);
    }
    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {
        update();
        Base::parameters_changed();
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixHeightfieldData));

            OptixHeightfieldData data = {
                bbox(),
                m_to_object,
                m_to_world,
                (const optix::Matrix4f *) m_bicubic_buf.data(),
                (int) N,
                (float) N,
                (float) invL,
                (float) invW,
                (float) L,
                (float) W,
                (float) H,
                (float) du_scale,
                (float) dv_scale,
                (float) flip
            };

            cuda_memcpy_to_device(m_optix_data_ptr, &data,
                                  sizeof(OptixHeightfieldData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Rectangle[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << ","
            << std::endl
            << "  frame = " << string::indent(m_frame) << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFrame3f m_frame;
    ScalarFloat m_inv_surface_area;
    ScalarFloat scale;
    ScalarFloat L, W, H;
    ScalarFloat invL, invW, du_scale, dv_scale, flip;
    size_t N;
    FloatStorage m_bicubic_buf;
    Matrix4d test;
};

MTS_IMPLEMENT_CLASS_VARIANT(Heightfield, Shape)
MTS_EXPORT_PLUGIN(Heightfield, "Heightfield intersection primitive");
NAMESPACE_END(mitsuba)
