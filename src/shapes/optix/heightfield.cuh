#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>

struct OptixHeightfieldData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_object;
    optix::Transform4f to_world;
    const optix::Matrix4f *bicubic_coefs;
    int N;
    float N_f;
    float invL;
    float invW;
    float L;
    float W;
    float H;
    float du_scale;
    float dv_scale;
    float flip;
};
#ifdef __CUDACC__

DEVICE optix::Matrix4f load_coef(OptixHeightfieldData *buf, int i, int j) {
    float *ptr = (float *) buf->bicubic_coefs + ((i * buf->N) + (j)) * 16;
    // enoki::Matrix4f A_enoki(ptr[0] ,ptr[1] ,ptr[2] ,ptr[3] ,ptr[4] ,ptr[5]
    // ,ptr[6] ,ptr[7] ,ptr[8] ,ptr[9]
    // ,ptr[10],ptr[11],ptr[12],ptr[13],ptr[14],ptr[15]);
    optix::Matrix4f A;
    A[0][0] = ptr[0];
    A[1][0] = ptr[1];
    A[2][0] = ptr[2];
    A[3][0] = ptr[3];
    A[0][1] = ptr[4];
    A[1][1] = ptr[5];
    A[2][1] = ptr[6];
    A[3][1] = ptr[7];
    A[0][2] = ptr[8];
    A[1][2] = ptr[9];
    A[2][2] = ptr[10];
    A[3][2] = ptr[11];
    A[0][3] = ptr[12];
    A[1][3] = ptr[13];
    A[2][3] = ptr[14];
    A[3][3] = ptr[15];

    return A;
}

DEVICE float textureBicubic(OptixHeightfieldData *buf, optix::Vector2f &p) {

    // eval point is outside of heightfield definition domain
    if (p[0] > 1 || p[1] > 1 || p[0] < 0 || p[1] < 0) {
        return 0.0f;
    }

    int N    = buf->N;
    float N_f    = buf->N_f;
    float x  = p[0];
    float y  = p[1];
    float x_ = x * N_f;
    float y_ = y * N_f;
    int X    = (floor(x_));
    int Y    = (floor(y_));
    float dx = (x_ - (float) X + 1.0f);
    float dy = (y_ - (float) Y + 1.0f);

    float d2y = dy * dy, d3y = d2y * dy, d2x = dx * dx, d3x = d2x * dx;

    X = min(max(X, 1), N - 3);
    Y = min(max(Y, 1), N - 3);

    optix::Vector4f vY(1.f, dy, d2y, d3y);
    optix::Vector4f vX(1.f, dx, d2x, d3x);

    optix::Matrix4f A = load_coef(buf, X, Y);

    return dot(vX, A.transform_vector4(vY));
}

DEVICE optix::Vector3f eval_curvature(OptixHeightfieldData *buf,
                                      optix::Vector2f p) {

    // eval point is outside of heightfield definition domain
    if (p[0] > 1 || p[1] > 1 || p[0] < 0 || p[1] < 0) {
        return 0.0f;
    }

    int N    = buf->N;
    float N_f    = buf->N_f;

    float x  = p[0];
    float y  = p[1];
    float x_ = x * N_f;
    float y_ = y * N_f;
    int X    = (floor(x_));
    int Y    = (floor(y_));
    float dx = (x_ - (float) X + 1.0f);
    float dy = (y_ - (float) Y + 1.0f);
    X        = min(max(X, 1), N - 3);
    Y        = min(max(Y, 1), N - 3);

    optix::Matrix4f A = load_coef(buf, X, Y);

    float d2y = dy * dy, d3y = d2y * dy, d2x = dx * dx, d3x = d2x * dx;
    optix::Vector4f vY(1.f, dy, d2y, d3y);
    optix::Vector4f vX(1.f, dx, d2x, d3x);
    optix::Vector4f vdX(0.f, 1.f, 2.f * dx, 3.f * d2x);
    optix::Vector4f vdY(0.f, 1.f, 2.f * dy, 3.f * d2y);

    return optix::Vector3f(dot(vX, A.transform_vector4(vY)),
                           dot(vdX, A.transform_vector4(vY)),
                           dot(vX, A.transform_vector4(vdY)));
}

DEVICE float SDF(OptixHeightfieldData *buf, optix::Vector3f p) {
    float x = p[0];
    optix::Vector2f uv(p[1] * buf->invL, p[2] * buf->invW);
    // return 1.f;
    float H = buf->H;
    float r = -H * textureBicubic(buf, uv) - x;
    // float r = 1.0f - x;
    return r;
}

DEVICE float sphere_trace(OptixHeightfieldData *rect, Ray3f &ray) {

    float epsilon = 1e-4f;
    float t       = ray.mint;
    float d       = SDF(rect, ray(t));

    // prevent unwanted backface culling
    float minus = (d >= 0 ? 1.0f : -1.0f);
    d           = minus * d;

    // main loop
    // while (d > epsilon) {
    for (int c = 0; c < 100; c++) {
        t = t + d;
        d = minus * SDF(rect, ray(t));
        if (d < epsilon || t > ray.maxt)
            break;
    }
    return (t > ray.maxt || t < ray.mint ? INFINITY : t);
}

extern "C" __global__ void __intersection__heightfield() {
    const OptixHitGroupData *sbt_data =
        (OptixHitGroupData *) optixGetSbtDataPointer();
    OptixHeightfieldData *rect = (OptixHeightfieldData *) sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();
    // Ray in object-space
    ray = rect->to_object.transform_ray(ray);

    float t = sphere_trace(rect, ray);

    // float t        = 1.0f;
    optix::Vector3f local = ray(t);

    bool in_bounds = (t <= ray.maxt && t >= ray.mint); // NaN-aware conditionals

    if (local.y() >= 0 && local.y() <= rect->L && local.z() <= rect->W &&
        local.z() >= 0 && in_bounds)
        optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
}

extern "C" __global__ void __closesthit__heightfield() {
    unsigned int launch_index = calculate_launch_index();

    if (params.is_ray_test()) { // ray_test
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data =
            (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixHeightfieldData *rect = (OptixHeightfieldData *) sbt_data->data;

        // Ray in instance-space
        Ray3f ray_ = get_ray();
        // Ray in object-space
        Ray3f ray = rect->to_object.transform_ray(ray_);

        // float t = sphere_trace(rect, ray);
        float t = ray.maxt;

        optix::Vector3f local = ray(t);
        optix::Vector2f prim_uv = optix::Vector2f(local.y() * rect->invL, local.z() * rect->invW);

        // Early return for ray_intersect_preliminary call
        if (params.is_ray_intersect_preliminary()) {
            write_output_pi_params(params, launch_index, sbt_data->shape_ptr, 0,
                                   prim_uv, t);
            return;
        }

        /* Compute and store information describing the intersection. This is
           very similar to Heightfield::compute_surface_interaction() */

        optix::Vector3f curvature = eval_curvature(rect, prim_uv);

        // gradient
        float df_du           = -rect->H * curvature[1];
        float df_dv           = -rect->H * curvature[2];
        optix::Vector3f dp_du_grad = optix::Vector3f(df_du, rect->du_scale, 0);
        optix::Vector3f dp_dv_grad = optix::Vector3f(df_dv, 0, rect->dv_scale);
        optix::Vector3f grad_local = normalize(cross(dp_du_grad, dp_dv_grad));

        // Fill surface interaction
        optix::Vector3f p     = ray_(t);
        optix::Vector2f uv    = prim_uv;
        optix::Vector3f ns    = rect->to_world.transform_normal( normalize(rect->flip * grad_local ));
        optix::Vector3f ng    = ns;
        optix::Vector3f dp_du = rect->to_world.transform_normal((float)rect->N_f*dp_du_grad);
        optix::Vector3f dp_dv = rect->to_world.transform_normal((float)rect->N_f*dp_dv_grad);

        write_output_si_params(params, launch_index, sbt_data->shape_ptr, 0, p,
                               uv, ns, ng, dp_du, dp_dv, optix::Vector3f(0.f),
                               optix::Vector3f(0.f), t);
    }
}
#endif
