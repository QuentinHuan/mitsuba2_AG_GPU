#include <mitsuba/render/fermatNEE.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
typename FermatNEE<Float, Spectrum>::Vector4d FermatNEE<Float, Spectrum>::newton_solver_double(L_data const &data,
                                                      Vector4f *X_out,
                                                      Float *f_out) const {
    // ScopedPhase scpe_phase(ProfilerPhase::FermaNEE_NewtonSolver);

    Float t = 1.f;
    Vector4d G(0);
    Matrix4d H(0);
    Vector4d X1 = Vector4d(0), X0 = Vector4d(*(X_out));

    for (int i = 0; i < 20; i++) {
        Float f = L(X0, &G, &H, data, f_out);

        Vector4d v = solve_4x4(H, G);

        // line search Armijo
        X1 = X0 + t * v;
        ENOKI_UNROLL for (size_t i = 0; i < 8; i++) {
            Mask armijo_condition =
                L(X1, nullptr, nullptr, data, f_out, !armijo_condition) >
                f + m_config.alpha * t *
                        dot(G, v); // 1 ==> Armijo condition satisfied
            t  = select(armijo_condition, t, t * 0.5f);
            X1 = X0 + t * v;
        }
        X0 = X1;

   //     // gradient norm small enough, return
   //     // Mask converged = norm(G) < 0.00001;
    }
    return X0;
}

template <typename Float, typename Spectrum>
Float FermatNEE<Float, Spectrum>::L(Vector4d x_in, Vector4d *grad_out,
                                    Matrix4d *hess_out, L_data const &data,
                                    Float *f_out, Mask active) const {
    *f_out = 0.f;
    return 1.0f;
}

template <typename Float, typename Spectrum>
bool FermatNEE<Float, Spectrum>::fermat_connection(L_data const &data,
                                                   Vector4f const &sample,
                                                   Vector3f *result,
                                                   Mask active) const {
    // ScopedPhase scope_phase(ProfilerPhase::FermaNEE_principal);
    //
    // Setup solver input data
    Vector4d X(sample);
    Vector4f X_in(0.f);

    // proposal computation
    Float f_out = 0.0f;
    X = newton_solver_double(data, &X_in, &f_out);

    // Output
    Point3f P1(f_out, X.x() * data.dimensions.y(), X.y() * data.dimensions.z());
    *result = normalize(P1 - data.O);
    return true;
}

MTS_INSTANTIATE_CLASS(FermatNEE)
NAMESPACE_END(mitsuba)
