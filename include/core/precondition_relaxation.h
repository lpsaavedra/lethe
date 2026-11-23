#ifndef lethe_precondition_relaxation_h
#define lethe_precondition_relaxation_h

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>

namespace dealii
{


  namespace internal
  {
    // ------------------------------------------------------------
    // 1. Optimal relaxation information
    // ------------------------------------------------------------

    struct OptimalRelaxation
    {
      double omega_opt       = 0.0;
      double omega_critical  = 0.0;
      double spectral_radius = 0.0;
    };


    // ------------------------------------------------------------
    // 2. Arnoldi eigenvalue estimation
    //
    // Estimates eigenvalues of T = P^{-1} A.
    // ------------------------------------------------------------

    template <typename MatrixType,
              typename VectorType,
              typename PreconditionerType>
    std::vector<std::complex<double>>
    arnoldi_iteration(const MatrixType         &matrix,
                      const VectorType         &initial_vector,
                      const PreconditionerType &preconditioner,
                      const unsigned int        n_iterations)
    {
      if (n_iterations == 0)
        throw std::invalid_argument("Arnoldi requires at least one iteration.");

      // Arnoldi basis vectors.
      std::vector<VectorType> basis;
      basis.reserve(n_iterations + 1);

      basis.emplace_back();
      basis.back().reinit(initial_vector);
      basis.back() = initial_vector;

      const double initial_norm = basis[0].l2_norm();

      if (!(initial_norm > 0.0) || !std::isfinite(initial_norm))
        throw std::runtime_error(
          "Arnoldi requires a finite, nonzero initial vector.");

      basis[0] /= initial_norm;

      // Upper Hessenberg matrix.
      FullMatrix<double> H(n_iterations + 1, n_iterations);

      VectorType tmp;
      VectorType w;

      tmp.reinit(initial_vector);
      w.reinit(initial_vector);

      unsigned int m = 0;

      for (unsigned int j = 0; j < n_iterations; ++j)
        {
          // Apply T = P^{-1} A.
          matrix.vmult(tmp, basis[j]);
          preconditioner.vmult(w, tmp);

          const double norm_before = w.l2_norm();

          if (!std::isfinite(norm_before))
            throw std::runtime_error(
              "Arnoldi encountered a non-finite vector norm.");

          // Modified Gram-Schmidt with reorthogonalization.
          for (unsigned int pass = 0; pass < 2; ++pass)
            {
              for (unsigned int i = 0; i <= j; ++i)
                {
                  const double hij = basis[i] * w;

                  H(i, j) += hij;

                  w.add(-hij, basis[i]);
                }
            }

          const double beta = w.l2_norm();

          if (!std::isfinite(beta))
            throw std::runtime_error(
              "Arnoldi encountered a non-finite residual norm.");

          H(j + 1, j) = beta;

          m = j + 1;

          // Detect numerical breakdown.
          const double tolerance =
            100.0 * std::numeric_limits<double>::epsilon();

          if (norm_before == 0.0 || beta <= tolerance * norm_before)
            break;

          // No additional basis vector is needed
          // after the last Arnoldi step.
          if (j + 1 == n_iterations)
            break;

          w /= beta;

          basis.emplace_back();
          basis.back().reinit(initial_vector);
          basis.back() = w;
        }

      // Extract the square Hessenberg matrix.
      LAPACKFullMatrix<double> H_m(m, m);

      for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < m; ++j)
          H_m(i, j) = H(i, j);

      // Compute Ritz values.
      H_m.compute_eigenvalues();

      std::vector<std::complex<double>> eigenvalues(m);

      for (unsigned int i = 0; i < m; ++i)
        eigenvalues[i] = H_m.eigenvalue(i);

      return eigenvalues;
    }


    // ------------------------------------------------------------
    // 3. Compute optimal Richardson relaxation parameter
    //
    // omega_opt = argmin_{omega > 0}
    //             max_i |1 - omega * lambda_i|
    // ------------------------------------------------------------

    inline OptimalRelaxation
    compute_optimal_omega(const std::vector<std::complex<double>> &eigenvalues)
    {
      const bool do_ignore_negative_evs = true; // valid?

      if (eigenvalues.empty())
        throw std::invalid_argument(
          "Cannot compute optimal omega from an empty spectrum.");

      OptimalRelaxation info;

      double omega_critical = std::numeric_limits<double>::infinity();

      // Determine the spectral stability limit.
      for (const auto &lambda : eigenvalues)
        {
          const double real_part = lambda.real();

          if (do_ignore_negative_evs && real_part <= 0.0)
            continue;

          const double norm_squared = std::norm(lambda);

          if (!std::isfinite(real_part) || !std::isfinite(norm_squared))
            throw std::runtime_error("Non-finite eigenvalue encountered.");

          // No strictly contractive positive Richardson
          // parameter exists if Re(lambda) <= 0.
          if (real_part <= 0.0)
            throw std::runtime_error("Nonpositive real part detected in the "
                                     "estimated spectrum of P^{-1}A.");

          const double omega_i = 2.0 * real_part / norm_squared;

          omega_critical = std::min(omega_critical, omega_i);
        }

      if (!(omega_critical > 0.0) || !std::isfinite(omega_critical))
        throw std::runtime_error("Invalid critical relaxation parameter.");

      info.omega_critical = omega_critical;

      // Spectral radius of I - omega * P^{-1}A,
      // evaluated using the estimated eigenvalues.
      const auto spectral_radius = [&eigenvalues](const double omega) {
        double rho = 0.0;

        for (const auto &lambda : eigenvalues)
          {
            const double amplification = std::abs(1.0 - omega * lambda);

            rho = std::max(rho, amplification);
          }

        return rho;
      };

      // Minimize the spectral radius using
      // golden-section search.
      //
      // The objective is convex for real omega.
      double left  = 0.0;
      double right = omega_critical;

      constexpr double phi = 0.6180339887498948482;

      double x1 = right - phi * (right - left);
      double x2 = left + phi * (right - left);

      double f1 = spectral_radius(x1);
      double f2 = spectral_radius(x2);

      for (unsigned int iteration = 0; iteration < 100; ++iteration)
        {
          const double tolerance = 1.e-12 * std::max(1.0, omega_critical);

          if (right - left <= tolerance)
            break;

          if (f1 > f2)
            {
              left = x1;

              x1 = x2;
              f1 = f2;

              x2 = left + phi * (right - left);
              f2 = spectral_radius(x2);
            }
          else
            {
              right = x2;

              x2 = x1;
              f2 = f1;

              x1 = right - phi * (right - left);
              f1 = spectral_radius(x1);
            }
        }

      info.omega_opt = 0.5 * (left + right);

      info.spectral_radius = spectral_radius(info.omega_opt);

      return info;
    }

  } // namespace internal

  /**
   * Base class for other preconditioners. Here, only some common features
   * Jacobi, SOR and SSOR preconditioners are implemented. For preconditioning,
   * refer to derived classes.
   *
   * One iteration is given as:
   * @f[
   *  x^{n+1} = x^{n} + \alpha P^{-1} (b-Ax^n).
   * @f]
   *
   * The relaxation parameter $\alpha$ has to be in the range:
   * @f[
   *  0 < \alpha < \frac{2}{\lambda_{\max}(P^{-1}A)}.
   * @f]
   * Its theoretically optimal value is given by:
   * @f[
   *    \alpha := \frac{2}{\lambda_{\min}(P^{-1}A)+\lambda_{\max}(P^{-1}A)}.
   * @f]
   *
   * For details on the algorithm, see @cite Varga2009.
   *
   * The relaxation parameter can be set manually or can be automatically
   * determined, for which we use the theoretically optimal value.
   * For this purpose, the user needs to set the relaxation parameter
   * to zero. Internally, the minimum and maximum eigenvalues of the
   * preconditioned system are estimated by an eigenvalue algorithm, and the
   * resulting estimate is multiplied by the safety factor (1.2 by default). For
   * more details on the underlying algorithms, see PreconditionChebyshev.
   */
  template <typename MatrixType         = SparseMatrix<double>,
            typename PreconditionerType = IdentityMatrix>
  class MyPreconditionRelaxation : public EnableObserverPointer
  {
  public:
    /**
     * Declare type for container size.
     */
    using size_type = types::global_dof_index;

    /**
     * Class for parameters.
     */
    class AdditionalData
      : public internal::EigenvalueAlgorithmAdditionalData<PreconditionerType>
    {
    public:
      using EigenvalueAlgorithm = internal::EigenvalueAlgorithm;

      /**
       * Constructor.
       */
      AdditionalData(
        const double              relaxation          = 1.,
        const unsigned int        n_iterations        = 1,
        const double              smoothing_range     = 0.,
        const unsigned int        eig_cg_n_iterations = 8,
        const double              eig_cg_residual     = 1e-2,
        const double              max_eigenvalue      = 1,
        const EigenvalueAlgorithm eigenvalue_algorithm =
          EigenvalueAlgorithm::lanczos,
        const double safety_factor  = 1.2,
        const double max_relaxation = std::numeric_limits<double>::max());

      /**
       * Relaxation parameter.
       */
      double relaxation;

      /**
       * Max. relaxation parameter.
       */
      double max_relaxation;

      /**
       * Number of smoothing steps to be performed in an
       * invocation of vmult() or step().
       */
      unsigned int n_iterations;
    };

    /**
     * Initialize matrix and relaxation parameter. The matrix is just stored in
     * the preconditioner object. The relaxation parameter should be larger than
     * zero and smaller than 2 for numerical reasons. It defaults to 1.
     */
    void
    initialize(const MatrixType     &A,
               const AdditionalData &parameters = AdditionalData());

    /**
     * Release the matrix and reset its pointer.
     */
    void
    clear();

    /**
     * Return the dimension of the codomain (or range) space. Note that the
     * matrix is of dimension $m \times n$.
     */
    size_type
    m() const;

    /**
     * Return the dimension of the domain space. Note that the matrix is of
     * dimension $m \times n$.
     */
    size_type
    n() const;

    /**
     * Apply preconditioner.
     */
    template <typename VectorType>
    void
    vmult(VectorType &, const VectorType &) const;

    /**
     * Apply transpose preconditioner. Since this is a symmetric preconditioner,
     * this function is the same as vmult().
     */
    template <typename VectorType>
    void
    Tvmult(VectorType &, const VectorType &) const;

    /**
     * Perform one step of the preconditioned Richardson iteration
     */
    template <typename VectorType>
    void
    step(VectorType &x, const VectorType &rhs) const;

    /**
     * Perform one transposed step of the preconditioned Richardson iteration.
     */
    template <typename VectorType>
    void
    Tstep(VectorType &x, const VectorType &rhs) const;

    using EigenvalueInformation = internal::EigenvalueInformation;

    /**
     * Estimate eigenvalues and set relaxation parameter.
     */
    template <typename VectorType>
    EigenvalueInformation
    estimate_eigenvalues(const VectorType &src) const;

    /**
     * Estimate eigenvalues and set relaxation parameter.
     */
    template <typename VectorType>
    internal::OptimalRelaxation
    estimate_eigenvalues_general(const VectorType &src) const;

    /**
     * Return the relaxation parameter. This function also allows to return
     * the parameter in case it was internally determined by running an
     * eigenvalue algorithm.
     */
    double
    get_relaxation() const;

  protected:
    /**
     * Pointer to the matrix object.
     */
    ObserverPointer<const MatrixType, MyPreconditionRelaxation<MatrixType>> A;

    /**
     * Stores the additional data passed to the initialize function, obtained
     * through a copy operation.
     */
    AdditionalData data;

    /**
     * Preconditioner.
     */
    std::shared_ptr<PreconditionerType> preconditioner;

    /**
     * Stores whether the preconditioner has been set up and eigenvalues have
     * been computed.
     */
    bool eigenvalues_are_initialized;
  };

  template <typename MatrixType, typename PreconditionerType>
  inline MyPreconditionRelaxation<MatrixType, PreconditionerType>::
    AdditionalData::AdditionalData(
      const double              relaxation,
      const unsigned int        n_iterations,
      const double              smoothing_range,
      const unsigned int        eig_cg_n_iterations,
      const double              eig_cg_residual,
      const double              max_eigenvalue,
      const EigenvalueAlgorithm eigenvalue_algorithm,
      const double              safety_factor,
      const double              max_relaxation)
    : internal::EigenvalueAlgorithmAdditionalData<PreconditionerType>(
        smoothing_range,
        eig_cg_n_iterations,
        eig_cg_residual,
        max_eigenvalue,
        eigenvalue_algorithm,
        safety_factor)
    , relaxation(relaxation)
    , max_relaxation(max_relaxation)
    , n_iterations(n_iterations)
  {}

  template <typename MatrixType, typename PreconditionerType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::initialize(
    const MatrixType     &rA,
    const AdditionalData &parameters)
  {
    A                           = &rA;
    eigenvalues_are_initialized = false;

    Assert(parameters.preconditioner, ExcNotInitialized());

    this->data = parameters;
  }


  template <typename MatrixType, typename PreconditionerType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::clear()
  {
    eigenvalues_are_initialized = false;
    A                           = nullptr;
    data.relaxation             = 1.0;
    data.preconditioner         = nullptr;
  }

  template <typename MatrixType, typename PreconditionerType>
  inline
    typename MyPreconditionRelaxation<MatrixType, PreconditionerType>::size_type
    MyPreconditionRelaxation<MatrixType, PreconditionerType>::m() const
  {
    Assert(A != nullptr, ExcNotInitialized());
    return A->m();
  }

  template <typename MatrixType, typename PreconditionerType>
  inline
    typename MyPreconditionRelaxation<MatrixType, PreconditionerType>::size_type
    MyPreconditionRelaxation<MatrixType, PreconditionerType>::n() const
  {
    Assert(A != nullptr, ExcNotInitialized());
    return A->n();
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(this->A != nullptr, ExcNotInitialized());
    Assert(data.preconditioner != nullptr, ExcNotInitialized());

    if (eigenvalues_are_initialized == false)
      estimate_eigenvalues(src);

    VectorType tmp1, tmp2;

    for (unsigned int i = 0; i < data.n_iterations; ++i)
      internal::PreconditionRelaxation::step_operations(*A,
                                                        *data.preconditioner,
                                                        dst,
                                                        src,
                                                        data.relaxation,
                                                        tmp1,
                                                        tmp2,
                                                        i,
                                                        false);
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::Tvmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(this->A != nullptr, ExcNotInitialized());
    Assert(data.preconditioner != nullptr, ExcNotInitialized());

    if (eigenvalues_are_initialized == false)
      estimate_eigenvalues(src);

    VectorType tmp1, tmp2;

    for (unsigned int i = 0; i < data.n_iterations; ++i)
      internal::PreconditionRelaxation::step_operations(*A,
                                                        *data.preconditioner,
                                                        dst,
                                                        src,
                                                        data.relaxation,
                                                        tmp1,
                                                        tmp2,
                                                        i,
                                                        true);
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::step(
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(this->A != nullptr, ExcNotInitialized());
    Assert(data.preconditioner != nullptr, ExcNotInitialized());

    if (eigenvalues_are_initialized == false)
      estimate_eigenvalues(src);

    VectorType tmp1, tmp2;

    for (unsigned int i = 1; i <= data.n_iterations; ++i)
      internal::PreconditionRelaxation::step_operations(*A,
                                                        *data.preconditioner,
                                                        dst,
                                                        src,
                                                        data.relaxation,
                                                        tmp1,
                                                        tmp2,
                                                        i,
                                                        false);
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline void
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::Tstep(
    VectorType       &dst,
    const VectorType &src) const
  {
    Assert(this->A != nullptr, ExcNotInitialized());
    Assert(data.preconditioner != nullptr, ExcNotInitialized());

    if (eigenvalues_are_initialized == false)
      estimate_eigenvalues(src);

    VectorType tmp1, tmp2;

    for (unsigned int i = 1; i <= data.n_iterations; ++i)
      internal::PreconditionRelaxation::step_operations(*A,
                                                        *data.preconditioner,
                                                        dst,
                                                        src,
                                                        data.relaxation,
                                                        tmp1,
                                                        tmp2,
                                                        i,
                                                        true);
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline internal::EigenvalueInformation
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::
    estimate_eigenvalues(const VectorType &src) const
  {
    // Assert(eigenvalues_are_initialized == false, ExcInternalError());

    EigenvalueInformation info;

    if (true)
      {
        VectorType solution_old, temp_vector1;

        solution_old.reinit(src);
        temp_vector1.reinit(src, true);

        info = internal::estimate_eigenvalues<MatrixType>(data,
                                                          A,
                                                          solution_old,
                                                          temp_vector1,
                                                          data.n_iterations,
                                                          data.safety_factor);

        const double alpha =
          (data.smoothing_range > 1. ?
             info.max_eigenvalue_estimate / data.smoothing_range :
             std::min(0.9 * info.max_eigenvalue_estimate,
                      info.min_eigenvalue_estimate));

        if (data.relaxation == 0.0)
          const_cast<
            MyPreconditionRelaxation<MatrixType, PreconditionerType> *>(this)
            ->data.relaxation =
            std::min(data.max_relaxation,
                     2.0 / (alpha + info.max_eigenvalue_estimate));
      }

    const_cast<MyPreconditionRelaxation<MatrixType, PreconditionerType> *>(this)
      ->eigenvalues_are_initialized = true;

    return info;
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  inline internal::OptimalRelaxation
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::
    estimate_eigenvalues_general(const VectorType &src) const
  {
    // Assert(eigenvalues_are_initialized == false, ExcInternalError());
    Assert(A != nullptr, ExcNotInitialized());
    Assert(data.preconditioner != nullptr, ExcNotInitialized());

    internal::OptimalRelaxation info{};

    if (true)
      {
        VectorType initial_vector;
        initial_vector.reinit(src);

        // Initialize the Arnoldi starting vector.
        internal::set_initial_guess(initial_vector);
        data.constraints.set_zero(initial_vector);

        AssertThrow(initial_vector.l2_norm() > 0.0,
                    ExcMessage("Arnoldi requires a nonzero initial vector."));

        // Estimate eigenvalues of P^{-1} A using Arnoldi.
        const auto eigenvalues = internal::arnoldi_iteration(
          *A, initial_vector, *data.preconditioner, data.eig_cg_n_iterations);

        // Compute the optimal Richardson relaxation parameter:
        //
        // omega_opt = argmin_{omega > 0}
        //             max_i |1 - omega * lambda_i|
        //
        info = internal::compute_optimal_omega(eigenvalues);

        if (data.relaxation == 0.0)
          const_cast<
            MyPreconditionRelaxation<MatrixType, PreconditionerType> *>(this)
            ->data.relaxation =
            std::min(data.max_relaxation, info.omega_opt) / data.safety_factor;
      }

    const_cast<MyPreconditionRelaxation<MatrixType, PreconditionerType> *>(this)
      ->eigenvalues_are_initialized = true;

    return info;
  }

  template <typename MatrixType, typename PreconditionerType>
  double
  MyPreconditionRelaxation<MatrixType, PreconditionerType>::get_relaxation()
    const
  {
    return data.relaxation;
  }
} // namespace dealii

#endif