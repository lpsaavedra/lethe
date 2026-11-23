#ifndef lethe_precondition_relaxation_h
#define lethe_precondition_relaxation_h

#include <deal.II/lac/precondition.h>

namespace dealii
{
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
    Assert(eigenvalues_are_initialized == false, ExcInternalError());

    EigenvalueInformation info;

    if (data.relaxation == 0.0)
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

        const_cast<MyPreconditionRelaxation<MatrixType, PreconditionerType> *>(
          this)
          ->data.relaxation =
          std::min(data.max_relaxation,
                   2.0 / (alpha + info.max_eigenvalue_estimate));
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
