#ifndef lethe_preconditioners_h
#define lethe_preconditioners_h

#include <core/parameters.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#if DEAL_II_VERSION_GTE(9, 8, 0)
#  include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>
#else
#  include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>
#endif

using namespace dealii;

/**
 * @brief A base class of preconditioners used by the smoother.
 */
template <typename VectorType>
class PreconditionBase : public EnableObserverPointer
{
public:
  /**
   * @brief Constructor.
   */
  PreconditionBase()
    : comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
  {}

  /**
   * @brief Apply preconditioner.
   *
   * @param[in,out] dst Destination vector holding the result.
   * @param[in] src Input source vector.
   */
  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  /**
   * @brief Print timers.
   */
  void
  timer_print() const
  {
    timer.print_wall_time_statistics(comm);
  }

  /**
   * @brief Reset timers.
   */
  void
  timer_reset() const
  {
    timer.reset();
  }

protected:
  /// Communicator used for timer output.
  const MPI_Comm comm;

  /// Stream used for timer output.
  ConditionalOStream pcout;

  /// Timer.
  mutable TimerOutput timer;
};

/**
 * @brief A wrapper class around a preconditioner to be
 * able to handle vector types that are not supported
 * by the preconditioner.
 */
template <typename VectorType, typename VectorTypePrecondition>
class PreconditionAdapter : public PreconditionBase<VectorType>
{
public:
  /**
   * Constructor.
   *
   * @param[in] preconditioner Preconditioner.
   */
  template <typename PreconditionType>
  PreconditionAdapter(const std::shared_ptr<PreconditionType> &preconditioner)
    : fu_preconditioner([preconditioner](VectorTypePrecondition       &dst,
                                         const VectorTypePrecondition &src) {
      preconditioner->vmult(dst, src);
    })
  {}

  /**
   * @brief Apply preconditioner.
   *
   * @param[in,out] dst Destination vector holding the result.
   * @param[in] src Input source vector.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    if constexpr (std::is_same_v<VectorType, VectorTypePrecondition>)
      {
        // vector is supported: nothing to do
        fu_preconditioner(dst, src);
      }
    else
      {
        // vector is not supported: copy vectors
        src_ = src;
        dst_ = dst;
        fu_preconditioner(dst_, src_);
        dst = dst_;
      }
  }

  /**
   * @brief Apply preconditioner.
   *
   * @param[in,out] dst Destination vector holding the result.
   * @param[in] src Input source vector.
   */
  template <typename U = VectorTypePrecondition>
    requires(!std::is_same_v<VectorType, U>)
  void
  vmult(U &dst, const U &src) const
  {
    fu_preconditioner(dst, src);
  }

private:
  /// Preconditioner.
  const std::function<void(VectorTypePrecondition &,
                           const VectorTypePrecondition &)>
    fu_preconditioner;

  /// Source vector supported by the preconditioner.
  mutable VectorTypePrecondition src_;

  /// Destination vector supported by the preconditioner.
  mutable VectorTypePrecondition dst_;
};

/**
 * @brief A wrapper class around DiagonalMatrix class from deal.II.
 */
template <typename VectorType>
class MyDiagonalMatrix : public PreconditionBase<VectorType>
{
public:
  /**
   * @brief Constructor.

   * @param[in] diagonal Vector containing the diagonal of the matrix.
   */
  MyDiagonalMatrix(const VectorType &diagonal)
    : diagonal_matrix(diagonal)
  {}

  /**
   * @brief Apply preconditioner.
   *
   * @param[in,out] dst Destination vector holding the result.
   * @param[in] src Input source vector.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    diagonal_matrix.vmult(dst, src);
  }

private:
  /// DiagonalMatrix class from deal.II.
  DiagonalMatrix<VectorType> diagonal_matrix;
};

/**
 * @brief An Additive Schwarz preconditioner.
 */
template <typename VectorType>
class PreconditionASM : public PreconditionBase<VectorType>
{
public:
  /// Value type.
  using Number = typename VectorType::value_type;

  /**
   * @brief Constructor.
   *
   * @param[in] parameters Linear solver parameters containing the smoothing
   * preconditioner options.
   */
  PreconditionASM(const Parameters::LinearSolver &parameters);

  /**
   * @brief Get the local index sets of the patches, as used by vmult().
   */
  const std::vector<std::vector<unsigned int>> &
  get_patches() const;

  /**
   * @brief Get the global DoF index sets of the patches, as computed by
   * initialize() before their conversion to local indices.
   */
  const std::vector<std::vector<types::global_dof_index>> &
  get_global_patches() const;

  /**
   * @brief Get the pointers to the beginning of the internal and boundary
   * velocity and pressure blocks of each patch, in global indices.
   */
  const std::vector<std::vector<unsigned int>> &
  get_global_patches_ptr() const;

  /**
   * @brief Get the number of patches.
   */
  unsigned int
  get_n_patches() const;

  /**
   * @brief Get the weighting type.
   */
  Parameters::LinearSolver::MultigridSMWeightingType
  get_weighting_type() const;

  /**
   * @brief Get the Schwarz type.
   */
  Parameters::LinearSolver::MultigridSMType
  get_schwarz_type() const;

  /**
   * @brief Get the patch type.
   */
  Parameters::LinearSolver::MultigridSMPatchType
  get_patch_type() const;

  /**
   * @brief Get the closure type.
   */
  Parameters::LinearSolver::MultigridSMClosureType
  get_closure_type() const;

  /**
   * @brief Initialize inverses of blocks in two steps: 1) build patches, 2) build blocks and calculate inverse.
   */
  template <int dim>
  void
  initialize(
    const DoFHandler<dim>                &dof_handler,
    const TrilinosWrappers::SparseMatrix &global_sparse_matrix,
    const DynamicSparsityPattern         &global_sparsity_pattern,
    const AffineConstraints<Number>      &constraints,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
    const unsigned int                                        mg_level);

  template <int dim>
  void
  build_patches(
    const DoFHandler<dim>                                    &dof_handler,
    const AffineConstraints<Number>                          &constraints,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
    const unsigned int                                        mg_level);

  void
  build_blocks(const TrilinosWrappers::SparseMatrix &global_sparse_matrix,
               const DynamicSparsityPattern         &global_sparsity_pattern,
               const unsigned int                   &dim);

  /**
   * @brief Apply preconditioner.
   *
   * @param[in,out] dst Destination vector holding the result.
   * @param[in] src Input source vector.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override;

private:
  void
  patch_solve(unsigned int      c,
              VectorType       &dst_ptr,
              const VectorType &src_ptr) const;

  /// DoF indices of patches.
  std::vector<std::vector<unsigned int>> patches;

  /// Inverse of patch matrices.
  std::vector<LAPACKFullMatrix<Number>> blocks;

  /// Global DoF indices of patches, as computed by initialize() before
  /// their conversion to local indices.
  std::vector<std::vector<types::global_dof_index>> global_patches;

  /// Pointers to the beginning of the internal and boundary velocity and
  /// pressure blocks of each patch, in global indices.
  std::vector<std::vector<unsigned int>> global_patches_ptr;

  /// Local copy of sparse matrix
  TrilinosWrappers::SparseMatrix global_sparse_matrix;

  /// Weights.
  mutable VectorType weights;

  /// Weighting type.
  const Parameters::LinearSolver::MultigridSMWeightingType weighting_type;

  /// Schwarz type.
  const Parameters::LinearSolver::MultigridSMType schwarz_type;

  /// Patch type
  const Parameters::LinearSolver::MultigridSMPatchType patch_type;

  /// Closure type
  const Parameters::LinearSolver::MultigridSMClosureType closure_type;

  /// Internal destination vector with correct ghosting.
  mutable VectorType dst_internal;

  /// Internal source vector with correct ghosting.
  mutable VectorType src_internal;

  /// Internal residual vector for multiplicative Schwarz.
  mutable VectorType residual_internal;
};


#endif
