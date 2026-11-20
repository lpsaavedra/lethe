
#include <core/preconditioners.h>

namespace
{
  FullMatrix<double>
  compute_schur_complement_matrix(const FullMatrix<double>        &matrix,
                                  const std::vector<unsigned int> &patch_ptr,
                                  const bool create_block_diagonal)
  {
    const unsigned int dim = patch_ptr.size() - 2;
    const unsigned int n_u = patch_ptr[dim];
    const unsigned int n   = matrix.m();
    const unsigned int n_p = n - n_u;

    AssertDimension(matrix.m(), matrix.n());
    AssertIndexRange(n_u, n);

    // Extract A, B, C, D.
    FullMatrix<double> A(n_u, n_u);
    FullMatrix<double> B(n_u, n_p);
    FullMatrix<double> C(n_p, n_u);
    FullMatrix<double> D(n_p, n_p);

    for (unsigned int i = 0; i < n_u; ++i)
      for (unsigned int j = 0; j < n_u; ++j)
        A(i, j) = matrix(i, j);

    for (unsigned int i = 0; i < n_u; ++i)
      for (unsigned int j = 0; j < n_p; ++j)
        B(i, j) = matrix(i, n_u + j);

    for (unsigned int i = 0; i < n_p; ++i)
      for (unsigned int j = 0; j < n_u; ++j)
        C(i, j) = matrix(n_u + i, j);

    for (unsigned int i = 0; i < n_p; ++i)
      for (unsigned int j = 0; j < n_p; ++j)
        D(i, j) = matrix(n_u + i, n_u + j);

    // Compute X = A^{-1} B using LU factorization.
    LAPACKFullMatrix<double> A_lu(n_u, n_u);
    A_lu.copy_from(A);
    A_lu.compute_lu_factorization();

    FullMatrix<double> X(n_u, n_p);

    for (unsigned int j = 0; j < n_p; ++j)
      {
        Vector<double> rhs(n_u);

        for (unsigned int i = 0; i < n_u; ++i)
          rhs[i] = B(i, j);

        A_lu.solve(rhs);

        for (unsigned int i = 0; i < n_u; ++i)
          X(i, j) = rhs[i];
      }

    // Compute CX = C * X.
    FullMatrix<double> CX(n_p, n_p);
    C.mmult(CX, X);

    // Compute S = D - C * A^{-1} * B.
    FullMatrix<double> S(n_p, n_p);
    S.copy_from(D);
    S.add(-1.0, CX);

    // Copy original matrix.
    FullMatrix<double> result(matrix);

    // Zero lower-left block C.
    for (unsigned int i = n_u; i < n; ++i)
      for (unsigned int j = 0; j < n_u; ++j)
        result(i, j) = 0.0;

    // Optionally zero upper-right block B.
    if (create_block_diagonal)
      for (unsigned int i = 0; i < n_u; ++i)
        for (unsigned int j = n_u; j < n; ++j)
          result(i, j) = 0.0;

    // Replace D with Schur complement S.
    for (unsigned int i = 0; i < n_p; ++i)
      for (unsigned int j = 0; j < n_p; ++j)
        result(n_u + i, n_u + j) = S(i, j);

    return result;
  }
} // namespace

template <typename VectorType>
PreconditionASM<VectorType>::PreconditionASM(
  const Parameters::LinearSolver &parameters)
  : weighting_type(parameters.mg_smoother_sm_weighting_type)
  , schwarz_type(parameters.mg_smoother_sm_type)
  , patch_type(parameters.mg_smoother_sm_patch_type)
  , closure_type(parameters.mg_smoother_sm_closure_type)
{}

template <typename VectorType>
const std::vector<std::vector<unsigned int>> &
PreconditionASM<VectorType>::get_patches() const
{
  return this->patches;
}

template <typename VectorType>
const std::vector<std::vector<types::global_dof_index>> &
PreconditionASM<VectorType>::get_global_patches() const
{
  return this->global_patches;
}

template <typename VectorType>
const std::vector<std::vector<unsigned int>> &
PreconditionASM<VectorType>::get_global_patches_ptr() const
{
  return this->global_patches_ptr;
}

template <typename VectorType>
unsigned int
PreconditionASM<VectorType>::get_n_patches() const
{
  return this->patches.size();
}

template <typename VectorType>
Parameters::LinearSolver::MultigridSMWeightingType
PreconditionASM<VectorType>::get_weighting_type() const
{
  return this->weighting_type;
}

template <typename VectorType>
Parameters::LinearSolver::MultigridSMType
PreconditionASM<VectorType>::get_schwarz_type() const
{
  return this->schwarz_type;
}

template <typename VectorType>
Parameters::LinearSolver::MultigridSMPatchType
PreconditionASM<VectorType>::get_patch_type() const
{
  return this->patch_type;
}

template <typename VectorType>
Parameters::LinearSolver::MultigridSMClosureType
PreconditionASM<VectorType>::get_closure_type() const
{
  return this->closure_type;
}

template <typename VectorType>
template <int dim>
void
PreconditionASM<VectorType>::initialize(
  const DoFHandler<dim>                &dof_handler,
  const TrilinosWrappers::SparseMatrix &global_sparse_matrix,
  const DynamicSparsityPattern         &global_sparsity_pattern,
  const AffineConstraints<typename PreconditionASM<VectorType>::Number>
                                                           &constraints,
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
  const unsigned int                                        mg_level)
{
  build_patches(dof_handler, constraints, partitioner, mg_level);

  build_blocks(global_sparse_matrix, global_sparsity_pattern, dim);
}

template <typename VectorType>
template <int dim>
void
PreconditionASM<VectorType>::build_patches(
  const DoFHandler<dim> &dof_handler,
  const AffineConstraints<typename PreconditionASM<VectorType>::Number>
                                                           &constraints,
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
  const unsigned int                                        mg_level)
{
  // TODO: make work (1) in parallel, (2) on multigrid levels, (3) on
  // locally refined meshes

  this->timer.enter_subsection("asm::indices");

  std::vector<std::vector<types::global_dof_index>> patches;

  std::vector<std::set<typename DoFHandler<dim>::active_cell_iterator>>
    patch_to_cells;

  // 1) build patches
  if (patch_type == Parameters::LinearSolver::MultigridSMPatchType::cell)
    {
      AssertThrow(
        closure_type ==
          Parameters::LinearSolver::MultigridSMClosureType::full_closure,
        ExcNotImplemented());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          std::set<typename DoFHandler<dim>::active_cell_iterator> tmp;
          tmp.insert(cell);
          patch_to_cells.emplace_back(tmp);
        }
    }
  else if (patch_type ==
           Parameters::LinearSolver::MultigridSMPatchType::cell_neighborhood)
    {
      std::vector<std::set<typename DoFHandler<dim>::active_cell_iterator>>
        vertex_to_cells(dof_handler.get_triangulation().n_vertices());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial())
            continue;

          for (const auto v : cell->vertex_indices())
            vertex_to_cells[cell->vertex_index(v)].insert(cell);
        }


      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          std::set<typename DoFHandler<dim>::active_cell_iterator> tmp;

          for (const auto v : cell->vertex_indices())
            for (const auto &c : vertex_to_cells[cell->vertex_index(v)])
              tmp.insert(c);
          patch_to_cells.emplace_back(tmp);
        }
    }
  else if (patch_type ==
           Parameters::LinearSolver::MultigridSMPatchType::vertex_star)
    {
      patch_to_cells.resize(dof_handler.get_triangulation().n_vertices());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial())
            continue;

          for (const auto v : cell->vertex_indices())
            patch_to_cells[cell->vertex_index(v)].insert(cell);
        }

      for (auto &cells : patch_to_cells)
        {
          if (cells.empty())
            continue;

          auto min_rank = dealii::numbers::invalid_subdomain_id;

          for (const auto &cell : cells)
            if (!cell->is_artificial())
              min_rank = std::min(min_rank, cell->subdomain_id());

          if (min_rank != Utilities::MPI::this_mpi_process(
                            dof_handler.get_mpi_communicator()))
            cells.clear();
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  std::vector<std::vector<unsigned int>> patches_ptr;

  // 2) dof indices
  for (const auto &cells : patch_to_cells)
    {
      if (cells.empty())
        continue;

      const auto &fe = dof_handler.get_fe();

      std::vector<std::set<types::global_dof_index>> internal_indices(
        fe.n_components());
      std::vector<std::set<types::global_dof_index>> boundary_indices(
        fe.n_components());

      // boundary patch indices
      for (const auto &cell : cells)
        {
          for (const auto f : cell->face_indices())
            if (!cell->at_boundary(f) &&
                (cells.find(cell->neighbor(f)) == cells.end()))
              {
                std::vector<types::global_dof_index> local_dof_indices(
                  cell->get_fe().n_dofs_per_face());
                cell->face(f)->get_dof_indices(local_dof_indices);

                for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                  boundary_indices[fe.face_system_to_component_index(i, f)
                                     .first]
                    .insert(local_dof_indices[i]);
              }
        }

      // internal patch indices
      for (const auto &cell : cells)
        {
          std::vector<types::global_dof_index> local_dof_indices(
            cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
            {
              const unsigned int comp = fe.system_to_component_index(i).first;

              if (boundary_indices[comp].find(local_dof_indices[i]) ==
                  boundary_indices[comp].end())
                internal_indices[comp].insert(local_dof_indices[i]);
            }
        }

      if (closure_type ==
          Parameters::LinearSolver::MultigridSMClosureType::interior)
        {
          for (unsigned int c = 0; c <= dim; ++c)
            boundary_indices[c].clear();
        }
      else if (closure_type == Parameters::LinearSolver::
                                 MultigridSMClosureType::velocity_closure)
        {
          boundary_indices[dim].clear();
        }
      else if (closure_type ==
               Parameters::LinearSolver::MultigridSMClosureType::full_closure)
        {
          // nothing to do
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      std::vector<types::global_dof_index> current_patch;
      std::vector<unsigned int>            current_patch_ptr = {{0}};

      const auto add_indices = [&](const auto &local_dof_indices) {
        for (const auto i : local_dof_indices)
          if (!constraints.is_constrained(i))
            current_patch.emplace_back(i);
      };


      for (unsigned int c = 0; c <= dim; ++c)
        {
          add_indices(internal_indices[c]);
          add_indices(boundary_indices[c]);

          current_patch_ptr.emplace_back(current_patch.size());
        }

      if (!current_patch.empty())
        {
          patches.emplace_back(current_patch);
          patches_ptr.emplace_back(current_patch_ptr);
        }
    }

  const IndexSet locally_owned_dofs =
    (mg_level == numbers::invalid_unsigned_int) ?
      dof_handler.locally_owned_dofs() :
      dof_handler.locally_owned_mg_dofs(mg_level);

  IndexSet ghost_dofs(locally_owned_dofs.size());
  for (const auto &indices : patches)
    ghost_dofs.add_indices(indices.begin(), indices.end());

  std::shared_ptr<const Utilities::MPI::Partitioner> partition =
    std::make_shared<Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                  ghost_dofs,
                                                  MPI_COMM_WORLD);

  if (dealii::internal::is_partitioner_contained(partition, partitioner))
    {
      partition = partitioner;
    }
  else
    {
      src_internal.reinit(partition);
      dst_internal.reinit(partition);
    }

  // Store global patch data for tests
  global_patches     = patches;
  global_patches_ptr = patches_ptr;

  // convert indices to local ones
  this->patches.resize(patches.size());
  for (unsigned int c = 0; c < patches.size(); ++c)
    {
      this->patches[c].resize(0);
      this->patches[c].reserve(patches[c].size());

      for (const auto &i : patches[c])
        this->patches[c].emplace_back(partition->global_to_local(i));
    }

  this->timer.leave_subsection("asm::indices");

  if (weighting_type !=
      Parameters::LinearSolver::MultigridSMWeightingType::none)
    {
      this->timer.enter_subsection("asm::weight");
      Vector<Number> vector_weights;
      weights.reinit(partition);

      for (const auto &patch : patches)
        {
          const unsigned int dofs_per_cell = patch.size();
          vector_weights.reinit(dofs_per_cell);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_weights[i] = 1.0;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            weights[patch[i]] += vector_weights[i];
        }

      weights.compress(VectorOperation::add);
      for (auto &i : weights)
        i = (weighting_type ==
             Parameters::LinearSolver::MultigridSMWeightingType::symm) ?
              std::sqrt(1.0 / i) :
              (1.0 / i);
      weights.update_ghost_values();

      this->timer.leave_subsection("asm::weight");
    }
}


template <typename VectorType>
void
PreconditionASM<VectorType>::build_blocks(
  const TrilinosWrappers::SparseMatrix &global_sparse_matrix,
  const DynamicSparsityPattern         &global_sparsity_pattern,
  const unsigned int                   &dim)
{
  this->global_sparse_matrix.copy_from(global_sparse_matrix);

  this->timer.enter_subsection("asm::restrict");

  std::vector<FullMatrix<Number>> blocks;

  // The deprecation warning is triggered internally by deal.II: the call
  // below instantiates SparseMatrixTools code that still relies on the
  // deprecated ConsensusAlgorithms::Process class. Suppress it locally until
  // it is resolved upstream.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  SparseMatrixTools::restrict_to_full_matrices(global_sparse_matrix,
                                               global_sparsity_pattern,
                                               this->global_patches,
                                               blocks);
#pragma GCC diagnostic pop

  this->timer.leave_subsection("asm::restrict");
  this->timer.enter_subsection("asm::invert");

  this->blocks.resize(blocks.size());

  for (unsigned int b = 0; b < blocks.size(); ++b)
    {
      if (false) // TODO: introduce parameter
        {
          const double tol = 1e-8;

          if (true)
            {
              blocks[b] =
                compute_schur_complement_matrix(blocks[b],
                                                this->global_patches_ptr[b],
                                                false);
            }

          if (false)
            {
              for (unsigned int i = 0; i < this->global_patches_ptr[b][dim];
                   ++i)
                for (unsigned int j = 0; j < this->global_patches_ptr[b][dim];
                     ++j)
                  if (i != j)
                    blocks[b][i][j] = 0.0;
            }
          else if (true)
            {
              for (unsigned int i = 0; i < this->global_patches_ptr[b][dim];
                   ++i)
                for (unsigned int j = 0; j < this->global_patches_ptr[b][dim];
                     ++j)
                  if (i != j)
                    blocks[b][i][j] = 0.0;
              for (unsigned int i = this->global_patches_ptr[b][dim];
                   i < this->global_patches_ptr[b][dim + 1];
                   ++i)
                for (unsigned int j = this->global_patches_ptr[b][dim];
                     j < this->global_patches_ptr[b][dim + 1];
                     ++j)
                  if (i != j)
                    blocks[b][i][j] = 0.0;
            }
          else if (false)
            {
              for (unsigned int i = this->global_patches_ptr[b][dim];
                   i < this->global_patches_ptr[b][dim + 1];
                   ++i)
                for (unsigned int j = this->global_patches_ptr[b][dim];
                     j < this->global_patches_ptr[b][dim + 1];
                     ++j)
                  blocks[b][i][j] = 0.0;
            }

          for (unsigned int i = 0; i < blocks[b].m(); ++i)
            for (unsigned int j = 0; j < blocks[b].n(); ++j)
              if (std::abs(blocks[b][i][j]) < tol)
                blocks[b][i][j] = 0.0;

          if (true) // TODO: create statistics
            {
              unsigned int nnz = 0;

              for (unsigned int i = 0; i < blocks[b].m(); ++i)
                for (unsigned int j = 0; j < blocks[b].n(); ++j)
                  if (blocks[b][i][j] != 0.0)
                    nnz++;
              std::cout << blocks[b].m() * blocks[b].n() << " -> " << nnz
                        << std::endl;
            }
        }

      this->blocks[b] = LAPACKFullMatrix<Number>(blocks[b].m(), blocks[b].n());
      this->blocks[b] = blocks[b];
      this->blocks[b].compute_lu_factorization();
    }

  this->timer.leave_subsection("asm::invert");
}

template <typename VectorType>
void
PreconditionASM<VectorType>::vmult(VectorType &dst, const VectorType &src) const
{
  this->timer.enter_subsection("asm::vmult");

  const auto &src_ptr = (src_internal.size() != 0) ? src_internal : src;
  auto       &dst_ptr = (dst_internal.size() != 0) ? dst_internal : dst;

  dst_ptr = 0.0;
  if (src_internal.size() != 0)
    src_internal.copy_locally_owned_data_from(src);
  src_ptr.update_ghost_values();

  for (unsigned int c = 0; c < patches.size(); ++c)
    {
      if (schwarz_type == Parameters::LinearSolver::MultigridSMType::additive)
        {
          patch_solve(c, dst_ptr, src_ptr);
        }
      else if (schwarz_type ==
               Parameters::LinearSolver::MultigridSMType::multiplicative)
        {
          // Update the residual vector
          residual_internal.reinit(src_ptr, true);

          global_sparse_matrix.vmult(residual_internal, dst_ptr);

          residual_internal.sadd(-1.0, 1.0, src_ptr);

          residual_internal.update_ghost_values();

          patch_solve(c, dst_ptr, residual_internal);
        }
    }

  src_ptr.zero_out_ghost_values();
  dst_ptr.compress(VectorOperation::add);

  if (dst_internal.size() != 0)
    dst.copy_locally_owned_data_from(dst_internal);

  this->timer.leave_subsection("asm::vmult");
}

template <typename VectorType>
void
PreconditionASM<VectorType>::patch_solve(unsigned int      c,
                                         VectorType       &dst_ptr,
                                         const VectorType &src_ptr) const
{
  Vector<Number> vector_src, vector_dst, vector_weights;

  const unsigned int dofs_per_cell = patches[c].size();

  vector_src.reinit(dofs_per_cell);
  vector_dst.reinit(dofs_per_cell);
  if (weighting_type !=
      Parameters::LinearSolver::MultigridSMWeightingType::none)
    vector_weights.reinit(dofs_per_cell);

  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    vector_src[i] = src_ptr.local_element(patches[c][i]);

  if (weighting_type !=
      Parameters::LinearSolver::MultigridSMWeightingType::none)
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      vector_weights[i] = weights.local_element(patches[c][i]);

  if (weighting_type ==
        Parameters::LinearSolver::MultigridSMWeightingType::symm ||
      weighting_type ==
        Parameters::LinearSolver::MultigridSMWeightingType::right)
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      vector_src[i] *= vector_weights[i];

  blocks[c].solve(vector_src);
  vector_dst = vector_src;

  if (weighting_type ==
        Parameters::LinearSolver::MultigridSMWeightingType::symm ||
      weighting_type ==
        Parameters::LinearSolver::MultigridSMWeightingType::left)
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      vector_dst[i] *= vector_weights[i];

  for (unsigned int i = 0; i < dofs_per_cell; ++i)
    dst_ptr.local_element(patches[c][i]) += vector_dst[i];
}

template class PreconditionASM<
  dealii::LinearAlgebra::distributed::Vector<double>>;

template void
PreconditionASM<dealii::LinearAlgebra::distributed::Vector<double>>::
  build_patches<2>(
    const DoFHandler<2> &,
    const AffineConstraints<PreconditionASM<
      dealii::LinearAlgebra::distributed::Vector<double>>::Number> &,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &,
    const unsigned int);

template void
PreconditionASM<dealii::LinearAlgebra::distributed::Vector<double>>::
  build_patches<3>(
    const DoFHandler<3> &,
    const AffineConstraints<PreconditionASM<
      dealii::LinearAlgebra::distributed::Vector<double>>::Number> &,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &,
    const unsigned int);

template void
PreconditionASM<dealii::LinearAlgebra::distributed::Vector<double>>::initialize<
  2>(const DoFHandler<2> &,
     const TrilinosWrappers::SparseMatrix &,
     const DynamicSparsityPattern &,
     const AffineConstraints<PreconditionASM<
       dealii::LinearAlgebra::distributed::Vector<double>>::Number> &,
     const std::shared_ptr<const Utilities::MPI::Partitioner> &,
     const unsigned int);

template void
PreconditionASM<dealii::LinearAlgebra::distributed::Vector<double>>::initialize<
  3>(const DoFHandler<3> &,
     const TrilinosWrappers::SparseMatrix &,
     const DynamicSparsityPattern &,
     const AffineConstraints<PreconditionASM<
       dealii::LinearAlgebra::distributed::Vector<double>>::Number> &,
     const std::shared_ptr<const Utilities::MPI::Partitioner> &,
     const unsigned int);
