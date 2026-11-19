// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief Check indices for cell neighborhood patches considering Q1 elements for velocity and pressure in a 4x4 grid.
 */

#include <core/preconditioners.h>

#include <deal.II/grid/grid_generator.h>


using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 2;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  FESystem<dim>   fe(FE_Q<dim>(1), dim + 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Initialize constraints
  const IndexSet locally_owned = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Define constraints
  AffineConstraints<double> constraints;
  constraints.reinit(locally_owned, locally_relevant);

  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0, // Boundary ID
                                           Functions::ZeroFunction<dim>(dim +
                                                                        1),
                                           constraints,
                                           fe.component_mask(velocities));
  constraints.close();

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  Parameters::LinearSolver parameters;
  parameters.mg_smoother_sm_patch_type =
    Parameters::LinearSolver::MultigridSMPatchType::cell_neighborhood;
  parameters.mg_smoother_sm_closure_type =
    Parameters::LinearSolver::MultigridSMClosureType::full_closure;
  parameters.mg_smoother_sm_weighting_type =
    Parameters::LinearSolver::MultigridSMWeightingType::none;

  PreconditionASM<LinearAlgebra::distributed::Vector<double>> preconditioner(
    parameters);

  preconditioner.build_patches(dof_handler,
                               constraints,
                               partitioner,
                               numbers::invalid_unsigned_int);

  // Get global patches and global patches ptr
  const std::vector<std::vector<types::global_dof_index>> &patches =
    preconditioner.get_global_patches();

  const std::vector<std::vector<unsigned int>> &patches_ptr =
    preconditioner.get_global_patches_ptr();

  AssertThrow(patches.size() == preconditioner.get_n_patches() &&
                patches_ptr.size() == patches.size(),
              ExcMessage("Inconsistent number of patches"));

  // Rank-invariant aggregates
  double sizes_sum = 0.0;
  double sizes_min = std::numeric_limits<double>::max();
  double sizes_max = 0.0;
  double index_sum = 0.0;
  for (const auto &patch : patches)
    {
      sizes_sum += patch.size();
      sizes_min = std::min(sizes_min, static_cast<double>(patch.size()));
      sizes_max = std::max(sizes_max, static_cast<double>(patch.size()));
      for (const auto i : patch)
        index_sum += i;
    }

  std::vector<types::global_dof_index> flat_patches;
  for (unsigned int p = 0; p < patches.size(); ++p)
    {
      flat_patches.emplace_back(
        static_cast<types::global_dof_index>(patches[p].size()));
      flat_patches.insert(flat_patches.end(),
                          patches_ptr[p].begin(),
                          patches_ptr[p].end());
      flat_patches.insert(flat_patches.end(),
                          patches[p].begin(),
                          patches[p].end());
    }


  // Gather the encoded patches of all ranks on rank 0 (collective call)
  const std::vector<std::vector<types::global_dof_index>> all_flat_patches =
    Utilities::MPI::gather(MPI_COMM_WORLD, flat_patches);

  const unsigned int n_patches_reduced =
    Utilities::MPI::sum<unsigned int>(patches.size(), MPI_COMM_WORLD);
  const double sizes_sum_reduced =
    Utilities::MPI::sum(sizes_sum, MPI_COMM_WORLD);
  const double sizes_min_reduced =
    Utilities::MPI::min(sizes_min, MPI_COMM_WORLD);
  const double sizes_max_reduced =
    Utilities::MPI::max(sizes_max, MPI_COMM_WORLD);
  const double index_sum_reduced =
    Utilities::MPI::sum(index_sum, MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "n_processes: "
                << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
      std::cout << "n_dofs: " << dof_handler.n_dofs() << std::endl;
      std::cout << "n_active_cells: " << triangulation.n_global_active_cells()
                << std::endl;
      std::cout << "n_patches: " << n_patches_reduced << std::endl;
      std::cout << "patch sizes sum/min/max: " << sizes_sum_reduced << " / "
                << sizes_min_reduced << " / " << sizes_max_reduced << std::endl;
      std::cout << "index sum: " << std::setprecision(12) << index_sum_reduced
                << std::endl;

      // Print the indices of every patch of every rank, grouped by
      // velocity and pressure component with the pointer array
      for (unsigned int rank = 0; rank < all_flat_patches.size(); ++rank)
        {
          const std::vector<types::global_dof_index> &flat =
            all_flat_patches[rank];

          std::vector<types::global_dof_index>::const_iterator it =
            flat.begin();
          for (unsigned int p = 0; it != flat.end(); ++p)
            {
              const types::global_dof_index size = *it++;

              const std::vector<unsigned int> ptr(it, it + dim + 2);
              it += dim + 2;

              std::cout << "patch (rank " << rank << ", #" << p << "): size "
                        << size << std::endl;
              for (unsigned int c = 0; c <= dim; ++c)
                {
                  std::cout
                    << "  component " << c << " ("
                    << (c < static_cast<unsigned int>(dim) ? "velocity" :
                                                             "pressure")
                    << "):";
                  for (unsigned int i = ptr[c]; i < ptr[c + 1]; ++i)
                    std::cout << ' ' << it[i];
                  std::cout << std::endl;
                }

              it += size;
            }
        }
    }

  return 0;
}