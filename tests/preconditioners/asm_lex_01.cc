// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief TODO
 */

#include <core/preconditioners.h>

#include <deal.II/grid/grid_generator.h>


using namespace dealii;

template <int dim>
std::vector<types::global_dof_index>
compute_renumbering(const DoFHandler<dim> &dof_handler, const unsigned int root)
{
  std::vector<types::global_dof_index> result;

  if (Utilities::MPI::this_mpi_process(dof_handler.get_mpi_communicator()) ==
      root)
    {
      result.resize(dof_handler.n_dofs());

      for (unsigned int i = 0; i < result.size(); ++i)
        result[i] = i;
    }

  return result;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim                  = 2;
  const unsigned int n_global_refinements = 2;
  const unsigned int fe_degree            = 1;
  const unsigned int root_process         = 0;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(n_global_refinements);

  FESystem<dim>   fe(FE_Q<dim>(fe_degree), dim + 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::vector<std::vector<types::global_dof_index>> patches;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        std::vector<types::global_dof_index> patch(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(patch);
        patches.emplace_back(patch);
      }

  const auto all_patches =
    Utilities::MPI::gather(MPI_COMM_WORLD, patches, root_process);

  const auto renumbering = compute_renumbering(dof_handler, root_process);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == root_process)
    {
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           ++i)
        {
          std::cout << "rank " << i << ":" << std::endl;

          for (const auto &patch : all_patches[i])
            {
              auto temp = patch;
              for (auto &i : temp)
                i = renumbering[i];

              std::sort(temp.begin(), temp.end());

              for (const auto &i : temp)
                std::cout << i << " ";
              std::cout << std::endl;
            }
          std::cout << std::endl;
        }
    }

  return 0;
}