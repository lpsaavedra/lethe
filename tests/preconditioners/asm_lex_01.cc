// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief TODO
 */

#include <core/preconditioners.h>

#include <deal.II/grid/grid_generator.h>


using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim                  = 2;
  const unsigned int n_global_refinements = 2;
  const unsigned int fe_degree            = 1;

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

  const auto all_patches = Utilities::MPI::gather(MPI_COMM_WORLD, patches, 0);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           ++i)
        {
          std::cout << "rank " << i << ":" << std::endl;

          for (const auto &patch : all_patches[i])
            {
              for (const auto &i : patch)
                std::cout << i << " ";
              std::cout << std::endl;
            }
          std::cout << std::endl;
        }
    }

  return 0;
}