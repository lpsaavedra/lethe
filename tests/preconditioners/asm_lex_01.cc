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

  std::vector<Point<dim>> points;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        points.emplace_back(cell->center());
      }

  const auto all_points = Utilities::MPI::gather(MPI_COMM_WORLD, points, 0);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           ++i)
        {
          std::cout << "rank " << i << ":" << std::endl;

          for (const auto &point : all_points[i])
            std::cout << point << std::endl;
          std::cout << std::endl;
        }
    }

  return 0;
}