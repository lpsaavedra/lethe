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
  MappingQ1<dim> mapping;

  std::vector<types::global_dof_index> result;

  std::vector<std::tuple<types::global_dof_index, unsigned int, Point<dim>>>
    points;

  const auto &fe = dof_handler.get_fe();

  std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

  FEValues<dim> fe_values(mapping,
                          fe,
                          Quadrature<dim>(fe.get_unit_support_points()),
                          update_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);

        for (unsigned int i = 0; i < dof_indices.size(); ++i)
          points.emplace_back(dof_indices[i],
                              fe.system_to_component_index(i).first,
                              fe_values.quadrature_point(i));
      }

  const auto all_points_per_rank =
    Utilities::MPI::gather(dof_handler.get_mpi_communicator(), points, root);

  if (Utilities::MPI::this_mpi_process(dof_handler.get_mpi_communicator()) ==
      root)
    {
      result.resize(dof_handler.n_dofs());

      std::vector<std::tuple<types::global_dof_index, unsigned int, Point<dim>>>
        all_points;

      for (const auto &i : all_points_per_rank)
        all_points.insert(all_points.end(), i.begin(), i.end());

      std::sort(all_points.begin(),
                all_points.end(),
                [](const auto &a, const auto &b) {
                  return std::get<0>(a) < std::get<0>(b);
                });
      all_points.erase(std::unique(all_points.begin(),
                                   all_points.end(),
                                   [](const auto &a, const auto &b) {
                                     return std::get<0>(a) == std::get<0>(b);
                                   }),
                       all_points.end());

      AssertDimension(all_points.size(), result.size());

      std::sort(all_points.begin(),
                all_points.end(),
                [](const auto &a, const auto &b) {
                  if (std::get<1>(a) != std::get<1>(b))
                    return std::get<1>(a) < std::get<1>(b);

                  std::vector<double> a_(dim);
                  std::vector<double> b_(dim);

                  std::get<2>(a).unroll(a_.begin(), a_.end());
                  std::reverse(a_.begin(), a_.end());

                  std::get<2>(b).unroll(b_.begin(), b_.end());
                  std::reverse(b_.begin(), b_.end());

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      if (std::abs(a_[d] - b_[d]) > 1e-8 /*epsilon*/)
                        return a_[d] < b_[d];
                    }

                  return true;
                });

      for (unsigned int i = 0; i < all_points.size(); ++i)
        result[get<0>(all_points[i])] = i;
    }

  return result;
}

void
test(const unsigned int                     fe_degree,
     const std::vector<types::boundary_id> &velocity_bid = {},
     const std::vector<types::boundary_id> &pressure_bid = {})
{
  const unsigned int dim                  = 2;
  const unsigned int n_global_refinements = 2;
  const unsigned int root_process         = 0;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0, 1, true);
  triangulation.refine_global(n_global_refinements);

  FESystem<dim>   fe(FE_Q<dim>(fe_degree), dim + 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));

  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto bid : velocity_bid)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             bid,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             constraints,
                                             fe.component_mask(velocities));
  for (const auto bid : pressure_bid)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             bid,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             constraints,
                                             fe.component_mask(pressure));
  constraints.close();

  std::vector<std::tuple<Point<dim>, std::vector<types::global_dof_index>>>
    patches;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        std::vector<types::global_dof_index> patch(
          cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(patch);

        std::vector<types::global_dof_index> temp;
        for (const auto i : patch)
          if (!constraints.is_constrained(i))
            temp.emplace_back(i);

        patches.emplace_back(cell->center(), temp);
      }

  const auto all_patches =
    Utilities::MPI::gather(MPI_COMM_WORLD, patches, root_process);

  const auto renumbering = compute_renumbering(dof_handler, root_process);

  const auto check_indices =
    [&](const Point<dim>                           &cell_mid_point,
        const std::vector<types::global_dof_index> &cell_dof_indices) {
      constexpr unsigned int n_cells = 1u << n_global_refinements;

      const unsigned int p = fe_degree;
      const unsigned int N = n_cells * p + 1;

      const unsigned int cx =
        static_cast<unsigned int>(cell_mid_point[0] * n_cells);
      const unsigned int cy =
        static_cast<unsigned int>(cell_mid_point[1] * n_cells);

      std::vector<types::global_dof_index> cell_dof_indices_reference;

      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int j = 0; j <= p; ++j)
          for (unsigned int i = 0; i <= p; ++i)
            {
              const unsigned int ix = cx * p + i;
              const unsigned int iy = cy * p + j;

              const auto &bids = (c < dim) ? velocity_bid : pressure_bid;

              const auto has_bid = [&](const types::boundary_id bid) {
                return std::find(bids.begin(), bids.end(), bid) != bids.end();
              };

              // Skip constrained boundary DoFs
              if ((ix == 0 && has_bid(0)) || (ix == N - 1 && has_bid(1)) ||
                  (iy == 0 && has_bid(2)) || (iy == N - 1 && has_bid(3)))
                continue;

              cell_dof_indices_reference.push_back(c * N * N + iy * N + ix);
            }

      AssertThrow(cell_dof_indices == cell_dof_indices_reference,
                  ExcInternalError());
    };

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == root_process)
    {
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           ++i)
        {
          std::cout << "rank " << i << ":" << std::endl;

          for (const auto &[point, indices] : all_patches[i])
            {
              auto temp = indices;
              for (auto &i : temp)
                i = renumbering[i];

              std::sort(temp.begin(), temp.end());

              check_indices(point, temp);

              for (const auto &i : temp)
                std::cout << i << " ";
              std::cout << std::endl;
            }
          std::cout << std::endl;
        }
    }

  std::cout << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  for (unsigned int p = 1; p <= 3; ++p)
    {
      test(p);
      test(p,
           std::vector<types::boundary_id>{0},
           std::vector<types::boundary_id>{});
      test(p,
           std::vector<types::boundary_id>{},
           std::vector<types::boundary_id>{0});
    }

  return 0;
}