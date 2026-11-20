// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief TODO
 */

#include "asm_lex.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters::LinearSolver parameters;
  parameters.mg_smoother_sm_patch_type =
    Parameters::LinearSolver::MultigridSMPatchType::cell_neighborhood;
  parameters.mg_smoother_sm_closure_type =
    Parameters::LinearSolver::MultigridSMClosureType::full_closure;
  parameters.mg_smoother_sm_weighting_type =
    Parameters::LinearSolver::MultigridSMWeightingType::none;

  for (unsigned int p = 1; p <= 3; ++p)
    {
      test(p, parameters);
      test(p,
           parameters,
           std::vector<types::boundary_id>{0},
           std::vector<types::boundary_id>{});
      test(p,
           parameters,
           std::vector<types::boundary_id>{},
           std::vector<types::boundary_id>{0});
    }

  return 0;
}
