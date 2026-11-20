// SPDX-FileCopyrightText: Copyright (c) 2026 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/**
 * @brief Check indices for cell_neighborhoodes for velocity and pressure in a 4x4 grid for Q1, Q2 and Q3 elements.
 */

#include "asm_patches_indices.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters::LinearSolver parameters;
  parameters.mg_smoother_sm_patch_type =
    Parameters::LinearSolver::MultigridSMPatchType::cell_neighborhood;
  parameters.mg_smoother_sm_weighting_type =
    Parameters::LinearSolver::MultigridSMWeightingType::none;

  for (const auto closure_type : magic_enum::enum_values<
         Parameters::LinearSolver::MultigridSMClosureType>())
    {
      parameters.mg_smoother_sm_closure_type = closure_type;

      for (unsigned int fe_degree = 1; fe_degree < 4; ++fe_degree)
        {
          patches_indices_test(fe_degree, parameters);
        }
    }


  return 0;
}
