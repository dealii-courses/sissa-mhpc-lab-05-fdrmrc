#  Lab 05 - Boundary conditions and constraints
## Theory and Practice of Finite Elements

**Luca Heltai** <luca.heltai@sissa.it>

* * * * *

## General Instructions

For each of the point below, extend the `Poisson` class with functions that
perform the indicated tasks, trying to minimize the amount of code you copy and
paste, possibly restructuring existing code by adding arguments to existing
functions, and generating wrappers similar to the `run` method (e.g.,
`run_exercise_3`).

Once you created a function that performs the given task, add it to the
`poisson-tester.cc` file, and make sure all the exercises are run through the
`gtest` executable, e.g., adding a test for each exercise, as in the following
snippet:

```C++
TEST_F(PoissonTester, Exercise3) {
   run_exercise_3();
}
```

By the end of this laboratory, you will have modified your Poisson code to
allow also non-homogeneous Neumann boundary conditions on different parts of
the domain, and you will have added some more options to the solver, enabling
usage of a direct solver, or of some more sofisticated preconditioners.

## Lab-05

### step-5

1. Add the parameters
   
    - `Neumann boundary condition expression`
    - `Dirichlet boundary ids`
    - `Neumann boundary ids`
   
   and the corresponding member variables to your Poisson problem problem, using 
   `std::set<dealii::types::boundary_id>` for the last two parameters

2. Modify your implementation of Dirichlet boundary conditions, in order to
apply the Dirichlet function to all boundary ids indicated in the parameter
file

3. Implement Neumann boundary conditions, using the function defined above, on
the ids of the Neumann boundary indicated in the parameter file

4. Create a function that parses parameters from a string, to be used in the
testing infrastructure

5. Create a few tests that actually solve a Poisson problem on very small grids
with very simple but non-trivial combinations of boundary conditions, on
different domains, and verify the correctness of your findings. Examples
include using globally linear (quadratic) exact functions, with linear
(quadratic) finite elements, and verify that the error you make is actually
zero (in this case, global interpolation gives the exact solution, therefore
the finite element should also provide the exact solution)

6. Modify your code to use `AffineConstraints` instead of
`VectorTools::apply_boundary_values` (see the documentation of `step-6`). Run
again all tests, and verify that you pass your own checks again with the new
code

7. Add the parameters:
    - `Local pre-refinement grid size expression`
   
   and the corresponding members. When creating the grid, instead of simply
   refining globally a fixed number of times (given by the parameter 
   `Number of global refinements`), refine locally your grid when the function
   above evaluated in the center of a cell is larger then the actual cell 
   diameter. Make sure you stop refining locally if the number of local
   refinement cycles you did is equal to the parameter 
   `Number of global refinements`. Setting the function above to `0` should produce the same results as before, i.e., a fixed number of global refinements up to `Number of global refinements`.

8. Make sure you compute correctly the hanging node constraints, and that your
solver works also with hanging nodes

9. Add a preconditioner to your solver. If you have trilinos installed and it
is configured inside `deal.II`, use its algebraic multigrid preconditioner (it
works also with `deal.II` matrices). Otherwise use one of the other available
preconditioners in the library. Verify that your solver is now faster.