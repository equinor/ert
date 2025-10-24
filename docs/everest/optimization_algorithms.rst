.. _cha_optimization_algorithms:

***********************
Optimization algorithms
***********************

Constrained optimization problems are characterized by imposed limitations on the weighted combinations of controls (linear input constraints) or on the outputs of the simulator (nonlinear output constraints). More advanced algorithms are required to solve such problems. The Dakota backend supports two algorithms for solving constrained optimization problems, accessible through the optimizers **opt++** and **conmin** respectively. Examples on the parameter specification can be found in The :ref:`cha_optimization_backends`.

A general formulation of a constrained optimization problem with inequality constraints is:

.. math::

	min_{x} f(x) \; , \; subject \; to \; c(x) <= 0


where :math:`x` is a vector containing e.g. well rates, pressures or any other controls that are defined in the EVEREST configuration file. :math:`f(x) = -J(x)` is the expected objective function value, where in reservoir applications :math:`J(x)` is typically the average NPV evaluated over all model realizations. :math:`c(x) = [c_1 (x),…,c_n (x)]`  is a constraint vector function, that is, it may consist of one or more constraint functions :math:`c_i` that each produce a single scalar value. Examples are a maximum field injection rate (which is a linear input constraint if well injection rates are controls, since it is computed as the sum of the controls), and a maximum limit on the total water production from the field (which must be computed from simulator output). If the constraint above can be formulated as :math:`c(x) = 0` , we are dealing with an equality constraint. An example is the constraint that all produced gas has to be re-injected. Bound constraints are usually formulated separately as :math:`\; a ≤ x ≤ b`.

.. note::

	In general :math:`f(x)` could also be a vector function; in that case we are dealing with multi-objective optimization. For the discussion here we assume that :math:`f(x)` is a single scalar function value.

If the constraint function is linear, i.e.  :math:`c(x) = C*x`, the optimizers do not require the calculation of gradients. Here we will focus on the case of nonlinear output constraints. EVEREST will automatically compute the gradients of these constraints with respect to the controls, in the same way that it computes the gradient of the objective function (e.g. NPV) with respect to the controls, and provide these gradients to the optimizer. Below we will briefly sketch the basic workings of the two algorithms.

The performance of **opt++** and **conmin** algorithms for bound-constrained optimization will not been discussed here.

Opt++ algorithm
#####################

Specification in the configuration file of the `optpp_q_newton` optimizer option will activate the constraint handling **Quasi-Newton Interior Point** algorithm of **opt++**. The term *interior point* refers to the search for a control solution that lies inside (i.e. in the interior of) the so-called *feasible region*, which is the set of control solutions for which all constraints are satisfied.

Internally **opt++** will handle the constraints by formulating a so-called merit function by adding a penalty term that measures the magnitude of the violation of the constraints to the objective function. If the constraints are not satisfied the penalty term will be non-zero. Control solutions that are infeasible will therefore lead to higher merit functions and **opt++** will try to avoid such solutions by minimizing the merit function using the so-called Quasi-Newton method.

The update equation at iteration k for this method can be written as :math:`x_{k+1} = x_k - H_k^{-1} * ∇M(x_k)`, where :math:`∇M(x_k)` is the gradient of the merit function, which is constructed internally from the objective function gradient :math:`∇f(x_k)` and the constraint function gradient :math:`∇c(x_k)` as calculated by EVEREST. :math:`H_k` is an approximation of the Hessian matrix of second derivatives of the merit function, constructed internally by use of the so-called *BFGS scheme*. The approximation relies on the gradients, which are themselves also approximate. It is not well known how these approximations affect each other, or if the update direction :math:`∇M(x_k)` becomes more or less reliable with increasing iteration number k.

.. note::

	Note that for :math:`H_k = I`, the identify matrix, the update equation becomes what is known as the **steepest descent update**.

.. important::

	Looking only at the objective function, it may seem that the optimizer sometimes accepts solutions that are worse than previous ones, as evidenced by a lower objective function value. However, it may be that the updated solution has resulted in a reduction of the constraint violation, and also in a reduction of the merit function value, and should therefore be considered an improved solution.

The **opt++** algorithm provides some options to the user on how to construct the merit function, how close to the boundary of the feasible region the solution is allowed to move, and some addition options that may influence the behaviour of the constrained optimizer.

.. note::

	EVEREST will use the default values for all of the above mentioned options.

Conmin algorithm
#####################

The `conmin_mfd` option of the conmin optimizer will activate its *Method of Feasible Directions* algorithm. The essence of the method is that in each iteration k it tries to find a search vector :math:`s_k` for which the control update :math:`x_{k+1}=x_k-α_k*s_k`, for sufficiently small step size :math:`α_k`, leads both to a feasible control solution :math:`x_{k+1}` and an improved objective function value. For such a search vector it is required that both:

.. math::

	∇f(x_k )s_k^T<0 \;\; and \;\; ∇c(x_k )s_k^T≤0

are statisfied. Where :math:`∇f(x_k)` and :math:`∇c(x_k)` are the objective function and constraint function gradients respectively, which are calculated by EVEREST.  *The method of Zoutendijk* is used internally to find such a search vector. The method can be seen as a generalization of steepest ascent method for unconstrained optimization. Different from **opt++**, the method does not use approximations of second derivates.

.. note::

	`conmin_mfd` does provide some user options but these are currently not provided through EVEREST.

Comments on the use of opt++ and conmin
##########################################

It is difficult so provide generic guidance on when to choose which optimizer. However, the DAKOTA manual and experience has suggested the following:

* The DAKOTA users’ manual (p. 115) notes that it has been observed that `conmin_mfd` does a poor job handling (nonlinear) equality constraints, e.g. :math:`c(x)=0`. It does seem to work better for linear input equality constraints. It could be considered to replace a single equality constraint by two inequality constraints :math:`c(x)≤0` and :math:`c(x)≥0` , which can only be both satisfied if :math:`c(x)=0`. We do not have much experience yet with this approach.
* Some issues have been encountered when using **opt++** in applications with nonlinear constraints, such as exiting without any messages, and producing infeasible solutions without warning.
* Experience has shown that sometimes restarting an optimization may be beneficial. This is facilitated by EVEREST.
