
Ab-Initio Molecular Dynamics
============================

In this section, working equations for AIMD is presented.
For the Forces see Geometry Optimization section.

Velocity Verlet integrator
--------------------------

The Velocity Verlet is given is three steps. First the position is updated:

.. math::
   x\left(t+\Delta t\right)=x\left(t\right)+v\left(t\right)\Delta t+\frac{1}{2}a\left(t\right)\Delta t^{2}

Then the forces are calculated. At last the velocties are updated:

.. math::
   v\left(t+\Delta t\right)=v\left(t\right)+\frac{1}{2}\left[a\left(t\right)+a\left(t+\Delta t\right)\right]\Delta t

FUNCTION:

- runBOMD.VelocityVerlet(inputBOMD, dt, results, set)

Input:

- inputBOMD, inputfile object for BOMD
- dt, integration step size
- results, results object
- set, settings object

Output:

- inputBOMD, inputfile object for BOMD
- results, results object
   
References:

- None