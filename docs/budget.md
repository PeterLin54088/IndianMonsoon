## General Budget Equation

We begin with the general budget equation:

$$
\frac{\partial q}{\partial t} + \nabla \cdot (\vec{u} q) = F
$$

## Practical Considerations

To apply this framework in practice, we introduce several limitations and constraints when comparing the ideal model with reanalysis data:

1. **Non-closure of Energy Budget in ERA5**

   Due to data assimilation, the ERA5 energy budget is not fully closed. Therefore, our focus is on identifying **relative contributions** rather than achieving exact closure.

2. **Neglect of the Source Term $F$**

   The term $F$ is not evaluated. It is most relevant within the boundary layer, whereas our primary interest lies in large-scale tendencies in the mid-troposphere.

3. **Mass Continuity Constraint**

   In height coordinates:

   $$
   \frac{D}{Dt} \int_V \rho \cdot (dx dy dz) = 0 \underset{\frac{D dV}{Dt} = dV \nabla \cdot \vec{u}}{\Longrightarrow} \int_V \left( \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{u}) \right) dV = 0
   $$

   In pressure coordinates, under the hydrostatic approximation (whose validity shall be checked in practice):

   $$
   \frac{\partial p}{\partial z} = \rho g \Longrightarrow \frac{\delta p}{g} = \rho \delta z
   $$

   Mass continuity then becomes:

   $$
   \frac{D}{Dt} \int_V \rho \cdot (dx dy dz) = 0 \Longrightarrow \frac{D}{Dt} \int_V \frac{1}{g} \cdot (dx dy dp) = 0
   $$

   which implies:

   $$
   \nabla_p \cdot \vec{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial \omega}{\partial p} = 0
   $$

With these considerations, the governing equation reduces to:

$$
\frac{\partial q}{\partial t} = - \vec{u} \cdot \nabla q + \text{Error}
$$

## Reynolds Decomposition

To separate mean and eddy contributions, we apply a Reynolds decomposition to any variable $\lambda$:

$$
\lambda = \overline{\lambda} + \lambda^{\prime}
$$

In this project, the operator $\overline{(\cdot)}$ denotes the **regional zonal mean**, defined as:

$$
\overline{(\cdot)} = \frac{1}{L_x} \int_0^{L_x} (\cdot) \, dx
$$

Applying Reynolds decomposition to the governing equation and expanding $\nabla \cdot$ gives:

$$
\frac{\partial \left( \overline{q} + q^{\prime} \right)}{\partial t} = - \left( \overline{u} + u^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial x} - \left( \overline{v} + v^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial y} - \left( \overline{\omega} + \omega^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial p} + \text{Error}
$$

Applying the $\overline{(\cdot)}$ operator to the entire equation yields:

$$
\overline{\frac{\partial \left( \overline{q} + q^{\prime} \right)}{\partial t}} = - \overline{ \left( \overline{u} + u^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial x}} - \overline{ \left( \overline{v} + v^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial y}} - \overline{ \left( \overline{\omega} + \omega^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial p}} + \text{Error}
$$

By properties of the $\overline{(\cdot)}$ operator:

- $\overline{\frac{\partial \left( \overline{q} + q^{\prime} \right)}{\partial t}} = \frac{\partial \overline{q}}{\partial t}$

- $\frac{\partial \overline{q}}{\partial x} = 0$

Thus, the equation simplifies to:

$$
\frac{\partial \overline{q}}{\partial t}
= - \overline{ \left( \overline{u} + u^{\prime} \right) \frac{\partial q^{\prime}}{\partial x}} - \overline{ \left( \overline{v} + v^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial y}} - \overline{ \left( \overline{\omega} + \omega^{\prime} \right) \frac{\partial (\overline{q} + q^{\prime})}{\partial p}} + \text{Error}
$$

By properties of the averaging operator:

- $\overline{ \overline{A_i} \partial_i \overline{B_i}} = \overline{A_i} \cdot \partial_i \overline{B_i}$
- $\overline{ \overline{A_i} \partial_i B_i^{\prime}} = \overline{A_i} \cdot \overline{ \partial_i B_i^{\prime}}$
- $\overline{ A_i^{\prime} \partial_i \overline{B_i}} = \overline{ A_i^{\prime}} \cdot \overline{ \partial_i \overline{B_i}} = 0$, since $\overline{ A_i^{\prime}} = 0$
- $\overline{ A_i^{\prime} \partial_i B_i^{\prime}}$ remains as is

The governing equation further simplifies to:

$$
\frac{\partial \overline{q}}{\partial t} = - \overline{\left(\overline{u} + u^{\prime}\right) \frac{\partial q^{\prime}}{\partial x}} - \overline{\overline{v} \, \frac{\partial \overline{q}}{\partial y}} - \overline{\left(\overline{v} + v^{\prime}\right) \frac{\partial q^{\prime}}{\partial y}} - \overline{\overline{\omega} \, \frac{\partial \overline{q}}{\partial p}} - \overline{\left(\overline{\omega} + \omega^{\prime}\right) \frac{\partial q^{\prime}}{\partial p}} + \text{Error}
$$
