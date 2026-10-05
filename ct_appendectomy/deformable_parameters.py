"""Validate documented PhysX 110.3 deformable parameter bounds before simulation."""
from numbers import Integral


def validate_solver_iterations(value):
    # PhysxSchemaAddition/resources/schema.usda, PhysxBaseDeformableBodyAPI.
    if isinstance(value,bool) or not isinstance(value,Integral) or not 1<=value<=255:
        raise ValueError('PhysX deformable solverPositionIterationCount must be an integer in [1, 255]')
    return int(value)


def validate_deformable_parameters(prim):
    attr=prim.GetAttribute('physxDeformableBody:solverPositionIterationCount')
    if attr and attr.HasAuthoredValueOpinion():
        validate_solver_iterations(attr.Get())
