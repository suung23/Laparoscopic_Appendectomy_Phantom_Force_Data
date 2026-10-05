"""Rest-state elastic Hessian: search metric only, not changed physics."""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized


def rest_stiffness(model):
    g = np.concatenate((-model.inverse.sum(axis=1)[:,None,:], model.inverse), axis=1)
    size = 3*len(model.rest)
    rows, columns, values = [], [], []
    for a in range(4):
        for b in range(4):
            dot = np.sum(g[:,a]*g[:,b], axis=1)
            for i in range(3):
                for j in range(3):
                    value = model.volume*(model.lam*g[:,a,i]*g[:,b,j] +
                        model.mu*((dot if i == j else 0) + g[:,a,j]*g[:,b,i]))
                    rows.append(3*model.tets[:,a]+i)
                    columns.append(3*model.tets[:,b]+j)
                    values.append(value)
    return coo_matrix((np.concatenate(values), (np.concatenate(rows),np.concatenate(columns))),
                      shape=(size,size)).tocsc()


def make_preconditioner(model, x, fixed, contact=None):
    matrix = rest_stiffness(model)
    if contact is not None:
        matrix += contact.hessian(x)
    free = np.flatnonzero(~fixed.ravel())
    reduced = matrix[free][:,free]
    # Regularize null rigid modes in the search metric, not the objective.
    regularization = max(float(np.mean(matrix.diagonal()))*1e-6, 1e-12)
    solve = factorized(reduced+diags(np.full(len(free),regularization)))
    def apply(gradient):
        direction = np.zeros(gradient.size)
        direction[free] = -solve(gradient.ravel()[free])
        return direction.reshape(gradient.shape)
    return apply
