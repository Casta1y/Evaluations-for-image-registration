import torch
import numpy as np


def JacobianDeterminant2D(disp):
    """输入一个位移场，返回一个对应的雅可比行列式。每个像素点的位移都对应一个雅可比行列式。
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 3D displacement field of size [nb_dims, *vol_shape]
    Returns:
        jacobian determinant (matrix)
    """

    # check inputs
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)

    # compute gradients
    [xFX, xFY] = np.gradient(grid[0] - disp[0])
    [yFX, yFY] = np.gradient(grid[1] - disp[1])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            jac_mij = [[xFX[i, j], xFY[i, j]], [yFX[i, j], yFY[i, j]]]
            jac_det[i, j] =  np.linalg.det(jac_mij)
    return jac_det



def JacobianDeterminant3D(disp):
    # check inputs
    volshape = disp.shape[1:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, 0)

    # compute gradients
    [xFX, xFY, xFZ] = np.gradient(grid[0] - disp[0])
    [yFX, yFY, yFZ] = np.gradient(grid[1] - disp[1])
    [zFX, zFY, zFZ] = np.gradient(grid[2] - disp[2])

    jac_det = np.zeros(grid[0].shape)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            for k in range(grid.shape[3]):
                jac_mij = [[xFX[i, j, k], xFY[i, j, k], xFZ[i, j, k]], [yFX[i, j, k], yFY[i, j, k], yFZ[i, j, k]], [zFX[i, j, k], zFY[i, j, k], zFZ[i, j, k]]]
                jac_det[i, j, k] =  np.linalg.det(jac_mij)
    return jac_det

def GetNegJDetNum(y_pred, device=torch.device('cpu'), inshape=(112, 112)):
    '''负雅可比行列式像素数量
    the number of pixels having negative JacobianDeterminant
    '''
    y_pred = y_pred.detach().squeeze().cpu().numpy()
    jdet = JacobianDeterminant2D(y_pred)
    neg_components = jdet[jdet < 0]
    return neg_components.shape[0]