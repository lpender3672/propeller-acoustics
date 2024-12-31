from stl import mesh
import numpy as np

from scipy.integrate import cumtrapz

def rotate(X,Y,theta):
    # function to rotate coordinates in the X-Y plane
    return (X[:]*np.cos(theta) - Y[:]*np.sin(theta)), (X[:]*np.sin(theta) + Y[:]*np.cos(theta)) 


def generate_blade_mesh(av):
    # adapted from code by Lily Board phd

    chord, twist, radius = av.prop['c'], av.prop['HX'], av.prop['r0']
    xnf, ynf = av.airfoil_data[:,0], av.airfoil_data[:,1]
    xf = np.interp(np.linspace(0, 1, av.prop['nx']), np.linspace(0, 1, xnf.shape[0]), xnf)
    yf = np.interp(np.linspace(0, 1, av.prop['nx']), np.linspace(0, 1, ynf.shape[0]), ynf)

    xf -= 0.25 # quarter chord

    xsweep = cumtrapz( av.prop['sweep'], radius, initial=0.0)
    thickness = av.prop['HX']

    nf = xf.shape[0]
    Nsect = radius.shape[0]
    X = np.zeros((Nsect+2,nf))
    Y = np.zeros((Nsect+2,nf))
    Z = np.zeros((Nsect+2,nf))

    # rotate the airfoil and scale
    X[0,:], Y[0,:] = rotate(xf*chord[0], yf*chord[0], twist[0])
    Z[0,:] = 0.0
    for i in range(1,Nsect+1):
        X[i,:], Y[i,:] = rotate(xf*chord[i-1], yf*chord[i-1], twist[i-1])
        X[i,:] += xsweep[i-1]
        Z[i,:] = radius[i-1]
    X[Nsect+1,:], Y[Nsect+1,:] = rotate(xf*chord[Nsect-1], yf*chord[Nsect-1], twist[Nsect-1])
    X[Nsect+1,:] += xsweep[Nsect-1]
    Z[Nsect+1,:] = radius[-1]
    
    coords = np.zeros((nf*(Nsect+2),3))
    # loop over blade elements
    for i in range(Nsect+2):
        # loop over nf in airfoil
        for j in range(nf):
            k = i*nf + j
            coords[k,0] = X[i,j]
            coords[k,1] = Y[i,j]
            coords[k,2] = Z[i,j]
    
    triangles = np.zeros((2*nf*(Nsect+1)+2*(nf-2),3),dtype = np.int64)
    # loop over blade elements
    # not including end faces
    k = 0
    for i in range(1,Nsect+2):
        # loop over nf in airfoil
        for j in range(nf):
            #    __@
            #   | 
            triangles[k,0] = i*nf + j
            if j == 0:
                triangles[k,1] = i*nf + nf-1
                triangles[k,2] = (i-1)*nf + nf-1
            else:
                triangles[k,1] = i*nf + j - 1
                triangles[k,2] = (i-1)*nf + j - 1

            k = k+1
            #     @
            #   __|
            triangles[k,0] = i*nf + j
            if j == 0:
                triangles[k,1] = (i-1)*nf + nf-1
            else:
                triangles[k,1] = (i-1)*nf + j - 1
            triangles[k,2] = (i-1)*nf + j 
            k = k+1
    # end faces
    for j in range(nf-2):
        # bottom face
        if j%2 == 0:
            triangles[k,0] = int(j/2)
            triangles[k,1] = nf-1 - int(j/2) 
            triangles[k,2] = int(j/2) + 1
        else:
            triangles[k,0] = nf-1 - int((j-1)/2)
            triangles[k,1] = nf-1 - int((j-1)/2) - 1 
            triangles[k,2] = int((j-1)/2) + 1
        k = k+1
        # top face
        if j%2 == 0:
            triangles[k,0] = nf*(Nsect+1) + int(j/2)
            triangles[k,1] = nf*(Nsect+1) + int(j/2) + 1
            triangles[k,2] = nf*(Nsect+1) + nf-1 - int(j/2)
        else:
            triangles[k,0] = nf*(Nsect+1) + nf-1 - int((j-1)/2)
            triangles[k,1] = nf*(Nsect+1) + int((j-1)/2) + 1
            triangles[k,2] = nf*(Nsect+1) + nf-1 - int((j-1)/2) - 1
        k = k+1

    blademesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for j in range(3):
            blademesh.vectors[i][j] = coords[f[j],:]

    # Write the mesh to file "cube.stl"
    return blademesh
