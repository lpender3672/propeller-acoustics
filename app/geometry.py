from stl import mesh
import numpy as np

from PyQt6.QtWidgets import QApplication

from routines import (
    load_prop_from_file,
    AppVars
)

from scipy.spatial.transform import Rotation as R

def rotate2(X,Z,theta):
    # function to rotate coordinates in the X-Z plane
    return (X*np.cos(theta) - Z*np.sin(theta)), (X*np.sin(theta) + Z*np.cos(theta))

def rotate_airfoil(airfoil, theta):
    # rotate airfoil about the x-axis
    x, z = airfoil[:,0], airfoil[:,1]
    return np.column_stack(rotate2(x, z, theta))

def rotate3(X,Y,Z,theta, axis):
    axis = axis / np.linalg.norm(axis)
    r = R.from_rotvec(theta*axis)
    return r.apply(np.array([X,Y,Z])).T

def rotate3p(X,Y,Z, theta, axis, P):
    # rotate about a point and axis
    axis = axis / np.linalg.norm(axis)
    r = R.from_rotvec(theta*axis)
    points = np.vstack([X, Y, Z])
    translated = points - P[:, np.newaxis]
    rotated = r.apply(translated.T).T
    out = rotated + P[:, np.newaxis]
    return out

def normalize(vec):
    return vec / np.linalg.norm(vec)

def hermite_curve_and_tangents(P0, P1, T0, T1, num_points=50):
    # cubic Hermite curve

    t_values = np.linspace(0, 1, num_points)
    
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)
    T0 = np.array(T0, dtype=float)
    T1 = np.array(T1, dtype=float)
    
    points = []
    tangents = []
    
    for t in t_values:

        h00 =  2*t**3 - 3*t**2 + 1
        h10 =      t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =      t**3 -    t**2
        
        C_t = h00*P0 + h10*T0 + h01*P1 + h11*T1
        
        # derivative of h00 =  6t^2 - 6t
        # derivative of h10 =  3t^2 - 4t + 1
        # derivative of h01 = -6t^2 + 6t
        # derivative of h11 =  3t^2 - 2t
        h00d =  6*t**2 - 6*t
        h10d =  3*t**2 - 4*t + 1
        h01d = -6*t**2 + 6*t
        h11d =  3*t**2 - 2*t
        # tangent
        Ct_prime = h00d*P0 + h10d*T0 + h01d*P1 + h11d*T1
        
        points.append(C_t)
        tangents.append(normalize(Ct_prime) if np.linalg.norm(Ct_prime) > 1e-12 else Ct_prime)
    
    return np.array(points), np.array(tangents)

def compute_local_frames(points, tangents):

    up = np.array([0, 0, 1], dtype=float)
    
    N_array = []
    B_array = []
    
    for i in range(len(points)):
        T = tangents[i]
        
        # 1) B = T x up
        B = np.cross(T, up)
        normB = np.linalg.norm(B)
        if normB < 1e-12:
            # try x
            B = np.cross(T, [1,0,0])
            normB = np.linalg.norm(B)
            if normB < 1e-12:
                # try y
                B = np.cross(T, [0,1,0])
                normB = np.linalg.norm(B)
        
        B = B / normB
        
        N = np.cross(B, T)
        N = N / np.linalg.norm(N)
        
        N_array.append(N)
        B_array.append(B)
    
    return np.array(tangents), np.array(N_array), np.array(B_array)

def place_airfoil_3D(airfoil_2D, center_point, N, B):

    M = airfoil_2D.shape[0]
    points_3D = np.zeros((M, 3))
    for j in range(M):
        x2d, z2d   = airfoil_2D[j]
        points_3D[j] = center_point + x2d*N + z2d*B
    return points_3D

def segment_segment_intersection_3d(p1, p2, p3, p4, eps=1e-9):

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    
    d1 = p2 - p1
    d2 = p4 - p3
    cross_d1_d2 = np.cross(d1, d2)
    denom = np.dot(cross_d1_d2, cross_d1_d2)

    if denom < eps:
        return None

    r = p3 - p1
    cross_r_d2 = np.cross(r, d2)
    u = np.dot(cross_r_d2, cross_d1_d2) / denom
    
    cross_r_d1 = np.cross(r, d1)
    v = np.dot(cross_r_d1, cross_d1_d2) / denom
    
    if 0.0 - eps <= u <= 1.0 + eps and 0.0 - eps <= v <= 1.0 + eps:
        # Intersection point:
        intersection = p1 + u*d1
        return intersection
    else:
        return None

def bounding_box_3d(points):

    pts = np.array(points, dtype=float)
    min_x, min_y, min_z = np.min(pts, axis=0)
    max_x, max_y, max_z = np.max(pts, axis=0)
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def boxes_intersect_3d(boxA, boxB):

    return not (
        (boxA[1] < boxB[0]) or (boxA[0] > boxB[1]) or  # x-range
        (boxA[3] < boxB[2]) or (boxA[2] > boxB[3]) or  # y-range
        (boxA[5] < boxB[4]) or (boxA[4] > boxB[5])     # z-range
    )

def intersect_3D_profiles(profileA, profileB, eps=1e-12):

    boxA = bounding_box_3d(profileA)
    boxB = bounding_box_3d(profileB)
    if not boxes_intersect_3d(boxA, boxB):
        return []

    intersections = []

    A = np.array(profileA, dtype=float)
    B = np.array(profileB, dtype=float)

    nA = len(A)
    nB = len(B)

    for i in range(nA):
        p1 = A[i]
        p2 = A[(i+1) % nA]

        for j in range(nB):
            p3 = B[j]
            p4 = B[(j+1) % nB]

            inter_pt = segment_segment_intersection_3d(p1, p2, p3, p4, eps=eps)
            if inter_pt is not None:
                intersections.append(inter_pt)

    return np.array(intersections)


def airfoil_to_ellipse(x, y, x_factor, y_factor):

    x_centered = x - 0.5

    x_ellipse = x_centered * (1 - np.abs(x_factor))
    y_ellipse = np.sqrt(1 - (x_centered**2)) * np.abs(y_factor)

    y_transformed = (1 - y_factor) * y + y_ellipse * np.sign(y)
    
    return x_ellipse, y_transformed

def generate_blade_mesh(av, are_sections_tangent=True, apply_min_thickness=False):
    """creates a mesh of a single blade of a propeller

    Args:
        av (_type_): _description_
        are_sections_tangent (bool, optional): _description_. Defaults to True.
        min_section_thickness (float, optional): _description_. Defaults to 0.0004.

    Returns:
        _type_: _description_
    """
    # adapted from code by Lily Board phd

    # for 3D printing a minimum airfoil thickness is applied
    #min_section_thickness = 0.0004
    min_section_thickness = 0

    chord, twist, radius = av.prop['c'], av.prop['twist'], av.prop['r0']
    xnf, znf = av.airfoil_data[:,0], av.airfoil_data[:,1]
    xf = np.interp(np.linspace(0, 1, 2*av.prop['nx']), np.linspace(0, 1, xnf.shape[0]), xnf)
    zf = np.interp(np.linspace(0, 1, 2*av.prop['nx']), np.linspace(0, 1, znf.shape[0]), znf)
    
    if apply_min_thickness:
        tf = zf[:av.prop['nx']] - zf[av.prop['nx']:]
        zmean = (zf[:av.prop['nx']] + zf[av.prop['nx']:]) / 2
        mask = tf*av.prop['c75'] < min_section_thickness
        correction = min_section_thickness / av.prop['c75']
        zf[:av.prop['nx']][mask] = zmean[mask] + correction / 2
        zf[av.prop['nx']:][mask] = zmean[mask] - correction / 2

    xf -= 0.5 # quarter chord

    sweep_angle = av.prop['sweep']
    
    assert len(chord) == len(twist) == len(radius) == len(sweep_angle)

    nf = xf.shape[0]
    Nsect = radius.shape[0]

    if av.prop['bdir'] == 'CW':
        bdir = 1
    else:
        bdir = -1

    xf *= bdir
    sweep_angle *= bdir
    twist *= bdir

    upscaled_airfoil = np.column_stack([xf + 0.5, zf])

    if av.prop['twinB'] == 'Single':
        Ntip = 0
        coords = np.zeros((nf*(Nsect+2),3))
        triangles = np.zeros((2*nf*(Nsect+1)+2*(nf-2),3),dtype = np.int64)

        X = np.zeros((Nsect+2,nf))
        Z = np.zeros((Nsect+2,nf))
        Y= np.zeros((Nsect+2,nf))

    else:
        Ntip = 20 # number of tip sections
        #Rtip = av.prop['rt'] / np.cos(sweep_angle[-1])
        coords = np.zeros((nf*(2*Nsect+2+Ntip),3))
        triangles = np.zeros((2*nf*(2*Nsect+1+Ntip)+2*(nf-2),3),dtype = np.int64)

        X = np.zeros((2*Nsect+Ntip+2,nf))
        Z = np.zeros((2*Nsect+Ntip+2,nf))
        Y= np.zeros((2*Nsect+Ntip+2,nf))

    if are_sections_tangent:
        # sections tangent to the radius
        # at 0 only rotate about radial
        sx, Z[0,:] = rotate2(xf*chord[0], zf*chord[0], -twist[0])
        sy = av.prop['dthread'] / 2 * np.ones(nf)
        X[0,:], Y[0,:] = rotate2(sx, sy, -sweep_angle[0])
        start = 0
        end = Nsect+1
        for i in range(1,end):
            sx, Z[i,:] = rotate2(xf*chord[i-1], zf*chord[i-1], -twist[i-1])
            sy = radius[i-1] * np.ones(nf)
            X[i,:], Y[i,:] = rotate2(sx, sy, -sweep_angle[i-1])

        if Ntip > 0:
            # second blade loops back through span
            start = Nsect+Ntip+1
            end = 2*Nsect+Ntip+1
            for i in range(Nsect+Ntip+1, end):
                sx, Z[i,:] = rotate2(xf*chord[end-i-1], zf*chord[end - i-1], -twist[end - i-1])
                sy = radius[end - i-1] * np.ones(nf)
                X[i,:], Y[i,:] = rotate2(sx, sy, +sweep_angle[end - i-1])

            # add another hub
            sx, Z[end,:] = rotate2(xf*chord[0], zf*chord[0], -twist[0])
            sy = av.prop['dthread'] / 2 * np.ones(nf)
            X[end,:], Y[end,:] = rotate2(sx, sy, sweep_angle[0])
            
            P0 = np.array([*rotate2(0.0, radius[Nsect-1], -sweep_angle[Nsect-1]), 0])
            P1 = np.array([*rotate2(0.0, radius[Nsect-1], sweep_angle[Nsect-1]), 0])

            T0 = -(np.array([X[start+1,0], Y[start+1,0], Z[start+1,0]]) - np.array([X[start,0], Y[start,0], Z[start,0]]))
            T1 = (np.array([X[Nsect+1,Nsect//2], Y[Nsect+1,Nsect//2], Z[Nsect+1,Nsect//2]]) - np.array([X[Nsect,Nsect//2], Y[Nsect,Nsect//2], Z[Nsect,Nsect//2]]))

        else:

            sx, Z[end,:] = rotate2(xf*chord[Nsect-1], zf*chord[Nsect-1], -twist[Nsect-1])
            sy = radius[Nsect-1] * np.ones(nf)
            X[end,:], Y[end,:] = rotate2(sx, sy, -sweep_angle[Nsect-1])

    else:
        # sections wrap around the radius

        # at 0 only rotate about radial
        sx, Z[0,:] = rotate2(xf*chord[0], zf*chord[0], -twist[0])
        thetas = sweep_angle[0] + sx / radius[0] 
        thetas = np.clip(thetas, -np.pi/2, np.pi/2)
        X[0,:] = radius[0] * np.sin(thetas)
        Y[0,:] = 0.0
        for i in range(1,Nsect+1):
            # first rotate about radial direction
            sx, sz = rotate2(xf*chord[i-1], zf*chord[i-1], -twist[i-1])
            # warp chordwise foil to at current radius
            thetas = sweep_angle[i-1] + sx / radius[i-1] # appy sweep
            thetas = np.clip(thetas, -np.pi/2, np.pi/2)
            Z[i,:] = sz
            X[i,:] = radius[i-1] * np.sin(thetas)
            Y[i,:] = radius[i-1] * np.cos(thetas)

        sx, sz = rotate2(xf*chord[Nsect-1], zf*chord[Nsect-1], -twist[Nsect-1])
        thetas = sweep_angle[Nsect-1] + sx / radius[Nsect-1]
        thetas = np.clip(thetas, -np.pi/2, np.pi/2)
        Z[Nsect+1,:] = sz
        X[Nsect+1,:] = radius[Nsect-1] * np.sin(thetas)
        Y[Nsect+1,:] = Y[Nsect,:]

    # create tip verticies
    #thetas = np.linspace(0, np.pi + 2 * sweep_angle[-1], Ntip, endpoint=True)
    #unitz = np.array([0,0,1])
    #rotcenter = np.array([0, Rtip, 0])
    #midfactorhalf = np.linspace(0, 1, Ntip // 2)
    #midfactor = np.hstack([midfactorhalf, -midfactorhalf[::-1]])
    #tipfactorhalf = np.linspace(1, 0, Ntip // 2)
    #tipfactor = np.hstack([tipfactorhalf, -tipfactorhalf[::-1]])
    #tiptwist = -twist[-1] * tipfactor
    #ellipsivity = 0.5 * midfactor

    #P0 = np.array([X[Nsect,:], Y[Nsect,:], Z[Nsect,:]])
    #P1 = np.array([X[Nsect+Ntip+1,:], Y[Nsect+Ntip+1,:], Z[Nsect+Ntip+1,:]])
    # T0 is normal to section and so is just cos and sin of sweep
    #T0 = 0.1 * np.array([np.sin(sweep_angle[-1]), np.cos(sweep_angle[-1]), 0])
    #T1 = -0.1 * np.array([np.sin(-sweep_angle[-1]), np.cos(-sweep_angle[-1]), 0])
    
    #for i, theta in zip(range(Nsect+1, Nsect+Ntip+1), thetas):
    #    j = i-(Nsect+1)
    #    if j > Ntip//2:
    #        xfs, zfs = airfoil_to_ellipse(-xf, zf, ellipsivity[j], 0)
    #    else:
    #        xfs, zfs = airfoil_to_ellipse(xf, zf, ellipsivity[j], 0)
    #        
    #    sx, sz = rotate2(xfs*chord[-1], zfs*chord[-1], tiptwist[j])
    #    sy = radius[-1] * np.ones(nf)
    #    sx, sy = rotate2(sx, sy, -sweep_angle[-1])
    #    X[i,:], Y[i,:], Z[i,:] = rotate3p(sx, sy, sz, theta, unitz, rotcenter)
    

    if Ntip > 0:
        #print(P0, P1, T0, T1)

        corrected_sections = generate_tip_verticies(
            P0, P1, T0, T1, 
            upscaled_airfoil, 
            chord[-1], twist[-1], 
            num_points=Ntip)
        for i in range(Ntip):
            for j in range(nf):
                X[Nsect+1+i, j] = corrected_sections[i][j,0]
                Y[Nsect+1+i, j] = corrected_sections[i][j,1]
                Z[Nsect+1+i, j] = corrected_sections[i][j,2] + 0.0001

    for i in range(end+1):
        # loop over nf in airfoil
        for j in range(nf):
            k = i*nf + j
            coords[k,0] = X[i,j]
            coords[k,1] = Y[i,j]
            coords[k,2] = Z[i,j]
    
    # loop over blade elements
    # not including end faces
    k = 0
    for i in range(1,end+1):
        # loop over nf in airfoil
        for j in range(nf):
            if Ntip > 0 and i == (Nsect + Ntip//2 + 2):
                l = nf//2 - j
                if j > nf//2:
                    l += nf
                    
                    #    __@
                    #   | 
                triangles[k,0] = i*nf + j
                
                if j == 0:
                    triangles[k,1] = i*nf + nf-1
                else:
                    triangles[k,1] = i*nf + j - 1
                if l == 0:
                    triangles[k,2] = (i-1)*nf + nf-1
                else:
                    triangles[k,2] = (i-1)*nf + l - 1

                k = k+1
                #     @
                #   __|
                
                if j == 0:
                    triangles[k,0] = i*nf + j - 1 + nf
                    triangles[k,1] = (i-1)*nf + nf//2-1
                else:
                    triangles[k,0] = i*nf + j - 1
                    triangles[k,1] = (i-1)*nf + l - 1
                triangles[k,2] = (i-1)*nf + l

                k = k+1
                continue

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
            triangles[k,0] = nf*end + int(j/2)
            triangles[k,1] = nf*end + int(j/2) + 1
            triangles[k,2] = nf*end + nf-1 - int(j/2)
        else:
            triangles[k,0] = nf*end + nf-1 - int((j-1)/2)
            triangles[k,1] = nf*end + int((j-1)/2) + 1
            triangles[k,2] = nf*end + nf-1 - int((j-1)/2) - 1
        k = k+1

    blademesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for j in range(3):
            blademesh.vectors[i][j] = coords[f[j],:]

    print("MAX X IS ", 2*np.max(X))
    print("MAX Y IS ", 2*np.max(Y))
    print("MAX Z IS ", 2*np.max(Z))

    # Write the mesh to file "cube.stl"
    return blademesh

def generate_hub(outer_radius, inner_radius, height, radial_segments, height_segments, outer_segments_per_inner=1):
    """
    Generate a cylinder mesh with top and bottom caps.

    Parameters:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        radial_segments (int): Number of segments around the circumference.
        height_segments (int): Number of segments along the height.

    Returns:
        cylinder_mesh (stl.mesh.Mesh): Generated cylinder mesh with caps.
    """
    vertices = []
    faces = []

    outer_segments = outer_segments_per_inner * radial_segments

    # Generate vertices for the sides
    for h in range(height_segments + 1):
        z = h * (height / height_segments) - height / 2
        for i in range(outer_segments):
            theta = 2 * np.pi * i / outer_segments
            x = outer_radius * np.cos(theta)
            y = outer_radius * np.sin(theta)
            vertices.append([x, y, z])
        
    # generate verticies for inner radius
    for h in range(height_segments + 1):
        z = h * (height / height_segments) - height / 2
        for i in range(radial_segments):
            theta = 2 * np.pi * i / (radial_segments)
            x = inner_radius * np.cos(theta)
            y = inner_radius * np.sin(theta)
            vertices.append([x, y, z])

    # Center vertices for caps
    #bottom_center = len(vertices)  # Index of the bottom center vertex
    #vertices.append([0.0, 0.0, -height / 2])
    #top_center = len(vertices)  # Index of the top center vertex
    #vertices.append([0.0, 0.0, height / 2])
    inner_offset = (height_segments + 1) * outer_segments
    for h in range(height_segments):
        for i in range(outer_segments):
            next_i = (i + 1) % outer_segments
            v1 = h * outer_segments + i
            v2 = h * outer_segments + next_i
            v3 = (h + 1) * outer_segments + i
            v4 = (h + 1) * outer_segments + next_i
            faces.append([v1, v2, v3])
            faces.append([v3, v2, v4])
        for i in range(radial_segments):
            next_i = (i + 1) % radial_segments
            v1 = inner_offset + h * radial_segments + i
            v2 = inner_offset + h * radial_segments + next_i
            v3 = inner_offset + (h + 1) * radial_segments + i
            v4 = inner_offset + (h + 1) * radial_segments + next_i
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
        
    # Generate faces for the bottom cap
    out_offset = 0
    inner_offset = (height_segments + 1) * outer_segments
    for i in range(radial_segments):
        for j in range(outer_segments_per_inner):
            k = (i * outer_segments_per_inner + j) % outer_segments
            next_k = (i * outer_segments_per_inner + j + 1) % outer_segments
            v1 = out_offset + k
            v2 = out_offset + next_k
            v3 = inner_offset + i
            faces.append([v1, v3, v2])
        # v3 v2 and v3+1
        next_i = (i + 1) % radial_segments
        v1 = inner_offset + next_i
        faces.append([v1, v2, v3])

    # Generate faces for the top cap
    out_offset = height_segments * outer_segments
    inner_offset = (height_segments + 1) * outer_segments + height_segments * radial_segments
    for i in range(radial_segments):
        for j in range(outer_segments_per_inner):
            k = (i * outer_segments_per_inner + j) % outer_segments
            next_k = (i * outer_segments_per_inner + j + 1) % outer_segments
            v1 = out_offset + k
            v2 = out_offset + next_k
            v3 = inner_offset + i
            faces.append([v1, v3, v2])
        # v3 v2 and v3+1
        next_i = (i + 1) % radial_segments
        v1 = inner_offset + next_i
        faces.append([v1, v2, v3])


    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Create the STL mesh
    cylinder_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cylinder_mesh.vectors[i,j] = vertices[face[j], :]

    return cylinder_mesh

def stitch_meshes(mesh1, mesh2):

    combined_vertices = np.vstack((mesh1.vectors.reshape(-1, 3), mesh2.vectors.reshape(-1, 3)))
    unique_vertices, inverse_indices = np.unique(combined_vertices, axis=0, return_inverse=True)
    
    num_faces_mesh1 = len(mesh1.vectors)
    faces_mesh1 = inverse_indices[:num_faces_mesh1 * 3].reshape(num_faces_mesh1, 3)
    
    num_faces_mesh2 = len(mesh2.vectors)
    faces_mesh2 = inverse_indices[num_faces_mesh1 * 3:].reshape(num_faces_mesh2, 3)
    
    combined_faces = np.vstack((faces_mesh1, faces_mesh2))
    
    combined_mesh = mesh.Mesh(np.zeros(combined_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(combined_faces):
        for j in range(3):
            combined_mesh.vectors[i][j] = unique_vertices[face[j],:]
    
    return combined_mesh


def generate_propeller_mesh(av, apply_min_thickness=True):
    main_blade = generate_blade_mesh(av, apply_min_thickness=apply_min_thickness)
    
    # List to hold all blade meshes
    blade_meshes = [main_blade.data]
    
    # Generate and rotate each additional blade
    for i in range(1, av.prop['B']):
        new_blade = mesh.Mesh(main_blade.data.copy())
        rotation_angle = 2 * np.pi * i / av.prop['B']
        new_blade.rotate([0.0, 0.0, 1.0], rotation_angle)
        blade_meshes.append(new_blade.data)

    height = (0.1 + np.abs(np.sin(av.prop['twist'][0]))) * av.prop['c75']
    
    hub = generate_hub(
        av.prop['rh'],
        av.prop['dthread']/2,
        height,
        radial_segments=12,
        height_segments=2,
        outer_segments_per_inner=2
    )
    blade_meshes.append(hub.data)
    
    combined = mesh.Mesh(np.concatenate(blade_meshes))
    return combined

def generate_and_save_propeller_mesh(av, filename):
    # scale up 1000x then save
    prop_mesh = generate_propeller_mesh(av)
    prop_mesh.vectors *= 1000 * 100 / 2.54 # I think Tony's software is in like 10 thousandths of an inch?
    prop_mesh.save(filename)

def generate_tip_verticies(P0, P1, T0, T1, airfoil_2D, chord_factor=1, twist_offset=0, num_points=10, correct_intersections=True):

    #P0 = (0, 0, 0)
    #P1 = (0, 0, 2)
    #T0 = (0, 1, 0.3)
    #T1 = (0, -1, 0.3)

    points, tangs = hermite_curve_and_tangents(P0, P1, T0, T1, num_points)

    Ts, Ns, Bs = compute_local_frames(points, tangs)

    transformed_airfoil = np.zeros(airfoil_2D.shape)

    twist_values = np.pi/2 + np.linspace(twist_offset, -twist_offset, num_points)

    swept_sections = []
    for i in range(len(points)):
        center_pt = points[i]
        N_i = Ns[i]
        B_i = Bs[i]

        fx = i / len(points)
        fx = 3 * fx * (fx - 1)
        transformed_airfoil[:,0],transformed_airfoil[:,1] = airfoil_to_ellipse(
            airfoil_2D[:,0], airfoil_2D[:,1], fx, 0
        )
        if i > len(points) // 2:
            transformed_airfoil[:,0] = -transformed_airfoil[:,0]
        sized_airfoil = chord_factor * rotate_airfoil(transformed_airfoil, twist_values[i])
        section_3D = place_airfoil_3D(sized_airfoil, center_pt, N_i, B_i)
        swept_sections.append(section_3D)

    if not correct_intersections:
        return swept_sections

    # handle intersections
    corrected_sections = swept_sections.copy()

    for i in range(len(swept_sections)):
        center_pt = points[i]
        N_i = Ns[i]
        B_i = Bs[i]
        fx = i / len(points)
        fx = 3 * fx * (fx - 1)
        transformed_airfoil[:,0],transformed_airfoil[:,1] = airfoil_to_ellipse(
            airfoil_2D[:,0], airfoil_2D[:,1], fx, 0
        )
        T_0 = np.cross(Ns[0], Bs[0])
        section_tomap = swept_sections[0] + 0.01 * i * T_0
        bottom_intersections = intersect_3D_profiles(section_tomap, swept_sections[i])
        if i > 0 and len(bottom_intersections) > 0:

            inter = np.array(bottom_intersections[-1])
            v = inter - center_pt
            tfx_2D = np.dot(v, N_i)
            #tfz_2D = np.dot(v, B_i)

            mask = transformed_airfoil[:, 0] > tfx_2D
            indices = np.where(mask)[0]

            T_i = np.cross(N_i, B_i)
            
            A = np.array([Ns[0], Bs[0], -T_i]).T

            for idx in indices:
                # build matrix to solve projection
                b = swept_sections[i][idx] - points[0]
                coeffs = np.linalg.solve(A, b)
                k = coeffs[2]
                corrected_sections[i][idx] +=  k * T_i + 0.01 * i * T_0

        T_0 = np.cross(Ns[-1], Bs[-1])
        section_tomap = swept_sections[-1] - 0.01 * (len(swept_sections) - i - 1) * T_0
        top_intersections = intersect_3D_profiles(swept_sections[i], section_tomap)
        if i < len(points) - 1 and len(top_intersections) > 0:
            inter = np.array(top_intersections[-1])

            v = inter - center_pt
            tfx_2D = np.dot(v, N_i)

            mask = transformed_airfoil[:, 0] < tfx_2D
            indices = np.where(mask)[0]

            T_i = np.cross(N_i, B_i)
            T_0 = np.cross(Ns[-1], Bs[-1])
            A = np.array([Ns[-1], Bs[-1], -T_i]).T

            for idx in indices:
                # build matrix to solve projection
                b = swept_sections[i][idx] - points[-1]
                coeffs = np.linalg.solve(A, b)
                corrected_sections[i][idx] += coeffs[2] * T_i - 0.01 * T_0

    return corrected_sections

def main2():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    P0 = [0.00634396, 0.04945627, 0.0        ]   
    P1 = [-0.00634396,  0.04945627,  0.0        ]
    T0 = [0,         0.1, 0]
    T1 = [ 0,         -0.1, 0]
    Np = 12

    points, tangs = hermite_curve_and_tangents(P0, P1, T0, T1, Np)

    print(points)
    ax.plot(points[:,0], points[:,1], points[:,2], '-o')
    airfoil_data = np.loadtxt('app/foils/naca0012.surf')

    corrected_sections = generate_tip_verticies(P0, P1, T0, T1, airfoil_data, 0.1, np.pi/2, Np)
    for section in corrected_sections:
        ax.plot(section[:,0], section[:,1], section[:,2], '-o')        

    ax.set_aspect('equal')
    plt.show()

def main():
    import vis
    import sys
    prop = load_prop_from_file('app/props/constant_chord_swept.prop')
    airfoil_data = np.loadtxt(prop['foil_path'])

    av = AppVars()
    av.prop = prop
    av.airfoil_data = airfoil_data

    app = QApplication(sys.argv)
    viewer = vis.STLViewerWidget()

    viewer.set_mesh(
        generate_propeller_mesh(av)
    )
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    #
    #main2()
    main()
