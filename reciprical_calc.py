import numpy as np

def reciprocal_lattice(a,b,c,alpha,beta,gamma):

    alpha=np.deg2rad(alpha)
    beta=np.deg2rad(beta)
    gamma=np.deg2rad(gamma)

    avec=np.array([a,0,0])

    bvec=np.array([
        b*np.cos(gamma),
        b*np.sin(gamma),
        0
    ])

    cx=c*np.cos(beta)

    cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)

    cz=np.sqrt(c**2-cx**2-cy**2)

    cvec=np.array([cx,cy,cz])

    V=np.dot(avec,np.cross(bvec,cvec))

    astar=2*np.pi*np.cross(bvec,cvec)/V
    bstar=2*np.pi*np.cross(cvec,avec)/V
    cstar=2*np.pi*np.cross(avec,bvec)/V

    return astar,bstar,cstar