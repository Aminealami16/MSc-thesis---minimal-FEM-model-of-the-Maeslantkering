import numpy as np


def M_local(ex,ey,ez,ep_M, type):
       
## Function that contains the mass matrix for a Timoshenko beam element in local coordinate system
## Input: 
              ## ex = [x1, x2] [m]
              ## ey = [y1, y2] [m]
              ## ez = [z1, z2] [m]
              ## ep_M = [rho, A, Iy, Iz, Ip]
                     # rho: density [kg/m^3]
                     # A: cross section area [m^2]
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # Ip: polar moment of inertia [m^4]
## Output: matrix of 12x12 containing the mass matrix of a three-dimensional 2-node Timoshenko beam element

       L   = np.sqrt((ex[0]-ex[1])**2+(ey[0]-ey[1])**2+(ez[0]-ez[1])**2)
       rho = ep_M[0]
       A   = ep_M[1]
       Iy  = ep_M[2]
       Iz  = ep_M[3]
       Ip  = ep_M[4]
       
       me  = rho * A * L
       
       Me11 = np.asmatrix([[1/3 ,0                       ,0                       ,0        ,0                       ,0],
                                     [0   ,13/35 + 6*Iz/(5*A*L**2) ,0                       ,0        ,0                       ,11*L/210 + Iz/(10*A*L)],
                                     [0   ,0                       ,13/35 + 6*Iy/(5*A*L**2) ,0        ,-11*L/210 - Iy/(10*A*L) ,0],
                                     [0   ,0                       ,0                       ,Ip/(3*A) ,0                       ,0],
                                     [0   ,0                       ,-11*L/210 - Iy/(10*A*L) ,0        ,L**2/105 + 2*Iy/(15*A)  ,0],
                                     [0   ,11*L/210 + Iz/(10*A*L)  ,0                       ,0        ,0                       ,L**2/105 + 2*Iz/(15*A)],
                                    ])
       
       Me21 = np.asmatrix([[1/6 ,0                       ,0                       ,0        ,0                       ,0],
                                     [0   ,9/70 - 6*Iz/(5*A*L**2)  ,0                       ,0        ,0                       ,13*L/420 - Iz/(10*A*L)],
                                     [0   ,0                       ,9/70 - 6*Iy/(5*A*L**2)  ,0        ,-13*L/420 + Iy/(10*A*L) ,0],
                                     [0   ,0                       ,0                       ,Ip/(6*A) ,0                       ,0],
                                     [0   ,0                       ,13*L/420 - Iy/(10*A*L)  ,0        ,-L**2/140 - Iy/(30*A)   ,0],
                                     [0   ,-13*L/420 + Iz/(10*A*L) ,0                       ,0        ,0                       ,-L**2/140 - Iz/(30*A)],
                                    ])

       Me = np.zeros((12,12))
       Me[0:6,0:6] = me * Me11
       Me[6:,0:6]  = me * Me21
       Me[0:6,6:]  = me * Me21.T
       Me[6:,6:]   = -me * Me11 + 2 * me * np.diag(np.diagonal(Me11))
   
       h1 = 7.5
       b2 = 12
       r1 = 237.5
       r2 = 237.5 + b2
       alpha = 50

       added_volume_total = (alpha / 360)  * np.pi * (r2**2 - r1**2) * h1
       water_density = 1025
       added_mass_total = added_volume_total * water_density
       me_added = added_mass_total / (120 - 71)
       Me_added = np.zeros((12,12))
       Me_added[0:6,0:6] = me_added * Me11
       Me_added[6:,0:6]  = me_added * Me21
       Me_added[0:6,6:]  = me_added * Me21.T
       Me_added[6:,6:]   = -me_added * Me11 + 2 * me_added * np.diag(np.diagonal(Me11))
      

       if type == 'normal':
              return Me
       elif type == 'added mass':
              return Me + Me_added
       


def K_local(ex,ey,ez,ep_K):
       
## Function that contains the stiffness matrix for a Timoshenko beam element in local coordinate system
## Input: 
              ## ex = [x1, x2] [m]
              ## ey = [y1, y2] [m]
              ## ez = [z1, z2] [m]
              ## ep_K = [E, G, A, Iy, Iz, It, k]
                     # E: modulus of elasticity [N/m^2]
                     # G: shear modulus [N/m^2]
                     # A: cross section area [m^2] 
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # It: torsional moment of inertia [m^4]
                     # k: shear correction factor [-]
## Output: matrix of 12x12 containing the stiffness matrix of a three-dimensional 2-node Timoshenko beam element

       L  = np.sqrt((ex[0]-ex[1])**2+(ey[0]-ey[1])**2+(ez[0]-ez[1])**2)
       E  = ep_K[0]
       G  = ep_K[1]
       A  = ep_K[2]
       Iy = ep_K[3]
       Iz = ep_K[4]
       It = ep_K[5]
       k  = ep_K[6]
       
       ke = 1 / L**3
       Py = 12 * E * Iz / (G * A * k * L**2)
       Pz = 12 * E * Iy / (G * A * k * L**2)
       
       Ke11 = np.asmatrix([[E*A*L**2 ,0               ,0                ,0         ,0                       ,0],
                                     [0        ,12*E*Iz/(1+Py)  ,0                ,0         ,0                       ,6*E*Iz*L/(1+Py)],
                                     [0        ,0               ,12*E*Iy/(1+Pz)   ,0         ,-6*E*Iy*L/(1+Pz)        ,0],
                                     [0        ,0               ,0                ,G*It*L**2 ,0                       ,0],
                                     [0        ,0               ,-6*E*Iy*L/(1+Pz) ,0         ,E*Iy*L**2*(4+Pz)/(1+Pz) ,0],
                                     [0        ,6*E*Iz*L/(1+Py) ,0                ,0         ,0                       ,E*Iz*L**2*(4+Py)/(1+Py)],
                                    ])
       
       Ke21 = np.asmatrix([[-E*A*L**2 ,0                ,0                ,0          ,0                       ,0],
                                     [0         ,-12*E*Iz/(1+Py)  ,0                ,0          ,0                       ,-6*E*Iz*L/(1+Py)],
                                     [0         ,0                ,-12*E*Iy/(1+Pz)  ,0          ,6*E*Iy*L/(1+Pz)         ,0],
                                     [0         ,0                ,0                ,-G*It*L**2 ,0                       ,0],
                                     [0         ,0                ,-6*E*Iy*L/(1+Pz) ,0          ,E*Iy*L**2*(2-Pz)/(1+Pz) ,0],
                                     [0         ,6*E*Iz*L/(1+Py)  ,0                ,0          ,0                       ,E*Iz*L**2*(2-Py)/(1+Py)],
                                    ])

       Ke = np.zeros((12,12))
       Ke[0:6,0:6] = ke * Ke11
       Ke[6:,0:6]  = ke * Ke21
       Ke[0:6,6:]  = ke * Ke21.T
       Ke[6:,6:]   = -ke * Ke11 + 2 * ke * np.diag(np.diagonal(Ke11))
   
       return Ke


def rotation(ex,ey,ez,eo):
       
## Function that contains the rotation matrix for a beam element
## Input: 
              ## ex = [x1, x2] [m]
              ## ey = [y1, y2] [m]
              ## ez = [z1, z2] [m]
              ## eo = [xz, yz, zz] --> global vector parallel with the positive local z axis of the beam
## Output: matrix of 12x12 containing the rotation matrix of a three-dimensional beam element
       
       b  = np.asmatrix([[ex[1]-ex[0]],[ey[1]-ey[0]],[ez[1]-ez[0]]])
       L  = np.sqrt(b.T*b).item()
       n1 = np.asarray(b.T/L).reshape(3,)
       
       eo = np.asmatrix(eo)
       lc = np.sqrt(eo*eo.T).item()
       n3 = np.asarray(eo/lc).reshape(3,)
       
       n2    = np.array([0.,0.,0.])
       n2[0] = n3[1]*n1[2]-n3[2]*n1[1]
       n2[1] = -n1[2]*n3[0]+n1[0]*n3[2]
       n2[2] = n3[0]*n1[1]-n1[0]*n3[1]
       
       G = np.asmatrix([
              [ n1[0], n1[1], n1[2], 0,     0,     0,     0,     0,     0,     0,     0,     0    ],
              [ n2[0], n2[1], n2[2], 0,     0,     0,     0,     0,     0,     0,     0,     0    ],
              [ n3[0], n3[1], n3[2], 0,     0,     0,     0,     0,     0,     0,     0,     0    ],
              [ 0,     0,     0,     n1[0], n1[1], n1[2], 0,     0,     0,     0,     0,     0    ],
              [ 0,     0,     0,     n2[0], n2[1], n2[2], 0,     0,     0,     0,     0,     0    ],
              [ 0,     0,     0,     n3[0], n3[1], n3[2], 0,     0,     0,     0,     0,     0    ],
              [ 0,     0,     0,     0,     0,     0,     n1[0], n1[1], n1[2], 0,     0,     0    ],
              [ 0,     0,     0,     0,     0,     0,     n2[0], n2[1], n2[2], 0,     0,     0    ],
              [ 0,     0,     0,     0,     0,     0,     n3[0], n3[1], n3[2], 0,     0,     0    ],
              [ 0,     0,     0,     0,     0,     0,     0,     0,     0,     n1[0], n1[1], n1[2]],
              [ 0,     0,     0,     0,     0,     0,     0,     0,     0,     n2[0], n2[1], n2[2]],
              [ 0,     0,     0,     0,     0,     0,     0,     0,     0,     n3[0], n3[1], n3[2]]])
       
       return G


def T_element(ex,ey,ez,eo,ep_K,ep_M):

## Function that contains the mass and stiffness matrices for a Timoshenko beam element in global coordinate system
## Input: 
              ## ex = [x1, x2] [m]
              ## ey = [y1, y2] [m]
              ## ez = [z1, z2] [m]
              ## eo = [xz, yz, zz] --> global vector parallel with the positive local z axis of the beam 
              ## ep_K = [E, G, A, Iy, Iz, It, k]            
                     # E: modulus of elasticity [N/m^2]
                     # G: shear modulus [N/m^2]
                     # A: cross section area [m^2] 
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # It: torsional moment of inertia [m^4]
                     # k: shear correction factor [-]
              ## ep_M = [rho, A, Iy, Iz, Ip]
                     # rho: density [kg/m^3]
                     # A: cross section area [m^2]
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # Ip: polar moment of inertia [m^4]
## Output: matrices of 12x12 containing the mass and stiffness matrices of a three-dimensional 2-node Timoshenko beam element
   
       K_l = K_local(ex, ey, ez, ep_K)
       M_l = M_local(ex, ey, ez, ep_M, type='normal')
       G   = rotation(ex,ey,ez,eo)
       
       K_e = G.T*K_l*G
       M_e = G.T*M_l*G
       
       return M_e, K_e


def T_element_added_mass_retaining_wall(ex,ey,ez,eo,ep_K,ep_M):

## Function that contains the mass and stiffness matrices for a Timoshenko beam element in global coordinate system
## Input: 
              ## ex = [x1, x2] [m]
              ## ey = [y1, y2] [m]
              ## ez = [z1, z2] [m]
              ## eo = [xz, yz, zz] --> global vector parallel with the positive local z axis of the beam 
              ## ep_K = [E, G, A, Iy, Iz, It, k]            
                     # E: modulus of elasticity [N/m^2]
                     # G: shear modulus [N/m^2]
                     # A: cross section area [m^2] 
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # It: torsional moment of inertia [m^4]
                     # k: shear correction factor [-]
              ## ep_M = [rho, A, Iy, Iz, Ip]
                     # rho: density [kg/m^3]
                     # A: cross section area [m^2]
                     # Iy: moment of inertia with respect to local y axis [m^4]
                     # Iz: moment of inertia with respect to local z axis [m^4]
                     # Ip: polar moment of inertia [m^4]
## Output: matrices of 12x12 containing the mass and stiffness matrices of a three-dimensional 2-node Timoshenko beam element
   
       K_l = K_local(ex, ey, ez, ep_K)
       M_l = M_local(ex, ey, ez, ep_M, type='added mass')
       G   = rotation(ex,ey,ez,eo)
       
       K_e = G.T*K_l*G
       M_e = G.T*M_l*G
       
       return M_e, K_e


def dynamic_stiffness_matrix(ex, ey, ez, eo, ep_K, ep_M, omega):
       
## Function that contains the dynamic stiffness matrix for a Timoshenko beam element in global coordinate system
## Input:
                       ## Mass_matrix: matrix of 12x12 containing the mass matrix of a three-dimensional 2-node Timoshenko beam element
                       ## Stiffness_matrix: matrix of 12x12 containing the stiffness matrix of a three-dimensional 2-node Timoshenko beam element
                       ## omega: angular frequency [rad/s]

          M_e, K_e = T_element(ex, ey, ez, eo, ep_K, ep_M)
## Output: matrix of 12x12 containing the dynamic stiffness matrix of a three-dimensional 2-node Timoshenko beam element
          K_dyn = K_e - omega**2 * M_e
          return K_dyn
