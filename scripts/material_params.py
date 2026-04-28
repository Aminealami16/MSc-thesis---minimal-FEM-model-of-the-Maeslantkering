import numpy as np

def effective_truss_stiffness(d1, d2,t1, t2, h, b): 
    '''Calculates equivalent moment of inertia and area to be used in a Timoshenko beam model

    Inputs:

    d1: Diameter of the diagonal truss elements
    d2: Diameter of the horizontal truss elements
    t1: Thickness of the diagonal truss elements
    t2: Thickness of the horizontal truss elements
    h: Height of the truss
    b: Width of the truss
    z_NC: Distance from the neutral axis to the centroid of the truss elements

    returns:
    A_eq: Effective area of the truss
    I_eqy: Effective moment of inertia around the y-axis (strong axis)
    I_eqz: Effective moment of inertia around the z-axis (weak axis)
    b_eq: Effective width of the truss for shear deformation calculations
    h_eq: Effective height of the truss for shear deformation calculations
    '''
    
    A1 = np.pi * (d1/2)**2 - np.pi * ((d1/2) - 2 * t1)**2
    A2 = np.pi * (d2/2)**2 - np.pi * ((d2/2) - 2 * t2)**2

    z_NC = (A2 * 0 +  2 * h * A2) / (3 * A2)
    y_NC = (A2 * 0 + A2 * 0.5 * b + A2 * b) / (3 * A2)
    I = (np.pi / 4) * ((d2/2)**4 - ((d2/2) - t2)**4)
    Iy_eq = I + A2 * z_NC**2 + 2 * (I + A2 * (h - z_NC)**2)
    Iz_eq = I + A2 * y_NC**2 + I + A2 * (0.5 * b - y_NC)**2 + I + A2 * (b - y_NC)**2
    A_eq = A1 * 5 + A2 * 3
    b_eq  = ((Iz_eq**3) / (Iy_eq* (1/12)**2))**(1/8)
    h_eq = Iz_eq / ((1/12)* b_eq**3)

    Iy_eq = 390 #390
    Iz_eq = 0.45*Iy_eq
    b_eq  = ((Iz_eq**3) / (Iy_eq* (1/12)**2))**(1/8)
    h_eq = Iz_eq / ((1/12)* b_eq**3)
    A_eq = A1 * 5 + A2 * 3
    



    return A_eq, Iy_eq, Iz_eq, b_eq, h_eq







    
def effective_retaining_wall_stiffness(t):

    '''Calculates the effective stiffness of a retaining wall to be used in a Timoshenko beam model

    Inputs:
    t: Thickness of the retaining wall

    parameters:
    h: Height of the retaining wall
    h1: Height of the first step of the retaining wall
    h2: Height of the second step of the retaining wall
    b1: Width of the lower part of the retaining wall
    b2: Width of the upper part of the retaining wall
    z_NC: Distance from the neutral axis to the centroid of the retaining wall elements
    y_NC: Distance from the neutral axis to the centroid of the retaining wall elements in the horizontal direction


    returns:
    A_eq: Effective area of the retaining wall
    I_eqy: Effective moment of inertia around the y-axis (strong axis)
    I_eqz: Effective moment of inertia around the z-axis (weak axis)
    b_eq: Effective width of the retaining wall for shear deformation calculations
    '''
    h = 22 #m
    h1 = 14.5 #m
    h2 = 7.5 #m
    b1 = 8 #m
    b2 = 12 #m

    z_NC = (t * b1 * 0 + h * t * 0.5 * h + h1 * t * 0.5 * h1 + (b2 - b1) * t * h1 + h2 * t * (h2 * 0.5 + h1) + b2 * t * h) / (2 * t * h + b1 * t + b2 * t + (b2 - b1) * t)
    y_NC = (h * t * 0 + t * b1 * 0.5 * b1 + t * b2 * 0.5 * b2 + h1 * t * b1 + h2 * t * b2 + (b2 - b1) * t * (b1 + 0.5 * (b2 - b1))) / (2 * t * h + b1 * t + b2 * t + (b2- b1) * t)
    I_eqy = (1/12) * b1 * t**3 + (1/12) * t * h**3 + (1/12) * t * h1**3 + (1/12) * t * h2**3 + (1/12) * (b2 - b1) * t**3 + (1/12) * b2 * t**3 + b1 * t * z_NC**2 + t * h * (z_NC - 0.5 * h)**2 + h1 * t * (0.5*h1 - z_NC)**2 + h2 * t * (h1 + 0.5 * h2 - z_NC)**2 + b2 * t * (h - z_NC)**2 + (b2 - b1) * t * (z_NC - h1)**2
    I_eqz = (1/12) * h * t ** 3 + (1/12) * t * b2**3 + (1/12) * t * b1**3 + (1/12) * h1 * t**3 + (1/12) * t * (b2 - b1) **3  + (1/12) * h2 * t**3 + h * t * y_NC**2 + b1 * t * (y_NC - 0.5 * b1)**2 + b2 * t * (y_NC - 0.5 * b2)**2 + h1 * t * (b1 - y_NC)**2 + h2 * t * (b2 - y_NC)**2 + (b2 - b1) * t * (b1+ 0.5 * (b2 - b1) - y_NC)**2

    A_eq = 2 * t * h + b1 * t + b2 * t + (b2 - b1) * t
    b_eq = A_eq / h
    m =  6455148
    L_tot = 237.5*2* np.pi * (50/360)
    rho = m / ( A_eq * L_tot)

    return A_eq, I_eqy, I_eqz, b_eq, rho




def stiffness_fenders():

    '''Calculates the stiffness of the fenders to be used in a Timoshenko beam model

    Inputs:
    None

    returns:
    K_spring_linear: Linear stiffness of the fenders [N/m]
    '''

    # K_spring_linear = 4 * (250 * 1000) / (90 / 1000) # retrieved from report Handboek 2: Kerende wand en vakwerkarmen
    K_spring_linear = ((1200+1750) / 2) * 10**6  # Retrieved from HVR engineering report
    K_spring_linear = 10000 * 1000
    return K_spring_linear
   


def stiffness_connecting_beams():

    '''Calculates the stiffness of the connecting beams to be used in a Timoshenko beam model

    Inputs:
    None

    Parameters:
    h: Height of the connecting beams
    d_out: Outer diameter of the connecting beams
    d_in: Inner diameter of the connecting beams

    returns:
    Iy: Moment of inertia around the y-axis (strong axis) of the connecting beams
    Iz: Moment of inertia around the z-axis (weak axis) of the connecting beams
    Ip: Polar moment of inertia of the connecting beams
    It: Torsional constant of the connecting beams
    A: Cross-sectional area of the connecting beams
    '''
    # h = 18
    # d_out = 1.8 
    # d_in = 1.8  - (2 * 80) / 1000
    # A = 2 * np.pi * (d_out/2)**2 - np.pi * (d_in/2)**2
    # Iy = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) + A * (h/2)**2) 
    # Iz = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) )
    # Ip = Iy + Iz    
    # It = Ip 

    d_out = 1.2
    d_in = 1.2 - (30 / 1000)
    h = 18
    A = 2 * np.pi * (d_out/2)**2 - np.pi * (d_in/2)**2
    Iy = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) + A * (h/2)**2) 
    Iz = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) ) + A
    Ip = Iy + Iz
    It = Ip
    

    return Iy, Iz, Ip, It, A

def stiffness_braces():
    d_out = 0.5
    d_in = 0.5 - (10 / 1000)
    h = 18
    A = 2 * np.pi * (d_out/2)**2 - np.pi * (d_in/2)**2
    Iy = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) + A * (h/2)**2) 
    Iz = 2 * ( (np.pi / 4) * ((d_out/2)**4 - (d_in/2)**4) ) + A
    Ip = Iy + Iz
    It = Ip
    return Iy, Iz, Ip, It, A

def stiffness_connecting_truss(d1, d2,t1, t2, h, b):

    A1 = np.pi * (d1/2)**2 - np.pi * ((d1/2) - 2 * t1)**2
    A2 = np.pi * (d2/2)**2 - np.pi * ((d2/2) - 2 * t2)**2

    z_NC = (A2 * 0 +  2 * h * A2) / (3 * A2)
    y_NC = (A2 * 0 + A2 * 0.5 * b + A2 * b) / (3 * A2)
    I = (np.pi / 4) * ((d2/2)**4 - ((d2/2) - t2)**4)
    Iy_eq = I + A2 * z_NC**2 + 2 * (I + A2 * (h - z_NC)**2)
    Iz_eq = I + A2 * y_NC**2 + I + A2 * (0.5 * b - y_NC)**2 + I + A2 * (b - y_NC)**2
    A_eq = A1 * 5 + A2 * 3
    b_eq  = ((Iz_eq**3) / (Iy_eq* (1/12)**2))**(1/8)
    h_eq = Iz_eq / ((1/12)* b_eq**3)

    Ieqy = 300
    Ieqz = 0.3 * Ieqy
    b_eq  = ((Iz_eq**3) / (Iy_eq* (1/12)**2))**(1/8)
    h_eq = Iz_eq / ((1/12)* b_eq**3)
    A_eq = A1 * 5 + A2 * 3

    return A_eq, Iy_eq, Iz_eq, b_eq, h_eq

