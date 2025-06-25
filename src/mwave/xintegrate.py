import numpy as np
from numba import jit
from tqdm import tqdm
import math

def kvec_to_x(kvec, klaser):
    return np.fft.fftshift(np.fft.fftfreq(len(kvec), np.diff(kvec)[0])*2*np.pi/klaser)

def x_to_kvec(x, klaser):
    return np.fft.fftshift(np.fft.fftfreq(len(x), np.diff(x)[0])*2*np.pi/klaser)

def pspace_to_xspace(vec):
    return np.fft.fftshift(np.fft.ifft(vec))

def xspace_to_pspace(vec):
    return np.fft.fft(np.fft.ifftshift(vec))

# Define split step integrator
def splitstep(x, tfinal, dt, Vfnc, psi0, klaser, store_hist = True, progress=True):

    # Ensure phi0 is correct type
    psi0 = psi0.astype(np.complex128)
    
    # Determine number of steps, create time vector
    nsteps = math.ceil(tfinal/dt)
    tvec = np.linspace(0, tfinal, nsteps)
    dt = np.diff(tvec)[0]

    # Create vector to store output
    if store_hist:
        psi = np.full((nsteps, len(psi0)), np.nan, dtype=np.complex128)
    
    # Create n state vector
    nvec = x_to_kvec(x, klaser)/2

    # Create step operators
    Hp = 4*nvec.astype(np.complex128)**2
    @jit(nopython=True)
    def Ux(t):
        return np.exp(-1j*Vfnc(t).astype(np.complex128)*dt/2)
    Up = np.exp(-1j*Hp*dt)

    # Enter loop
    if store_hist:
        psi[0,:] = np.copy(psi0)
    loopiter = range(1, nsteps)
    if progress:
        loopiter = tqdm(loopiter)
    for i in loopiter:
        t0 = tvec[i-1]
        tf = tvec[i]
        teval1 = t0+dt/4
        teval2 = t0+3*dt/4
        
        psi0 = Ux(teval2)*pspace_to_xspace(Up*xspace_to_pspace(Ux(teval1)*psi0))
        if store_hist:
            psi[i,:] = np.copy(psi0)
        
    # Return
    if store_hist:
        return psi
    else:
        return psi0
    
def prop_cn(xvec, tfinal, dt, Vvec, psi0):
    
    nsteps = math.ceil(tfinal/dt)
    tvec = np.linspace(0, tfinal, nsteps)
    dt = np.diff(tvec)[0]

    dx = np.diff(xvec)[0]
    nx = len(xvec)
    H_od = (-np.ones(nx-1)/(2*(dx**2))).astype(np.complex128)
    H_diag_ke = 2*np.ones(nx)/(2*(dx**2))
    
    psi1 = np.copy(psi0)
    psi2 = np.full_like(psi1, np.nan, dtype=np.complex128)
    for i in tqdm(range(1, len(tvec))):
        H_diag1 = (Vvec(tvec[i-1])+H_diag_ke).astype(np.complex128)
        H_diag2 = (Vvec(tvec[i])+H_diag_ke).astype(np.complex128)
        
        _trimatmul(-0.5j*H_od*dt, 1-0.5j*H_diag1*dt, np.copy(-0.5j*H_od*dt), psi1, psi2)
        _trimatdiv(0.5j*H_od*dt, 1+0.5j*H_diag2*dt, np.copy(0.5j*H_od*dt), psi2, psi1)

    return psi1
    
@jit(nopython=True)
def _trimatmul(a,b,c,x,y):
    "a, b, c are the diagonals of the matrix, x is the vector to multiply, y is the new vector. Solves y=Ax for known x."

    # First element
    y[0] = x[0]*b[0]+x[1]*c[0]

    # All intermediate elements
    for i in range(1,len(b)-1):
        y[i] = x[i-1]*a[i-1]+x[i]*b[i]+x[i+1]*c[i]

    # Last element
    y[-1] = x[-1]*b[-1] + x[-2]*a[-1]

@jit(nopython=True)
def _trimatdiv(a,b,c,x,y):
    "a, b, c are the diagonals of the matrix, solves Ay=x for known x. Uses the Thomas algorithm."

    # Forward sweep
    c[0] = c[0]/b[0]
    x[0] = x[0]/b[0]
    for i in range(1,len(b)-1):
        c[i] = c[i]/(b[i]-a[i-1]*c[i-1])
        x[i] = (x[i] - a[i-1]*x[i-1])/(b[i]-a[i-1]*c[i-1])
    x[-1] = (x[-1] - a[-1]*x[-2])/(b[-1]-a[-1]*c[-1])

    # Backward sweep
    y[-1] = x[-1]
    for i in range(len(b)-2,-1,-1):
        y[i] = x[i] - c[i]*y[i+1]