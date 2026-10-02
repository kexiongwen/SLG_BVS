import numpy as np
from scipy.linalg.blas import drot, drotg
from scipy.linalg import solve_triangular

# references for updates:
#   - Golub and van Loan (4th ed.) Section 6.5.4
#   - http://mathoverflow.net/questions/30162/is-there-a-way-to-simplify-block-cholesky-decomposition-if-you-already-have-deco
#
# references for downdates:
#   - Golub and van Loan (4th ed.) Section 6.5.4
#   - Alexander, Pan, and Plemmons, 1988
#     http://ac.els-cdn.com/0024379588901589/1-s2.0-0024379588901589-main.pdf?_tid=c35b3e32-8640-11e4-a30d-00000aacb35d&acdnat=1418857524_90c69b8bbe27d77d950c45d2c5d43410
# algorithm numbers refer to the APP98 paper

# other implementations:
#   - M. Seeger implementation is better than LINPACK mainly because it uses
#     BLAS, but it's all matlabby and also GPL:
#     http://www.ams.org/journals/mcom/1974-28-126/S0025-5718-1974-0343558-6/home.html
#     http://ipg.epfl.ch/~seeger/lapmalmainweb/software/index.shtml
#   - M. Hoffman wrapped the dchud/dchdd routines of LINPACK
#     https://github.com/mwhoffman/pychud


def update(R,z):
    n = z.shape[0]
    for k in range(n):
        c, s = drotg(R[k,k],z[k])
        R[k,:], z[:] = drot(R[k,:],z,c,s,overwrite_x=True,overwrite_y=True)
    
    return R

def downdate(R,z):
    n = z.shape[0]

    for k in range(n):
        tk = z[k] / R[k,k]

        if not np.abs(tk)<1:
            raise ValueError("downdate failed: matrix not positive definite")

        ck = 1./np.sqrt(1-tk**2)
        sk = ck*tk

        R[k,:], z[:] = ck*R[k,:] - sk*z, -sk*R[k,:] + ck*z
    
    return R

def downdate_stable(R,z):
    
    n = R.shape[0]

    for k in range(n):
        if not R[k,k]>np.abs(z[k]):
            raise ValueError("downdate_stable failed: matrix not positive definite")

        rbar = np.sqrt((R[k,k] - z[k])*(R[k,k] + z[k]))
        for j in range(k+1,n):
            R[k,j] = 1./rbar * (R[k,k]*R[k,j] - z[k]*z[j])
            z[j] = 1./R[k,k] * (rbar*z[j] - z[k]*R[k,j])
        R[k,k] = rbar
        
    return R

def chol_del(L,index):

    P,_=np.shape(L)
    L=L.T
    
    if index>0 and index< P-1:
    
        L_new=np.zeros((P-1,P-1))
        
        #update L11
        
        L_new[0:index,0:index]=L[0:index,0:index]
        
        #update L13
        
        L_new[0:index,index:]=L[0:index,index+1:]
        
        #update L33
                
        L_new[index:,index:]=update(L[index+1:,index+1:].copy(),L[index,index+1:].copy())
    
    elif index==P-1:
        
        L_new=L[0:P-1,0:P-1]
    
    else:
        
        #update L33
        
        L_new=update(L[index+1:,index+1:].copy(),L[index,index+1:].copy())
    
    return L_new.T


def chol_add(A,L,index):
    
    P,_=np.shape(A)
    L=L.T
    L_new=np.zeros((P,P))
    
    if index>0 and index< P-1:
        
        #update L11
        
        L_new[0:index,0:index]=L[0:index,0:index]
        
        #update L12
        
        L_new[0:index,index]=solve_triangular(L[0:index,0:index].T,A[0:index,index], lower=True, check_finite=False)
        
        #update L13
        
        L_new[0:index,index+1:]=L[0:index,index:]
        
        #update L22

        d22=A[index,index]-(L_new[0:index,index]**2).sum()

        if d22<=0:
            raise ValueError("chol_add failed: matrix not positive definite")

        L_new[index,index]=np.sqrt(d22)

        #update L23

        L_new[index,index+1:]=(A[index,index+1:]-L_new[0:index,index].T@L_new[0:index,index+1:])/L_new[index,index]
        
        #update L33
        
        L_new[index+1:,index+1:]=downdate(L[index:,index:].copy(),L_new[index,index+1:].copy())
        
    elif index==P-1:
        
        #update L11
        
        L_new[0:index,0:index]=L[0:index,0:index]
        
        #update L12
        
        L_new[0:index,index]=solve_triangular(L[0:index,0:index].T,A[0:index,index], lower=True, check_finite=False)
        
        #update L22

        d22=A[index,index]-(L_new[0:index,index]**2).sum()

        if d22<=0:
            raise ValueError("chol_add failed: matrix not positive definite")

        L_new[index,index]=np.sqrt(d22)
        
    else:
        
        #update L22
        
        L_new[index,index]=np.sqrt(A[index,index])
        
        #update L23
        
        L_new[index,index+1:]=A[index:index+1,index+1:]/L_new[index,index]
        
        #update L33
        
        L_new[index+1:,index+1:]=downdate(L[index:,index:].copy(),L_new[index,index+1:].copy())

    return L_new.T

def chol_add_col(v,a,L,index):

    #insert a column at position index given only its cross-products v against
    #the current model (in model order) and its squared norm a; equivalent to
    #chol_add(A,L,index) with A the Gram submatrix, A[0:index,index]=v[0:index],
    #A[index,index]=a, A[index,index+1:]=v[index:], but avoids assembling A

    P=len(v)+1
    L=L.T
    L_new=np.zeros((P,P))

    #update L11, L13

    L_new[0:index,0:index]=L[0:index,0:index]
    L_new[0:index,index+1:]=L[0:index,index:]

    #update L12

    if index>0:
        L_new[0:index,index]=solve_triangular(L[0:index,0:index].T,v[0:index], lower=True, check_finite=False)

    #update L22

    d22=a-(L_new[0:index,index]**2).sum()

    if d22<=0:
        raise ValueError("chol_add_col failed: matrix not positive definite")

    L_new[index,index]=np.sqrt(d22)

    #update L23, L33

    if index<P-1:
        L_new[index,index+1:]=(v[index:]-L_new[0:index,index].T@L_new[0:index,index+1:])/L_new[index,index]
        L_new[index+1:,index+1:]=downdate(L[index:,index:].copy(),L_new[index,index+1:].copy())

    return L_new.T

