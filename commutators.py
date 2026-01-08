import sympy as sp 
δ = sp.KroneckerDelta

from sympy.tensor.indexed import IndexedBase

g = IndexedBase('g')

def op(p, q): # symbolic label for a†_p a_q 
    return sp.Symbol(f"a†_{p}a_{q}")

# def op2(p, q, r, s): # symbolic label for a†_p a_q 
#     return sp.Symbol(f"a†_{p}a†_{r}a_{q}a_{s}")

def op2(p, q, r, s): #a_p^dagger a_q a_r^dagger a_s 
    # return sp.Symbol(
    #     f"RDM2({p}, {q}, {r}, {s}, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2)"
    # )
    return sp.Symbol(
        f"{p}, {s}, {r}, {q}"
    )
    
def one_comm(P, Q, R, S): # [a†_P a_Q, a†_R a_S] 
    return δ(Q, R) * op(P, S) - δ(P, S) * op(R, Q)

# def nested(P,Q,R,S,M,N): # [a†_P a_Q, [a†_R a_S, a†_M a_N]] 
#     # P,Q,R,S,M,N, T, U= sp.symbols("P,Q,R,S,M,N,T,U", integer=True) 
#     # N=sp.symbols
#     # element = [(S,M),(Q, R)] sp.Symbol(f"a†_{P}a_{N}")
#     return δ(S,M)*δ(Q, R) * op(P,N) - δ(S,M)*δ(P, N) * op(R,Q) - δ(R,N)*δ(Q, M) * op(P,S) + δ(R,N)*δ(P, S) * op(M,Q)
#     # return (S,M)(Q,R)


def nested(P, Q, R, S, M, N):  # [a†_P a_Q, [a†_R a_S, a†_M a_N]] 
    return δ(S,M)*δ(Q,R)*op(P,N) - δ(S,M)*δ(P,N)*op(R,Q) - δ(R,N)*δ(Q,M)*op(P,S) + δ(R,N)*δ(P,S)*op(M,Q)

def two_and_one(P,Q,R,S,M,N): #[a†_R a_S a†_M a_N,a†_P a_Q]
    return -δ(Q,R)*op(P,S)*op(M,N)+δ(P,S)*op(R,Q)*op(M,N)-δ(Q,M)*op(R,S)*op(P,N)+δ(P,N)*op(R,S)*op(M,Q)

def two_electron_part(T, U, P, Q, R, S, M, N):
    term1=(δ(N,P)*δ(U,M)*op2(T,S,R,Q)
    -δ(N,P)*δ(T,Q)*op2(M,S,R,U)
    +δ(N,P)*δ(U,R)*op2(M,S,T,Q)
    -δ(N,P)*δ(T,S)*op2(M,U,R,Q))
    
    term2=(-δ(M,Q)*δ(U,P)*op2(T,S,R,N)
    +δ(M,Q)*δ(T,N)*op2(P,S,R,U)
    -δ(M,Q)*δ(U,R)*op2(P,S,T,N)
    +δ(M,Q)*δ(T,S)*op2(P,U,R,N))
    
    
    term3=(+δ(N,R)*δ(U,P)*op2(T,S,M,Q)
    -δ(N,R)*δ(T,Q)*op2(P,S,M,U)
    +δ(N,R)*δ(U,M)*op2(P,S,T,Q)
    -δ(N,R)*δ(T,S)*op2(P,U,M,Q))
    
    
    term4=(-δ(M,S)*δ(U,P)*op2(T,N,R,Q)
    +δ(M,S)*δ(T,Q)*op2(P,N,R,U)
    -δ(M,S)*δ(U,R)*op2(P,N,T,Q)
    +δ(M,S)*δ(T,N)*op2(P,U,R,Q))

    total=term1+term2+term3+term4
    
    return -total


    
P = sp.symbols("P", integer=True)
Q = sp.symbols("Q", integer=True)
M = sp.symbols("M", integer=True)
N = sp.symbols("N", integer=True)
T = sp.symbols("T", integer=True)
U = sp.symbols("U", integer=True)
R = sp.symbols("R", integer=True)
S=  sp.symbols("S", integer=True)



# result = nested(U, T, P, Q, N, M)
# print(" [Q, [H, Q']] where H=a†_P a_Q, Q=a†_T a_U, Q'=a†_M a_N :") 
# print(result)

# two_electron_part_result= two_electron_part(U, T, P, Q, R, S, N, M)
# print(two_electron_part_result)


test=one_comm(Q, P, M, N) # [a†_P a_Q, a†_R a_S] P,Q,R,S
print(-test)