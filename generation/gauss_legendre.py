import mpmath as mp
mp.dps = 200

#legendre roots and weights generator
def legendre_roots_and_weights(n):
    xs = [mp.cos(mp.pi*(4*k - 1)/(4*n + 2)) for k in range(1, n+1)]
    Pn = lambda x: mp.legendre(n, x)

    def Pn_prime(x):
        den = x*x - 1
        if abs(den) > mp.mpf('1e-30'):
            return n * (x * mp.legendre(n, x) - mp.legendre(n-1, x)) / den
        else:
            return mp.diff(Pn, x)

    for k in range(n):
        x = xs[k]
        for _ in range(80):
            fx, dfx = Pn(x), Pn_prime(x)
            if dfx == 0:
                break
            x_new = x - fx/dfx
            if mp.almosteq(x_new, x, rel_eps=mp.eps*100, abs_eps=mp.eps*100):
                break
            x = x_new
        xs[k] = x

    ws = []
    for x in xs:
        dPn = Pn_prime(x)
        ws.append( mp.mpf(2) / ((1 - x*x) * (dPn*dPn)) )

    c = [ (x + 1)/2 for x in xs ]
    b = [ w/2 for w in ws ]
    perm = sorted(range(n), key=lambda i: c[i])   
    c_sorted = [c[i] for i in perm]
    b_sorted = [b[i] for i in perm]
    return c_sorted, b_sorted, perm

#lagrange basis polynomial
def lagrange_basis(c, j):
    xj = c[j]
    others = [c[k] for k in range(len(c)) if k != j]
    denom = mp.mpf(1)
    for xk in others:
        denom *= (xj - xk)
    def Lj(x):
        num = mp.mpf(1)
        for xk in others:
            num *= (x - xk)
        return num / denom
    return Lj

#build gauss legendre irk tableau
def build_gauss_legendre_irk(s):
    mp.mp.dps = 200
    c, b, perm = legendre_roots_and_weights(s)

    A = [[mp.mpf(0) for _ in range(s)] for _ in range(s)]
    for j in range(s):
        Lj = lagrange_basis(c, j)
        for i in range(s):
            A[i][j] = mp.quad(Lj, [0, c[i]])

    for i in range(s):
        rs, diff = mp.fsum(A[i]), mp.fsum(A[i]) - c[i]

    for k in range(min(10, 2*s)):
        lhs = mp.fsum([b[j]*c[j]**k for j in range(s)])
        rhs = mp.mpf(1)/(k+1)

    return A, b, c