import mpmath as mp
mp.dps = 200

#jacobi polynomial function
def jacobi_P(n, alpha, beta, x):
    x = mp.mpf(x)
    return mp.binomial(n + alpha, n) * mp.hyp2f1(-n, n + alpha + beta + 1, alpha + 1, (1 - x) / 2)

#find roots of jacobi polynomial
def jacobi_roots_radau_interior(n, dps_scan=None):
    alpha, beta = mp.mpf(1), mp.mpf(0)
    f = lambda x: jacobi_P(n, alpha, beta, x)

    if dps_scan is None:
        dps_scan = max(2000, 400 * n)
    xs = [mp.mpf(-1) + 2*i/(dps_scan-1) for i in range(dps_scan)]
    fs = [f(x) for x in xs]

    roots = []
    for i in range(dps_scan-1):
        a, b = xs[i], xs[i+1]
        fa, fb = fs[i], fs[i+1]
        if fa == 0:
            roots.append(a)
            continue
        if fa*fb < 0:
            for _ in range(200):
                m = (a + b) / 2
                fm = f(m)
                if abs(fm) < mp.mpf('1e-70') or abs(b - a) < mp.mpf('1e-50'):
                    roots.append(m)
                    break
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm

    roots = [r for r in roots if -1 < r < 1]
    roots = sorted(set([mp.nstr(r, 60) for r in roots]))
    roots = [mp.mpf(r) for r in roots]

    if len(roots) != n:
        if dps_scan < 20000:
            return jacobi_roots_radau_interior(n, dps_scan=20000)
        raise RuntimeError(f"Expected {n} interior roots, got {len(roots)}")

    return roots

#generate radau nodes
def radau_right_nodes_on_01(s):
    if s < 1:
        raise ValueError("Radau quadrature requires s >= 1.")
    if s == 1:
        return [mp.mpf('1')]
    interior = jacobi_roots_radau_interior(s - 1)
    x_all = interior + [mp.mpf(1)]
    c = [ (x + 1) / 2 for x in x_all ]
    return c

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

#build tableau matrices
def build_A_b(c):
    s = len(c)
    A = [[mp.mpf(0) for _ in range(s)] for _ in range(s)]
    b = [mp.mpf(0)] * s
    for j in range(s):
        Lj = lagrange_basis(c, j)
        b[j] = mp.quad(Lj, [0, 1])
        for i in range(s):
            A[i][j] = mp.quad(Lj, [0, c[i]])
    return A, b

#format number string
def nstr_fixed(x, digits=80):
    return mp.nstr(x, n=digits)


#verify tableau correctness
def check_tableau(A, b, c):
    for i in range(len(c)):
        ssum = mp.fsum(A[i])
        diff = ssum - c[i]
    for k in range(min(10, 2*len(c))):
        lhs = mp.fsum([b[j] * c[j]**k for j in range(len(c))])
        rhs = mp.mpf(1)/(k+1)
        err = lhs - rhs

#main radau builder function
def build_radau_irk(s):
    mp.mp.dps = 200
    c = radau_right_nodes_on_01(s)
    A, b = build_A_b(c)
    check_tableau(A, b, c)
    return A, b, c