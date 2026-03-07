import mpmath as mp

#compute bdf coefficients
def bdf_coeffs(k, prec=300):
    mp.mp.dps = prec
    h = mp.mpf('1')
    A = mp.matrix([[(-j*h)**m for j in range(k+1)] for m in range(k+1)])
    b = mp.matrix([mp.mpf('0') if m == 0 else h*m*(0**(m-1)) for m in range(k+1)])
    alpha = mp.lu_solve(A, b)
    alpha = [a/alpha[0] for a in alpha]
    
    #computes the rhs beta naught coefficient
    beta_0 = -mp.fsum([j * alpha[j] for j in range(k+1)])
    return alpha, beta_0

#verify bdf order accuracy
def check_order_bdf(alpha, beta_0):
    k = len(alpha) - 1
    errs = []
    h = mp.mpf('1')
    for p in range(k+1):
        lhs = mp.fsum(alpha[j]*(-j*h)**p for j in range(k+1))
        rhs = h * beta_0 * p * (0**(p-1)) if p > 0 else 0
        errs.append(abs(lhs - rhs))
    return max(errs)

#adaptive precision bdf computation
def adaptive_bdf(k, max_prec=2000, target_error=1e-80):
    prec = 200
    best_coeffs = None
    best_beta = None
    best_err = mp.inf
    while prec <= max_prec:
        mp.mp.dps = prec
        coeffs, beta_0 = bdf_coeffs(k, prec)
        err = check_order_bdf(coeffs, beta_0)
        if err < best_err:
            best_err, best_coeffs, best_beta = err, coeffs, beta_0
        if err < target_error:
            return best_coeffs, best_beta, prec, err
        prec = int(prec * 1.5)
    return best_coeffs, best_beta, prec, best_err