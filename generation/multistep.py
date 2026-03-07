import mpmath as mp

def adams_bashforth_coeffs(k, prec=500):
    #compute adams-bashforth coefficients for order k
    if k >= 12: prec = max(prec, 1000)
    
    mp.mp.dps = prec
    #lagrange interpolation points for ab
    xs = [-j for j in range(k)]
    coeffs = []
    for j in range(k):
        def ell(x):
            #lagrange basis polynomial for point j
            num, den = mp.mpf('1'), mp.mpf('1')
            for m in range(k):
                if m != j:
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
            return num/den
        #integrate basis polynomial over [0,1]
        coeffs.append(mp.quad(ell, [0, 1], method='gauss-legendre', maxdegree=100))
    return coeffs

#compute adams-moulton coefficients for order k
def adams_moulton_coeffs(k, prec=500):
    if k >= 12: prec = max(prec, 1000)
    mp.mp.dps = prec

    #lagrange interpolation points for am
    xs = [1] + [-j for j in range(k-1)]
    coeffs = []
    for j in range(k):
        def ell(x):
            #lagrange basis polynomial for point j
            num, den = mp.mpf('1'), mp.mpf('1')
            for m in range(k):
                if m != j:
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
            return num/den
        #integrate basis polynomial over [0,1]
        coeffs.append(mp.quad(ell, [0, 1], method='gauss-legendre', maxdegree=100))
    return coeffs

#checks the order conditions for the multistep coefficients
def check_order(coeffs, kind):
    k = len(coeffs)
    xs = [-j for j in range(k)] if kind=="AB" else [1]+[-j for j in range(k-1)]
    errs = []
    for p in range(k):
        lhs = mp.fsum(coeffs[j]*xs[j]**p for j in range(k))
        rhs = 1/(p+1)
        error = abs(lhs-rhs)
        errs.append(error)
    max_error = max(errs)
    return max_error

#returns the adaptive precision coefficients
def adaptive_precision_coeffs(k, kind="AB", max_prec=2000, target_error=1e-100):
    prec = 200
    best_coeffs = None
    best_error = float('inf')
    
    while prec <= max_prec:
        try:
            if kind == "AB": coeffs = adams_bashforth_coeffs(k, prec)
            else: coeffs = adams_moulton_coeffs(k, prec)
            current_error = check_order(coeffs, kind)
            
            if current_error < best_error:
                best_coeffs = coeffs
                best_error = current_error
            
            if current_error < target_error: return coeffs, prec, current_error
            prec = int(prec * 1.5)
        except Exception as e:
            break
    
    return best_coeffs, prec//2, best_error

#returns the Adams-Bashforth coefficients
def get_ab_coeffs(k, prec=500, adaptive=False, max_prec=2000, target_error=1e-100):
    if adaptive:
        coeffs, _, _ = adaptive_precision_coeffs(k, "AB", max_prec=max_prec, target_error=target_error)
        return coeffs
    return adams_bashforth_coeffs(k, prec=prec)

#returns the Adams-Moulton coefficients
def get_am_coeffs(k, prec=500, adaptive=False, max_prec=2000, target_error=1e-100):
    if adaptive:
        coeffs, _, _ = adaptive_precision_coeffs(k, "AM", max_prec=max_prec, target_error=target_error)
        return coeffs
    return adams_moulton_coeffs(k, prec=prec)