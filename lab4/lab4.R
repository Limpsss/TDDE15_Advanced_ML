# ------------ 2.1 ------------ 

# Squared exponential kernel: k(x, x') = sigmaF^2 * exp(-(x - x')^2 / (2 l^2))
# Returns the n1 x n2 covariance matrix where entry (i, j) = k(x1[i], x2[j])
SquaredExpKernel <- function(x1,x2,sigmaF=1,l=3) {
  n1 <- length(x1) # number of points in x1 (rows of K)
  n2 <- length(x2) # number of points in x2 (columns of K)
  K <- matrix(NA,n1,n2) # empty n1 x n2 matrix to fill in
  for (i in 1:n2){
    # Column i: similarity between every x1 and the single point x2[i]
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}

posteriorGP <- function(X, y, XStar, sigmaNoise, k, ...) {
  n <- length(X) # number of training points
  K <- k(X, X, ...)  # K(X, X), n x n: covariance between training points
  kStar <- k(X, XStar, ...) # K(X, X*), n x m: covariance between training and test points

  # Algorithm 2.1
  noise_diag <- sigmaNoise^2 * diag(n) # σ^2*I, noise variance on the diagonal
  # Cholesky decomposition. R returns upper triangular matrix so we transpose
  L <- t(chol(K + noise_diag))

  alpha <- solve(t(L), solve(L, y))
  # Posterior mean of f at XStar
  fStar <- t(kStar) %*% alpha

  v <- solve(L, kStar)
  # Posterior covariance of f at XStar
  V_fStar <- k(XStar, XStar, ...) - t(v) %*% v

  # Log marginal likelihood, not needed?
  log_lik <- -1/2*t(y)%*%alpha - sum(log(diag(L))) - n/2*log(2*pi)
  
  return(list(mean = as.vector(fStar), variance = diag(V_fStar)))
}

X <- c(0.4)
y <- c(0.719)

XStar <- seq(-1, 1, length.out = 100)

res <- posteriorGP(X, y, XStar, 0.1, k = SquaredExpKernel, sigmaF = 1, l = 0.3)
res$mean
res$variance


