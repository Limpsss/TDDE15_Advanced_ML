# ----------------------------------- 2.1 -------------------------------------

# ------------------ (1) --------------------
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

# ------------------ (2) --------------------

plot_gp <- function(XStar, pos_mean, pos_variance, X, y) {
  # 95% confidence intervals
  lower_bound <- pos_mean - 1.96*sqrt(pos_variance)
  upper_bound <- pos_mean + 1.96*sqrt(pos_variance)
  
  # plot posterior mean
  plot(XStar, pos_mean, type="p", ylim = range(c(lower_bound, upper_bound)))
       
  # plot confidence intervals
  lines(XStar, lower_bound, col = "blue", lwd = 2)
  lines(XStar, upper_bound, col = "blue", lwd = 2)
  
  # plot training data points
  points(X, y, pch = 19, col = "red")
}

# (x, y) = (0.4, 0.719)
X <- c(0.4)
y <- c(0.719)
# interval x ∈ [−1, 1]
XStar <- seq(-1, 1, length.out = 100)

res <- posteriorGP(X, y, XStar, sigmaNoise=0.1, SquaredExpKernel, sigmaF=1, l=0.3)
plot_gp(XStar, res$mean, res$variance, X, y)

# ------------------ (3) --------------------

# updating prior data
X <- c(0.4, -0.6)
y <- c(0.719, -0.044)

# same hyperparams
res <- posteriorGP(X, y, XStar, sigmaNoise=0.1, SquaredExpKernel, sigmaF=1, l=0.3)
plot_gp(XStar, res$mean, res$variance, X, y)

# ------------------ (4) --------------------

# updating prior data
X <- c(-1, -0.6, -0.2, 0.4, 0.8)
y <- c(0.768, -0.044, -0.940, 0.719, -0.664)

# same hyperparams
res <- posteriorGP(X, y, XStar, sigmaNoise=0.1, SquaredExpKernel, sigmaF=1, l=0.3)
plot_gp(XStar, res$mean, res$variance, X, y)

# ------------------ (5) --------------------

# l = 1
res <- posteriorGP(X, y, XStar, sigmaNoise=0.1, SquaredExpKernel, sigmaF=1, l=1)
plot_gp(XStar, res$mean, res$variance, X, y)

# ----------------------------------- 2.2 -------------------------------------

# ------------------ (setup) --------------------
library(kernlab)

data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/TempTullinge.csv", header=TRUE, sep=";")

time <- seq(1, 2190, 5)
day <- rep(seq(1, 365, 5), 6)

# ------------------ (1) --------------------

SEKernel <- function(sigmaF, l) {
  kval = function (X, XStar) {
    K = matrix(NA, length(X), length(XStar))
    for (i in 1:length(X)) {
      K[, i] = sigmaF ^ 2 * exp(-0.5 * ((X - XStar[i]) / l) ^ 2)
    }
    return(K)
  }
  class(kval) = 'kernel'
  return (kval)
}

se_kernel <- SEKernel(1, 0.3)
se_kernel(1, 2)

X <- c(1, 3, 4)
XStar <- c(2, 3, 4)
kernelMatrix(kernel = se_kernel, x = X, y = XStar)

# ------------------ (2) --------------------





















