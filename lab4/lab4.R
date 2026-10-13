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
  noise_diag <- sigmaNoise^2 * diag(n) # sigma^2*I, noise variance on the diagonal
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
# interval x in [-1, 1]
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
temp <- data$temp[time] # subsampled temperatures (every fifth day)

# ------------------ (1) --------------------

SEKernel <- function(sigmaF, l) {
  kval = function (X, XStar) {
    K = matrix(NA, length(X), length(XStar))
    for (i in 1:length(XStar)) {
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

# Noise variance from the residuals of a quadratic regression on time
quadFit <- lm(temp ~ time + I(time^2))
sigmaNoise <- sd(quadFit$residuals)

# GP fit with our SE kernel, sigmaF = 20, l = 100
GPfit <- gausspr(time, temp, kernel = SEKernel, kpar = list(sigmaF = 20, l = 100),
                 var = sigmaNoise^2, variance.model = TRUE, scaled = FALSE)
meanPred <- predict(GPfit, time)
sdPred <- predict(GPfit, time, type = "sdeviation")

plot(time, temp, pch = 20, col = "grey", main = "GP on time (gausspr)")
lines(time, meanPred, col = "red", lwd = 2)
# 95% bands for f
lines(time, meanPred - 1.96*sdPred, col = "blue", lwd = 2)
lines(time, meanPred + 1.96*sdPred, col = "blue", lwd = 2)

# ------------------ (3) --------------------

# Same model, but with own implementation of Algorithm 2.1 from 2.1
res <- posteriorGP(time, temp, time, sigmaNoise = sigmaNoise, SquaredExpKernel, sigmaF = 20, l = 100)

plot(time, temp, pch = 20, col = "grey", main = "GP on time (Algorithm 2.1)")
lines(time, res$mean, col = "red", lwd = 2)
# 95% bands for f
lines(time, res$mean - 1.96*sqrt(res$variance), col = "blue", lwd = 2)
lines(time, res$mean + 1.96*sqrt(res$variance), col = "blue", lwd = 2)

# ------------------ (4) --------------------

# Noise variance from a quadratic regression on day
quadFitDay <- lm(temp ~ day + I(day^2))
sigmaNoiseDay <- sd(quadFitDay$residuals)

GPfitDay <- gausspr(day, temp, kernel = SEKernel, kpar = list(sigmaF = 20, l = 100),
                    var = sigmaNoiseDay^2, scaled = FALSE)
meanPredDay <- predict(GPfitDay, day)

# Both posterior means with time on the horizontal axis
plot(time, temp, pch = 20, col = "grey", main = "GP on time vs GP on day")
lines(time, meanPred, col = "red", lwd = 2)
lines(time, meanPredDay, col = "darkgreen", lwd = 2)
legend("bottomright", legend = c("f(time)", "f(day)"), col = c("red", "darkgreen"), lwd = 2)

# ------------------ (5) --------------------

# Locally periodic kernel: periodic part (l1, d) times SE part (l2)
PeriodicKernel <- function(sigmaF, l1, l2, d) {
  kval = function (X, XStar) {
    K = matrix(NA, length(X), length(XStar))
    for (i in 1:length(XStar)) {
      K[, i] = sigmaF ^ 2 * exp(-2 * sin(pi * abs(X - XStar[i]) / d) ^ 2 / l1 ^ 2) *
                            exp(-0.5 * ((X - XStar[i]) / l2) ^ 2)
    }
    return(K)
  }
  class(kval) = 'kernel'
  return (kval)
}

GPfitPer <- gausspr(time, temp, kernel = PeriodicKernel,
                    kpar = list(sigmaF = 20, l1 = 1, l2 = 100, d = 365),
                    var = sigmaNoise^2, scaled = FALSE)
meanPredPer <- predict(GPfitPer, time)

# Compare all three models
plot(time, temp, pch = 20, col = "grey", main = "Comparison of the three GP models")
lines(time, meanPred, col = "red", lwd = 2)
lines(time, meanPredDay, col = "darkgreen", lwd = 2)
lines(time, meanPredPer, col = "purple", lwd = 2)
legend("bottomright", legend = c("SE on time", "SE on day", "Locally periodic on time"),
       col = c("red", "darkgreen", "purple"), lwd = 2)

# ----------------------------------- 2.3 -------------------------------------

# ------------------ (setup) --------------------

data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/banknoteFraud.csv", header=FALSE, sep=",")
names(data) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
data[,5] <- as.factor(data[,5])

set.seed(111); SelectTraining <- sample(1:dim(data)[1], size = 1000, replace = FALSE)
train <- data[SelectTraining, ]
test <- data[-SelectTraining, ]

# ------------------ (1) --------------------

# GP classification with default kernel (rbfdot) and hyperparameters
GPfitFraud <- gausspr(fraud ~ varWave + skewWave, data = train)
GPfitFraud

# Grid over varWave and skewWave
x1 <- seq(min(train$varWave), max(train$varWave), length = 100)
x2 <- seq(min(train$skewWave), max(train$skewWave), length = 100)
gridPoints <- expand.grid(varWave = x1, skewWave = x2) # x1 varies fastest

# Probability of fraud = 1 over the grid
probPreds <- predict(GPfitFraud, gridPoints, type = "probabilities")
contour(x1, x2, matrix(probPreds[, "1"], 100, 100), 20,
        xlab = "varWave", ylab = "skewWave", main = "Prob(fraud = 1)")
points(train$varWave[train$fraud == 1], train$skewWave[train$fraud == 1], col = "blue")
points(train$varWave[train$fraud == 0], train$skewWave[train$fraud == 0], col = "red")

# Confusion matrix and accuracy on the training data
predTrain <- predict(GPfitFraud, train[, 1:2])
confTrain <- table(predTrain, train$fraud)
confTrain
sum(diag(confTrain)) / sum(confTrain)

# ------------------ (2) --------------------

# Same model on the test set
predTest <- predict(GPfitFraud, test[, 1:2])
confTest <- table(predTest, test$fraud)
confTest
sum(diag(confTest)) / sum(confTest)

# ------------------ (3) --------------------

# Model with all four covariates
GPfitFraudAll <- gausspr(fraud ~ ., data = train)
GPfitFraudAll

predTestAll <- predict(GPfitFraudAll, test[, 1:4])
confTestAll <- table(predTestAll, test$fraud)
confTestAll
sum(diag(confTestAll)) / sum(confTestAll)
