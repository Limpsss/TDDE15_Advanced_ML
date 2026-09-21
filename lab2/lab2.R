library(HMM)

# ----------------------------------- PART 1 ----------------------------------

states <- 1:10
symbols <- 1:10


trans_probs <- matrix(nrow = 10, ncol = 10)
for(i in 1:10) {
  # Set probability to stay in current sector to 0.5
  trans_probs[i, i] <- 0.5
  # Set probability to move to next sector to 0.5
  trans_probs[i, if (i == 10) 1 else i + 1] <- 0.5
}

emis_probs <- matrix(0, nrow = 10, ncol = 10)
# Observe any of the 5 sectors [i-2, i+2] with equal probability
# Since it's a ring, the window wraps around
for(i in 1:10) {
  for(d in -2:2) {
    j <- ((i + d - 1) %% 10) + 1  # wrap-around for the ring
    emis_probs[i, j] <- 0.2
  }
}

# Equal probability to start at any sector
start_probs <- rep(0.1, 10)

hmm <- initHMM(
  States = states,
  Symbols = symbols,
  startProbs = start_probs,
  transProbs = trans_probs,
  emissionProbs = emis_probs
)

# ----------------------------------- PART 2 ----------------------------------

set.seed(12345)
sim <- simHMM(hmm, 100)
print(sim)

# ----------------------------------- PART 3 ----------------------------------

observations <- sim$observation

log_forward_probs <- forward(hmm, observations)
log_backward_probs <- backward(hmm, observations)

# Filtering: P(x_t | y_1:t)
filtered_probs <- exp(log_forward_probs)
# Normalize each time points distribution to sum to 1
filtered_probs <- sweep(filtered_probs, 2, colSums(filtered_probs), "/")

# Smoothing: P(x_t | y_1:100)
smoothed_probs <- exp(log_forward_probs + log_backward_probs)
# Normalize each time points distribution to sum to 1
smoothed_probs <- sweep(smoothed_probs, 2, colSums(smoothed_probs), "/")

# Most probable path using the viterbi algorithm
viterbi_path <- viterbi(hmm, observations)
print(viterbi_path)

# ----------------------------------- PART 4 ----------------------------------

accuracy <- function(y, y_hat) {
  return( mean(y == y_hat) )
}

# Get the most probable state at each time step
filtered_preds <- apply(filtered_probs, 2, which.max)
smoothed_preds <- apply(smoothed_probs, 2, which.max)

# Actual states
true_states <- sim$states

# Calculate accuracy
filtered_accuracy <- accuracy(true_states, filtered_preds)
smoothed_accuracy <- accuracy(true_states, smoothed_preds)
viterbi_accuracy <- accuracy(true_states, viterbi_path)

# ----------------------------------- PART 5 ----------------------------------





