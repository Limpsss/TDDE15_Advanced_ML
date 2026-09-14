library(bnlearn)

data("asia")

# --------------------------------- 1 ---------------------------------

# Learn structure using HC with different score
dag1 <- hc(asia, score="bde")
dag2 <- hc(asia, score="aic")
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))

# Learn structure using HC with different imaginary sample size
dag1 <- hc(asia, score="bde", iss=5)
dag2 <- hc(asia, score="bde", iss=10)
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))

# Learn structure using HC with random initial structure
set.seed(12345)
dag1 = hc(asia, score="bde", start=random.graph(colnames(asia)))
dag2 = hc(asia, score="bde", start=random.graph(colnames(asia)))
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))

# Using different score methods, imaginary sample size, or initial structures
# yields in non-equivalent Bayesian networks, as shown by the plots and by the
# all.equal comparison of the DAG's. This is because hill climbing is a greedy 
# algorithm which can and often will get stuck in local optima.

# --------------------------------- 2 ---------------------------------
library(gRain)

n <- dim(asia)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.8))
train <- asia[id, ]
test <- asia[-id, ]

hc_dag <- hc(train, score="bde", iss=5)
true_dag <- model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
graphviz.plot(hc_dag)
graphviz.plot(true_dag)

# Fit parameters using maximum likelihood estimation
hc_dag_fit <- bn.fit(hc_dag, data=train, method="mle")
true_dag_fit <- bn.fit(true_dag, data=train, method="mle")

# Convert both fitted networks to gRain objects for exact inference
hc_grain <- as.grain(hc_dag_fit)
true_grain <- as.grain(true_dag_fit)

# Classifies the test set: S = "yes" if P(S = yes | evidence) > 0.5, else "no"
classifier <- function(model, observed_nodes) {
  preds <- numeric(nrow(test))
  for(i in 1:nrow(test)) {
    # Get the states of the current row
    states = as.matrix(test[i, observed_nodes])
    # Set the evidence of current row
    evidence <- setEvidence(model, nodes=observed_nodes, states=states)
    # Query posterior probability distribution of node S
    pred <- querygrain(evidence, nodes="S")$S
    preds[i] <- ifelse(pred["yes"] > 0.5, "yes", "no")
  }
  return(preds)
}

# Use all nodes except S (column 2) as evidence
nodes <- colnames(test[-2])

# Classify with the network learned by hill climbing
hc_model_preds <- classifier(hc_grain, nodes)
table(test$S, hc_model_preds)

# Classify with the true network
true_model_preds <- classifier(true_grain, nodes)
table(test$S, true_model_preds)

# --------------------------------- 3 ---------------------------------

# Restrict the evidence to the Markov blanket of S only
hc_mb <- mb(hc_dag_fit, node="S")
true_mb <- mb(true_dag_fit, node="S")

# Classify using the Markov blanket as observed nodes
hc_mb_preds <- classifier(hc_grain, hc_mb)
table(test$S, hc_mb_preds)

true_mb_preds <- classifier(true_grain, true_mb)
table(test$S, true_mb_preds)

# --------------------------------- 4 ---------------------------------

# Naive Bayes classifier assumes that all feature variables are conditionally 
# independent given the class variable S.
# This can be represented as a Bayesian network where the class node S is the
# single parent of all feature nodes and no edges exist between features.
nbc_network <- model2network("[S][A|S][T|S][L|S][B|S][E|S][X|S][D|S]")
graphviz.plot(nbc_network)

# Fit the network and convert to a gRain model.
nbc_fit <- bn.fit(nbc_network, data=train, method="mle")
nbc_grain <- as.grain(nbc_fit)

# Classify the test set with the naive Bayes classifier.
nbc_preds <- classifier(nbc_grain, nodes)
table(test$S, nbc_preds)

# --------------------------------- 5 ---------------------------------

# Naive Bayes enforces a strict assumption that all features are conditionally 
# independent given the target node S, ignoring dependencies among the features
# which leads to worse predictive performance

