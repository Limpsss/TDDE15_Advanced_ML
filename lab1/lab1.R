library(bnlearn)

data("asia")

# --------------------------------- 1 ---------------------------------

# HC with different score
dag1 <- hc(asia, score="bde")
dag2 <- hc(asia, score="aic")
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))
score(dag1, asia, type="bde") == score(dag2, asia, type="bde")

# HC with different imaginary sample size
dag1 <- hc(asia, score="bde", iss=5)
dag2 <- hc(asia, score="bde", iss=10)
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))
score(dag1, asia, type="bde") == score(dag2, asia, type="bde")

# HC with random initial structure
set.seed(12345)
dag1 = hc(asia, score="bde", start=random.graph(colnames(asia)))
dag2 = hc(asia, score="bde", start=random.graph(colnames(asia)))
graphviz.plot(dag1)
graphviz.plot(dag2)
all.equal(cpdag(dag1), cpdag(dag2))
score(dag1, asia, type="bde") == score(dag2, asia, type="bde")

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

hc_dag <- hc(train, score="bde", restart=20)
true_dag <- model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
graphviz.plot(hc_dag)
graphviz.plot(true_dag)

hc_dag_fit <- bn.fit(hc_dag, data=train, method="mle")
true_dag_fit <- bn.fit(hc_dag, data=train, method="mle")

# inference here:



