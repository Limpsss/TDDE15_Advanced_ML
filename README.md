# TDDE15 Advanced Machine Learning

Lab work for [TDDE15 Advanced Machine Learning](https://studieinfo.liu.se/en/kurs/TDDE15) at Linköping University, autumn 2026. All labs are written in R and compiled to PDF reports with R Markdown.

## Labs

### Lab 1 – Bayesian Networks
Structure learning with hill climbing on the Asia dataset, showing how the score, imaginary sample size and random restarts lead to non-equivalent DAGs. The learned network is then used to classify *S* (smoker), compared against the true network, a Markov-blanket-only classifier and naive Bayes.

`bnlearn` · `gRain` — [report](lab1/lab1.pdf) · [source](lab1/lab1.Rmd) · [instructions](lab1/lab1_instructions.pdf)

### Lab 2 – Hidden Markov Models
A robot moving around a 10-sector ring, modelled as an HMM. The lab covers simulation, filtering, smoothing and the Viterbi path, compares their accuracy over many simulations, and studies the entropy of the filtered distributions.

`HMM` · `entropy` — [report](lab2/lab2.pdf) · [source](lab2/lab2.Rmd) · [instructions](lab2/lab2_instructions.pdf)

### Lab 3 – Reinforcement Learning
Tabular Q-learning in grid-world environments, studying the effect of exploration (ε), discount (γ) and slipperiness (β). Policy-gradient learning with REINFORCE is in the accompanying [Colab notebook](lab3/RL_Lab3_Colab.ipynb).

`ggplot2` — [report](lab3/lab3.pdf) · [source](lab3/lab3.Rmd) · [instructions](lab3/lab3_instructions.pdf)

### Lab 4 – Gaussian Processes
GP regression implemented from scratch (Algorithm 2.1 in Rasmussen & Williams), then applied with `kernlab` to daily temperatures in Tullinge using squared-exponential and locally periodic kernels. The lab ends with GP classification of banknote fraud.

`kernlab` — [report](lab4/lab4.pdf) · [source](lab4/lab4.Rmd) · [instructions](lab4/lab4_instructions.pdf)

## Repository layout

```
labN/
├── labN.Rmd               report source
├── labN.R                 working script
├── labN.pdf               rendered report
└── labN_instructions.pdf  lab assignment
```

## How to run

Install the CRAN packages:

```r
install.packages(c("bnlearn", "gRain", "HMM", "entropy", "ggplot2",
                   "kernlab", "mvtnorm", "rmarkdown", "BiocManager"))
BiocManager::install(c("Rgraphviz", "RBGL", "gRbase"))  # used by gRain and graphviz.plot
```

Then knit a report, either by opening `labN/labN.Rproj` in RStudio or from R:

```r
rmarkdown::render("lab1/lab1.Rmd")
```

PDF output requires a LaTeX installation (e.g. `tinytex::install_tinytex()`).

## Credits

Lab instructions and datasets are from the course. The Q-learning scaffold in lab 3 is by Jose M. Peña and Joel Oskarsson, and `lab4/GaussianProcesses.r` and `lab4/KernLabDemo.r` are by Mattias Villani.
