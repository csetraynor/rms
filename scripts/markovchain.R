library("markovchain")

## Function to create a simplex
runif_simplex <- function(T) {
  x <- -log(runif(T))
  x / sum(x)
}

clinicalStates <- c("healthy", "recurrent", "dead")
clinicalMatrix <- matrix(c(runif_simplex(K), c(0, runif_simplex(K-1)), c(0,0, 1) ), nrow = 3, byrow = TRUE)
mcClinical <- new("markovchain", states = clinicalStates,
                 byrow = TRUE, transitionMatrix = clinicalMatrix,
                 name = "Clinical")

plot(mcClinical, main="Clinical Markov Chain")

initialState <- c(1, 0, 0)
#multiplication
after2Days <- initialState * (mcClinical * mcClinical)
after2Days
mcClinical^2


as(mcClinical, "data.frame")
transientStates(mcClinical)
absorbingStates(mcClinical)

resultsOfDay <- rmarkovchain(n = 50, object = mcClinical, t0 = "healthy")
resultsOfDay

McFitMle <- markovchainFit(data = resultsOfDay,method = "bootstrap", name = "Clinical Markov Chain")
McFitMle


markovchainFit(data = rain$rain, method = "mle", name = "Alofi")

data(rain)
rain$rain[1:10]
createSequenceMatrix(stringchar = rain$rain)


library("msSurv")
data("RCdata")
RCdata[70:76, ]
Nodes <- c("1", "2", "3", "4", "5")
Edges <- list("1" = list(edges = c("2", "3")),
               "2" = list(edges = c("4", "5")),
               "3" = list(edges = NULL),
               "4" = list(edges = NULL),
               "5" = list(edges = NULL))

treeobj <- new("graphNEL", nodes = Nodes, edgeL = Edges,
               edgemode = "directed")

ex1 <- msSurv(RCdata, treeobj, bs = TRUE)

Pst(ex1, s = 1, t = 3.1)
Pst(ex1, s = 1, t = 3.1, covar = TRUE)
SOPt(ex1, t = 0.85)
SOPt(ex1, t = 0.85, covar = TRUE)

data("LTRCdata")

Nodes <- c("1", "2", "3")
Edges <- list("1" = list(edges = c("2", "3")),
                 "2" = list(edges = c("3")),
                 "3" = list(edges = NULL))
LTRCtree <- new("graphNEL", nodes = Nodes, edgeL = Edges,
                   edgemode = "directed")
ex2 <- msSurv(LTRCdata, LTRCtree, LT = TRUE)
summary(ex2)


if(exists("data.sdc")) head(data.sdc)
data(data.sdc)
