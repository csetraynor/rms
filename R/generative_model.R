
## Function to create a simplex
runif_simplex <- function(T) {
  x <- -log(runif(T))
  x / sum(x)
}


## Function to generate HMM
hmm_generate <- function(K, T) {
  # 1. Parameters
  pi1 <- c(1, 0, 0) # initial state probabilities
  A <- matrix(c(runif_simplex(K), c(0,1,0), c(0, runif_simplex(K-1))), nrow = 3, byrow = TRUE) # transition probabilities
  
  mu <- sort(rnorm(K, 10 * 1:K, 1)) #observation means
  sigma <- abs(rnorm(K)) # observation standard deviations
  
  # 2. Hidden path
  z <- vector("numeric", T)
  z[1] <- sample(1:K, size = 1, prob = pi1)
  for (t in 2:T)
    z[t] <- sample(1:K, size = 1, prob = A[z[t - 1], ])
  
  # 3. Observations
  y <- vector("numeric", T)
  for (t in 1:T)
    y[t] <- rnorm(1, mu[z[t]], sigma[z[t]])
  list(y = y, z = z,
       theta = list(pi1 = pi1, A = A,
                    mu = mu, sigma = sigma))
}

## Function to initiate HMM
hmm_init <- function(K, y) {
  clasif <- kmeans(y, K)
  init.mu <- by(y, clasif$cluster, mean)
  init.sigma <- by(y, clasif$cluster, sd)
  init.order <- order(init.mu)
  
  list(
    mu = init.mu[init.order],
    sigma = init.sigma[init.order]
  )
}

## Function to fit HMM
hmm_fit <- function(K, y) {
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
  stan.model = 'stan/hmm_gaussian.stan'
  stan.data = list(
    T = length(y),
    K = K,
    y = y
  )
  
  stan(file = stan.model,
       data = stan.data, verbose = T,
       iter = 400, warmup = 200,
       thin = 1, chains = 1,
       cores = 1, seed = 900,
       init = function(){hmm_init(K, y)})
}

