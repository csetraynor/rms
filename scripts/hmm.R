set.seed(900)
K        <- 3
T_length <- 500
dataset  <- hmm_generate(K, T_length)
fit      <- hmm_fit(K, dataset$y)