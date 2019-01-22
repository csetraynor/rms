if (!require(biostan))
  devtools::install_github('jburos/biostan')
library(biostan)

stan_file <- system.file('stan', 'weibull_survival_null_model.stan', package =  'biostan')

stan_file <- "src/stan_files/simple_MS.stan"
print_stan_file(stan_file)


stanfit <- stanmodels$simple_MS
print(stanfit)
