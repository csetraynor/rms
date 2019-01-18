#'   # Generate times from user-defined log hazard function:
   
set.seed(9911)
covs <- data.frame(id = 1:1000, trt = stats::rbinom(1000, 1L, 0.5))

## Simulate terminal time
fn_t <- function(t, x, betas, ...){
  (-1 + 0.02 * t - 0.03 * t ^ 2 + 0.005 * t ^ 3 + betas * x)
}

s4 <- simsurv(loghazard = fn_t, x = covs, maxt = 1.5, betas = c(trt = -0.5))
s4$trt <- covs$trt

## Simulate non-terminal time
fn_t <- function(t, x, betas, ...)
  (-2 + 0.04 * t - 0.01 * t ^ 2 + 0.007 * t ^ 3)

 # Generate times from a Weibull model including a binary
  # treatment variable, with log(hazard ratio) = -0.5, and censoring  # after 5 years:

# simulate terminal event
beta01 <- 1
lambdas01 <- 0.1
gammas01 <- 1.5
beta02 <- 2
lambdas02 <- 0.15
gammas02 <- 1
beta12 <- 3
lambdas12 <- 0.75
gammas12 <- 1.25

set.seed(9911)





covs <- data.frame(id = 1:1000, trt = stats::rbinom(1000, 1L, 0.5))


rsimms <- function(lambdas01, gammas01, beta01 = 0, covs, maxt = 50, lambdas02, gammas02, beta02 = 0, lambdas12, gammas12, beta12 = 0){
  s1 <- simsurv(lambdas = lambdas01, gammas = gammas01,
                x = covs, maxt = maxt, betas = beta01)
  head(s1)
  
  colnames(s1)[2:3] <- c("yr", "dr")
  
  # simulate non-terminal event
  s2 <- simsurv(lambdas = lambdas02, gammas = gammas02,
                x = covs, maxt = maxt, betas = beta02)
  head(s2)
  colnames(s2)[2:3] <- c("yt", "dt")
  # s2 and s1
  s <- dplyr::left_join(s1, s2)
  
  s$ostime <- with(s, pmin(yt, yr))
  s$dftime <- with(s, pmin(yt, yr))
  s$osevent <- s$dt
  s$osevent[s$ostime == s$yr] <- 0
  s$dfevent <- s$dr
  s$dfevent[s$dftime == s$yt] <- 0
  
  
  covs3 <- covs[s$dfevent == 1, ]
  
  s3 <- simsurv(lambdas = lambdas12, gammas = gammas12,
                x = covs3, maxt = maxt, betas = beta12)
  head(s3)
  colnames(s2)[2:3] <- c("yt", "dt")
  s$ostime[s$dfevent == 1] <- s3$eventtime + s$ostime[s$dfevent == 1]
  s$osevent[s$dfevent == 1] <- s3$status
  
  s$trt <- covs$trt
  ms_data$timediff <- ms_data$ostime - ms_data$dftime 
  return(ms_data)
}
  





saveRDS(s, "data-raw/ms_data.RDS")
