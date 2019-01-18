# Options -----------------------------------------------------------------

library("SemiCompRisks")
library("scales")

set.seed(42)



# Data simulation ---------------------------------------------------------

# Data generation parameters
n <- 1500
beta1.true <- c(0.1,0.1)
beta2.true <- c(0.2, 0.1)
beta3.true <- c(0.5, 0.5)
alpha1.true <- 0.12
alpha2.true <- 0.23
alpha3.true <- 0.34
kappa1.true <- 0.33
kappa2.true <- 0.11
kappa3.true <- 0.22

# Make design matrix with single binary covariate
x_c <- rbinom(n, size = 1, prob = 0.7)
x_m <- cbind(1, x_c)

# Generate semicompeting data
dat_ID <- SemiCompRisks::simID(x1 = x_m, x2 = x_m, x3 = x_m,
                beta1.true = beta1.true, 
                beta2.true = beta2.true, 
                beta3.true = beta3.true,
                alpha1.true = alpha1.true, 
                alpha2.true = alpha2.true, 
                alpha3.true = alpha3.true,
                kappa1.true = kappa1.true, 
                kappa2.true = kappa2.true, 
                kappa3.true = kappa3.true,
                theta.true = 0,
                cens = c(240, 360))
dat_ID$x_c <- x_c
dat_ID$ydiff <- dat_ID$y2 - dat_ID$y1

formula01 <- Surv(time=R,event=delta_R)~x_c
formula02 <- Surv(time=tt,event=delta_T)~x_c
formula12 <- Surv(time=difft,event=delta_T)~x_c

# Plot --------------------------------------------------------------------

par(mfrow = c(1,2))
plot(survfit(Surv(R, delta_R) ~ x_c, data = dat_ID), mark.time = TRUE,
     col = hue_pal()(2), main = "Non-terminal", ylim = c(0, 1))
plot(survfit(Surv(tt, delta_T) ~ x_c, data = dat_ID), mark.time = TRUE,
     col = hue_pal()(2), main = "Terminal", ylim = c(0, 1))
par(mfrow = c(1,1))


# Data saving -------------------------------------------------------------

saveRDS(dat_ID, file = "data-raw/dat_ID.Rdata")



require("gems")
hf <- generateHazardMatrix(3)
hf[[1, 2]] <- "Weibull"
hf[[1, 3]] <- "Weibull"
hf[[2, 3]] <- "Weibull"
print(hf)

par <- generateParameterMatrix(hf)

par[[1, 2]] <- list(shape = 3, scale = 3)
par[[1, 3]] <- list(shape = 2, scale = 2)
par[[2, 3]] <- list(shape = 1, scale = 1)
print(par)

cohortSize <- 10000
cohort <- simulateCohort(transitionFunctions = hf,
                            parameters = par,
                            cohortSize = cohortSize,
                            to = 10)
head(cohort)
post <- transitionProbabilities(cohort, times = seq(0,5, .1))
cinc <- cumulativeIncidence(cohort, times = seq(0,5, .1))


head(post)
head(cinc)

plot(post, main = "Transition probabilities", ci = TRUE)

plot(cinc, main = "Cumulative incidence", ci = TRUE)
