source("R/helpers-data.R")
dat_ID <- readRDS("data-raw/dat_ID.Rdata")
library(dplyr)


library(lava)
library(prodlim)
set.seed(17)
d <- SmoothHazard::simulateIDM(100)
# right censored data
fitRC <- SmoothHazard::idm(formula01=Hist(time=observed.illtime,event=seen.ill)~X1+X2,
             formula02=Hist(time=observed.lifetime,event=seen.exit)~X1+X2,
             formula12=Hist(time=observed.lifetime,event=seen.exit)~X1+X2,data=d,
             conf.int=FALSE)
fitRC

d <- SmoothHazard::simulateIDM(300)
fitRC.splines <- SmoothHazard::idm(formula01=Hist(time=observed.illtime,event=seen.ill)~X1+X2,
                                   formula02=Hist(time=observed.lifetime,event=seen.exit)~X1+X2,
                                   formula12=Hist(time=observed.lifetime,event=seen.exit)~1,data=d,
                                   conf.int=FALSE,method="splines")
fitRC.splines



d$delta_R = d$seen.ill
d$delta_T = d$seen.exit
d$R = d$observed.illtime + 10


standata <- idm_stan(X = d,
                     K1 = 2:3,
                     K2 = 2:3,
                     K3 = 2:3)

dat_ID$tt = dat_ID$T





formula01 <- Surv(time=R,event=delta_R)~x_c
formula02 <- Surv(time=tt,event=delta_T)~x_c
formula12 <- Surv(time=difft,event=delta_T)~x_c



# Data simulation ---------------------------------------------------------

# Data generation parameters
library(SemiCompRisks)
set.seed(42)

n <- 2000
beta1.true <- c(-0.1, -0.1)
beta2.true <- c(-0.15, -0.15)
beta3.true <- c(-0.2, -0.2)
alpha1.true <- 0.12
alpha2.true <- 0.23
alpha3.true <- 0.34
kappa1.true <- 0.33
kappa2.true <- 0.11
kappa3.true <- 0.22
theta.true <- 0

# Make design matrix with single binary covariate
x_c <- rbinom(n, size = 1, prob = 0.7)
x_m <- cbind(1, x_c)

# Generate semicompeting data
dat_ID <- simID(x1 = x_m, x2 = x_m, x3 = x_m,
                beta1.true = beta1.true,
                beta2.true = beta2.true,
                beta3.true = beta3.true,
                alpha1.true = alpha1.true,
                alpha2.true = alpha2.true,
                alpha3.true = alpha3.true,
                kappa1.true = kappa1.true,
                kappa2.true = kappa2.true,
                kappa3.true = kappa3.true,
                theta.true = theta.true,
                cens = c(240, 360))
dat_ID$x_c <- x_c
colnames(dat_ID)[1:4] <- c("R", "delta_R", "tt", "delta_T")

#dat_ID <- readRDS("data-raw/dat_ID.Rdata")



dat_ID$ydiff <- 0
dat_ID$ydiff[(dat_ID$tt - dat_ID$R) > 0] <- dat_ID$tt[(dat_ID$tt - dat_ID$R) > 0] - dat_ID$R[(dat_ID$tt - dat_ID$R) > 0]

#
data = dat_ID
formula01 <- Surv(time=R,event=delta_R)~x_c
formula02 <- Surv(time=tt,event=delta_T)~x_c
formula12 <- Surv(time=ydiff,event=delta_T)~x_c


standata <- idm_stan(formula01 = Surv(time=R,event=delta_R)~x_c,
                     formula02 = Surv(time=tt,event=delta_T)~x_c,
                     formula12 = Surv(time=ydiff,event=delta_T)~x_c,
                     data = dat_ID,
                     basehaz01 = "weibull",
                     basehaz02 = "weibull",
                     basehaz12 = "weibull",
                     prior01           = rstanarm::normal(),
                     prior_intercept01 = rstanarm::normal(),
                     prior_aux01       = rstanarm::normal(),
                     prior02           = rstanarm::normal(),
                     prior_intercept02 = rstanarm::normal(),
                     prior_aux02       = rstanarm::normal(),
                     prior12           = rstanarm::normal(),
                     prior_intercept12 = rstanarm::normal(),
                     prior_aux12       = rstanarm::normal()
)


stanpars <- c(if (standata$has_intercept01) "alpha01",
              if (standata$K01)             "beta01",
              if (standata$nvars01)         "aux01",
              if (standata$has_intercept02) "alpha02",
              if (standata$K02)             "beta02",
              if (standata$nvars02)         "aux02",
              if (standata$has_intercept12) "alpha12",
              if (standata$K12)             "beta12",
              if (standata$nvars12)         "aux12")


gen_inits <- function() {
  list(
    beta01 = rnorm(1),
    beta02 = rnorm(1),
    beta12 = rnorm(1),
    aux01 = rnorm(1),
    aux02 = rnorm(1),
    aux12 = rnorm(1),
    alpha01 = log(abs(rnorm(1))),
    alpha02 = log(abs(rnorm(1))),
    alpha12 = log(abs(rnorm(1)))
  )
}


stanfile <-  "src/stan_files/MS.stan"
library(rstan)
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')
options(mc.cores = 2)
fit <- stan(file = stanfile,
            data = standata,
            pars = stanpars,
            iter = 2000,
            chains = 4,
            control = list(adapt_delta = 0.99))
s_elapsed <- sum(get_elapsed_time(fit))



print(fit)
rstan::traceplot(fit, 'lp__')

rstan::traceplot(fit, c('beta01','beta02', 'beta12'), ncol = 1)

#### Simulated data from own function


ms_data <- readRDS("data-raw/ms_data.RDS")



#res = log(shape) + (shape - 1) * log(t) + eta;
alpha1.true <- 0.12
alpha2.true <- 0.23
alpha3.true <- 0.34
kappa1.true <- 0.33
kappa2.true <- 0.11
kappa3.true <- 0.22
set.seed(9911)
covs <- data.frame(id = 1:1000, trt = stats::rbinom(1000, 1L, 0.5))
ms_data <- rsimms(
  beta01 = c(trt = -0.1), lambdas01 = 0.12, gammas01 = 0.33,
  covs = covs,
  maxt = 50,
  beta02 = c(trt = -0.2), lambdas02 = 0.23, gammas02 = 0.11,
  beta12 = c(trt = -0.3), lambdas12 = 0.34, gammas12 = 0.22)

ms_data$dfevent[(ms_data$STATUS == 1) & ms_data$timediff == 0] <- 0

ms_data$timediff[(ms_data$dfevent == 1) & (ms_data$timediff == 0)] <- 1e-5
ms_data <- s
saveRDS(s, "data-raw/ms_simdata.RDS")
ms_data <- s

standata <- idm_stan(formula01 = Surv(time=dftime,event=dfevent)~trt,
                     formula02 = Surv(time=ostime,event=osevent)~trt,
                     formula12 = Surv(time=timediff,event=osevent)~trt,
                     data = ms_data,
                     basehaz01 = "weibull",
                     basehaz02 = "weibull",
                     basehaz12 = "weibull",
                     prior01           = rstanarm::normal(),
                     prior_intercept01 = rstanarm::normal(),
                     prior_aux01       = rstanarm::normal(),
                     prior02           = rstanarm::normal(),
                     prior_intercept02 = rstanarm::normal(),
                     prior_aux02       = rstanarm::normal(),
                     prior12           = rstanarm::normal(),
                     prior_intercept12 = rstanarm::normal(),
                     prior_aux12       = rstanarm::normal()
                     )
standata$nevent01
standata$nevent12

stanpars <- c(if (standata$has_intercept01) "alpha01",
              if (standata$K01)             "beta01",
              if (standata$nvars01)         "aux01",
              if (standata$has_intercept02) "alpha02",
              if (standata$K02)             "beta02",
              if (standata$nvars02)         "aux02",
              if (standata$has_intercept12) "alpha12",
              if (standata$K12)             "beta12",
              if (standata$nvars12)         "aux12")

stanfile <-  "src/stan_files/MS.stan"
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = 2)
fit <- stan(file = stanfile,
            data = standata,
            pars = stanpars,
            iter = 2000,
            chains = 4,
            control = list(adapt_delta = 0.95))
s_elapsed <- sum(get_elapsed_time(fit))


print(fit)
rstan::traceplot(fit, 'lp__')

rstan::traceplot(fit, c('beta01','beta02', 'beta12'), ncol = 1)



pp_alpha <- exp(rstan::extract(fit,'alpha01')$alpha01)
pp_mu <- rstan::extract(fit,'aux01')$aux01
pp_beta <- rstan::extract(fit,'beta01')$beta01

ggplot(data.frame(alpha = pp_alpha, mu = pp_mu)) +
  geom_density2d(aes(x = alpha, y = mu)) +
  geom_point(aes(x = kappa1.true, y = alpha1.true), colour = 'red', size = 2) +
  ggtitle('Posterior distributions of mu and alpha\nshowing true parameter values in red')


ggplot(data.frame(beta = pp_beta)) +
  geom_density(aes(x = beta)) +
  geom_vline(aes(xintercept = beta1.true[1]), colour = 'red') +
  ggtitle('Posterior distribution of mu\nshowing true value in red')


mean(pp_alpha >= kappa1.true)
mean(pp_mu >= alpha1.true)


mean(pp_mu >= alpha1.true & pp_alpha >= kappa1.true)
