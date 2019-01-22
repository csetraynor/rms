compdata <- sas7bdat::read.sas7bdat("/media/mtr/A5C2-009E/SCL/c9732_demographic.sas7bdat")

compdata$STATUS[compdata$STATUS == 1] <- 0
compdata$STATUS[compdata$STATUS == 3] <- 1
compdata$STATUS[compdata$STATUS == 2] <- 1


compdata$TIMEDIFF <- compdata$OS_TIME - compdata$PFS_TIME
unique(compdata$TRT_ARM_LABEL)

compdata <- compdata[compdata$OS_TIME > 0, ]

compdata$PFS_STATUS[(compdata$STATUS == 1) & compdata$TIMEDIFF == 0] <- 0

compdata$TIMEDIFF[(compdata$PFS_STATUS == 1) & (compdata$TIMEDIFF == 0)] <- 1e-5

saveRDS(compdata, "data-raw/sclc.RDS")

compdata <- readRDS("data-raw/sclc.RDS")

standata <- idm_stan(
  formula01 = Surv(time=PFS_TIME,event=PFS_STATUS)~TRT_ARM_LABEL,
  formula02 = Surv(time=OS_TIME,event=STATUS)~1,
  formula12 = Surv(time=TIMEDIFF,event=STATUS)~1,
  data = compdata,
  basehaz01 = "ms",
  basehaz02 = "exp",
  basehaz12 = "exp",
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

stanfile <-  "src/stan_files/simple_MS.stan"
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
