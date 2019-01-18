set.seed(13579)
covs <- data.frame(id = 1:1000, trt = stats::rbinom(1000, 1L, 0.5))
s01 <- simsurv(lambdas = 0.1,
               gammas = 1.5,
              x = covs,
              betas = c(trt = 10),
              maxt = 10)
s02 <- simsurv(lambdas = 0.1,
               gammas = 1.5,
               x = covs,
               betas = c(trt = -10),
               maxt = 10)
s03 <- simsurv(lambdas = 0.1,
               gammas = 1.5,
               x = covs, 
               betas = c(trt = 1),
               maxt = 10)


y <- data.frame( id = s01$id,
                 dftime = s01$eventtime,
                 dfevent = s01$status,
                 ost = s02$eventtime,
                 osevent = s02$status,
                 dfost = s03$eventtime + s01$eventtime,
                 dfose = s03$status )

y$ostime <- with(y, pmin(dfost, ost))

y$dfevent[y$ostime < y$dftime] <- 0

y$osevent[y$dfost < y$ostime] <- y$dfose[y$dfost < y$ostime]

y$difftime <- s03$eventtime


y$trt <- covs$trt

# y


standata <- idm_stan(formula01 = Surv(time=dftime,event=dfevent)~1+trt,
                     formula02 = Surv(time=ostime,event=osevent)~1+trt,
                     formula12 = Surv(time=difftime,event=osevent)~1+trt,
                     data = y,
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
                     prior_aux12       = rstanarm::normal())
standata$nevent01
standata$nevent12
