#' Bayesian multi-state models via Stan
#'
#' \if{html}{\figure{stanlogo.png}{options: width="25px" alt="http://mc-stan.org/about/logo/"}}
#' Bayesian inference for multi-state models (sometimes known as models for
#' competing risk data). Currently, the command fits standard parametric
#' (exponential, Weibull and Gompertz) and flexible parametric (cubic
#' spline-based) hazard models on the hazard scale, with covariates included
#' under assumptions of either proportional or non-proportional hazards.
#' Where relevant, non-proportional hazards are modelled using a flexible
#' cubic spline-based function for the time-dependent effect (i.e. the
#' time-dependent hazard ratio).
#'
#' @export
#' @importFrom splines bs
#' @import splines2
#'
#' @param formula A two-sided formula object describing the model.
#'   The left hand side of the formula should be a \code{Surv()}
#'   object. Left censored, right censored, and interval censored data
#'   are allowed, as well as delayed entry (i.e. left truncation). See
#'   \code{\link[survival]{Surv}} for how to specify these outcome types.
#'   If you wish to include time-dependent effects (i.e. time-dependent
#'   coefficients, also known as non-proportional hazards) in the model
#'   then any covariate(s) that you wish to estimate a time-dependent
#'   coefficient for should be specified as \code{tde(varname)}  where
#'   \code{varname} is the name of the covariate. See the \strong{Details}
#'   section for more information on how the time-dependent effects are
#'   formulated, as well as the \strong{Examples} section.
#' @param data A data frame containing the variables specified in
#'   \code{formula}.
#' @param basehaz A character string indicating which baseline hazard to use
#'   for the event submodel. Current options are:
#'   \itemize{
#'     \item \code{"ms"}: a flexible parametric model using cubic M-splines to
#'     model the baseline hazard. The default locations for the internal knots,
#'     as well as the basis terms for the splines, are calculated with respect
#'     to time. If the model does \emph{not} include any time-dependendent
#'     effects then a closed form solution is available for both the hazard
#'     and cumulative hazard and so this approach should be relatively fast.
#'     On the other hand, if the model does include time-dependent effects then
#'     quadrature is used to evaluate the cumulative hazard at each MCMC
#'     iteration and, therefore, estimation of the model will be slower.
#'     \item \code{"bs"}: a flexible parametric model using cubic B-splines to
#'     model the \emph{log} baseline hazard. The default locations for the
#'     internal knots, as well as the basis terms for the splines, are calculated
#'     with respect to time. A closed form solution for the cumulative hazard
#'     is \strong{not} available regardless of whether or not the model includes
#'     time-dependent effects; instead, quadrature is used to evaluate
#'     the cumulative hazard at each MCMC iteration. Therefore, if your model
#'     does not include any time-dependent effects, then estimation using the
#'     \code{"ms"} baseline hazard will be faster.
#'     \item \code{"exp"}: an exponential distribution for the event times.
#'     (i.e. a constant baseline hazard)
#'     \item \code{"weibull"}: a Weibull distribution for the event times.
#'     \item \code{"gompertz"}: a Gompertz distribution for the event times.
#'   }
#' @param basehaz_ops A named list specifying options related to the baseline
#'   hazard. Currently this can include: \cr
#'   \itemize{
#'     \item \code{df}: a positive integer specifying the degrees of freedom
#'     for the M-splines or B-splines. An intercept is included in the spline
#'     basis and included in the count of the degrees of freedom, such that
#'     two boundary knots and \code{df - 4} internal knots are used to generate
#'     the cubic spline basis. The default is \code{df = 6}; that is, two
#'     boundary knots and two internal knots.
#'     \item \code{knots}: An optional numeric vector specifying internal
#'     knot locations for the M-splines or B-splines. Note that \code{knots}
#'     cannot be specified if \code{df} is specified. If \code{knots} are
#'     \strong{not} specified, then \code{df - 4} internal knots are placed
#'     at equally spaced percentiles of the distribution of uncensored event
#'     times.
#'   }
#'   Note that for the M-splines and B-splines - in addition to any internal
#'   \code{knots} - a lower boundary knot is placed at the earliest entry time
#'   and an upper boundary knot is placed at the latest event or censoring time.
#'   These boundary knot locations are the default and cannot be changed by the
#'   user.
#' @param qnodes The number of nodes to use for the Gauss-Kronrod quadrature
#'   that is used to evaluate the cumulative hazard when \code{basehaz = "bs"}
#'   or when time-dependent effects (i.e. non-proportional hazards) are
#'   specified. Options are 15 (the default), 11 or 7.
#' @param prior_intercept The prior distribution for the intercept. Note
#'   that there will only be an intercept parameter when \code{basehaz} is set
#'   equal to one of the standard parametric distributions, i.e. \code{"exp"},
#'   \code{"weibull"} or \code{"gompertz"}, in which case the intercept
#'   corresponds to the parameter \emph{log(lambda)} as defined in the
#'   \emph{stan_surv: Survival (Time-to-Event) Models} vignette. For the cubic
#'   spline-based baseline hazards there is no intercept parameter since it is
#'   absorbed into the spline basis and, therefore, the prior for the intercept
#'   is effectively specified as part of \code{prior_aux}.
#'
#'   Where relevant, \code{prior_intercept} can be a call to \code{normal},
#'   \code{student_t} or \code{cauchy}. See the \link[=priors]{priors help page}
#'   for details on these functions. Note however that default scale for
#'   \code{prior_intercept} is 20 for \code{stan_surv} models (rather than 10,
#'   which is the default scale used for \code{prior_intercept} by most
#'   \pkg{rstanarm} modelling functions). To omit a prior on the intercept
#'   ---i.e., to use a flat (improper) uniform prior--- \code{prior_intercept}
#'   can be set to \code{NULL}.
#' @param prior_aux The prior distribution for "auxiliary" parameters related to
#'   the baseline hazard. The relevant parameters differ depending
#'   on the type of baseline hazard specified in the \code{basehaz}
#'   argument. The following applies:
#'   \itemize{
#'     \item \code{basehaz = "ms"}: the auxiliary parameters are the coefficients
#'     for the M-spline basis terms on the baseline hazard. These parameters
#'     have a lower bound at zero.
#'     \item \code{basehaz = "bs"}: the auxiliary parameters are the coefficients
#'     for the B-spline basis terms on the log baseline hazard. These parameters
#'     are unbounded.
#'     \item \code{basehaz = "exp"}: there is \strong{no} auxiliary parameter,
#'     since the log scale parameter for the exponential distribution is
#'     incorporated as an intercept in the linear predictor.
#'     \item \code{basehaz = "weibull"}: the auxiliary parameter is the Weibull
#'     shape parameter, while the log scale parameter for the Weibull
#'     distribution is incorporated as an intercept in the linear predictor.
#'     The auxiliary parameter has a lower bound at zero.
#'     \item \code{basehaz = "gompertz"}: the auxiliary parameter is the Gompertz
#'     scale parameter, while the log shape parameter for the Gompertz
#'     distribution is incorporated as an intercept in the linear predictor.
#'     The auxiliary parameter has a lower bound at zero.
#'   }
#'   Currently, \code{prior_aux} can be a call to \code{normal}, \code{student_t}
#'   or \code{cauchy}. See \code{\link{priors}} for details on these functions.
#'   To omit a prior ---i.e., to use a flat (improper) uniform prior--- set
#'   \code{prior_aux} to \code{NULL}.
#' @param prior_smooth This is only relevant when time-dependent effects are
#'   specified in the model (i.e. the \code{tde()} function is used in the
#'   model formula. When that is the case, \code{prior_smooth} determines the
#'   prior distribution given to the hyperparameter (standard deviation)
#'   contained in a random-walk prior for the cubic B-spline coefficients used
#'   to model the time-dependent coefficient. Lower values for the hyperparameter
#'   yield a less a flexible smooth function for the time-dependent coefficient.
#'   Specifically, \code{prior_smooth} can be a call to \code{exponential} to
#'   use an exponential distribution, or \code{normal}, \code{student_t} or
#'   \code{cauchy}, which results in a half-normal, half-t, or half-Cauchy
#'   prior. See \code{\link{priors}} for details on these functions. To omit a
#'   prior ---i.e., to use a flat (improper) uniform prior--- set
#'   \code{prior_smooth} to \code{NULL}. The number of hyperparameters depends
#'   on the model specification (i.e. the number of time-dependent effects
#'   specified in the model) but a scalar prior will be recylced as necessary
#'   to the appropriate length.
#'
#' @details
#' \subsection{Time dependent effects (i.e. non-proportional hazards)}{
#'   By default, any covariate effects specified in the \code{formula} are
#'   included in the model under a proportional hazards assumption. To relax
#'   this assumption, it is possible to estimate a time-dependent coefficient
#'   for a given covariate. This can be specified in the model \code{formula}
#'   by wrapping the covariate name in the \code{tde()} function (note that
#'   this function is not an exported function, rather it is an internal function
#'   that can only be evaluated within the formula of a \code{stan_surv} call).
#'
#'   For example, if we wish to estimate a time-dependent effect for the
#'   covariate \code{sex} then we can specify \code{tde(sex)} in the
#'   \code{formula}, e.g. \code{Surv(time, status) ~ tde(sex) + age + trt}.
#'   The coefficient for \code{sex} will then be modelled
#'   using a flexible smooth function based on a cubic B-spline expansion of
#'   time.
#'
#'   The flexibility of the smooth function can be controlled in two ways:
#'   \itemize{
#'   \item First, through control of the prior distribution for the cubic B-spline
#'   coefficients that are used to model the time-dependent coefficient.
#'   Specifically, one can control the flexibility of the prior through
#'   the hyperparameter (standard deviation) of the random walk prior used
#'   for the B-spline coefficients; see the \code{prior_smooth} argument.
#'   \item Second, one can increase or decrease the number of degrees of
#'   freedom used for the cubic B-spline function that is used to model the
#'   time-dependent coefficient. By default the cubic B-spline basis is
#'   evaluated using 3 degrees of freedom (that is a cubic spline basis with
#'   boundary knots at the limits of the time range, but no internal knots).
#'   If you wish to increase the flexibility of the smooth function by using a
#'   greater number of degrees of freedom, then you can specify this as part
#'   of the \code{tde} function call in the model formula. For example, to
#'   use cubic B-splines with 7 degrees of freedom we could specify
#'   \code{tde(sex, df = 7)} in the model formula instead of just
#'   \code{tde(sex)}. See the \strong{Examples} section below for more
#'   details.
#'   }
#'   In practice, the default \code{tde()} function should provide sufficient
#'   flexibility for model most time-dependent effects. However, it is worth
#'   noting that the reliable estimation of a time-dependent effect usually
#'   requires a relatively large number of events in the data (e.g. >1000).
#' }
#'
#' @examples
#' \donttest{
#' #---------- Proportional hazards
#'
#' # Simulated data
#' library("SemiCompRisks")
#' library("scales")
#'
#'
#' # Data generation parameters
#' n <- 1500
#' beta1.true <- c(0.1, 0.1)
#' beta2.true <- c(0.2, 0.2)
#' beta3.true <- c(0.3, 0.3)
#' alpha1.true <- 0.12
#' alpha2.true <- 0.23
#' alpha3.true <- 0.34
#' kappa1.true <- 0.33
#' kappa2.true <- 0.11
#' kappa3.true <- 0.22
#' theta.true <- 0
#'
#' # Make design matrix with single binary covariate
#' x_c <- rbinom(n, size = 1, prob = 0.7)
#' x_m <- cbind(1, x_c)
#'
#' # Generate semicompeting data
#' dat_ID <- SemiCompRisks::simID(x1 = x_m, x2 = x_m, x3 = x_m,
#'                                beta1.true = beta1.true,
#'                                beta2.true = beta2.true,
#'                                beta3.true = beta3.true,
#'                                alpha1.true = alpha1.true,
#'                                alpha2.true = alpha2.true,
#'                                alpha3.true = alpha3.true,
#'                                kappa1.true = kappa1.true,
#'                                kappa2.true = kappa2.true,
#'                                kappa3.true = kappa3.true,
#'                                theta.true = theta.true,
#'                                cens = c(240, 360))
#'
#' dat_ID$x_c <- x_c
#' colnames(dat_ID)[1:4] <- c("R", "delta_R", "tt", "delta_T")
#'
#'
#'
#' formula01 <- Surv(time=rr,event=delta_R)~x_c
#' formula02 <- Surv(time=tt,event=delta_T)~x_c
#' formula12 <- Surv(time=difft,event=delta_T)~x_c
#'
#' }
idmstan <- function(formula01,
                     formula02,
                     formula12,
                     data,
                     h = 3,
                     basehaz01 = "ms",
                     basehaz02 = "ms",
                     basehaz12 = "ms",
                     basehaz_ops01,
                     basehaz_ops02,
                     basehaz_ops12,
                     prior01           = rstanarm::normal(),
                     prior_intercept01 = rstanarm::normal(),
                     prior_aux01       = rstanarm::normal(),
                     prior02           = rstanarm::normal(),
                     prior_intercept02 = rstanarm::normal(),
                     prior_aux02       = rstanarm::normal(),
                     prior12           = rstanarm::normal(),
                     prior_intercept12 = rstanarm::normal(),
                     prior_aux12       = rstanarm::normal(),
                     prior_PD        = FALSE,
                     algorithm       = c("sampling", "meanfield", "fullrank"),
                     adapt_delta     = 0.95, ...
){


  #-----------------------------
  # Pre-processing of arguments
  #-----------------------------

  if (!requireNamespace("survival"))
    stop("the 'survival' package must be installed to use this function.")

  if (missing(basehaz_ops01))
    basehaz_ops01 <- NULL
  if (missing(basehaz_ops02))
    basehaz_ops02 <- NULL
  if (missing(basehaz_ops12))
    basehaz_ops12 <- NULL

  if (missing(data) || !inherits(data, "data.frame"))
    stop("'data' must be a data frame.")

  dots      <- list(...)
  algorithm <- match.arg(algorithm)


  formula <- list(formula01, formula02, formula12)

  formula <- lapply(formula, function(f) parse_formula(f, data))

  data01 <- nruapply(0:1, function(o){
    lapply(0:1, function(p) make_model_data(formula = formula[[1]], aux_formula = formula[[2]], data = data, cens = o, aux_cens = p))
  }) # row subsetting etc.

  data02_13 <- lapply(0:1, function(o)
    make_model_data(formula = formula[[2]], aux_formula = formula[[1]], data = data, cens = o, aux_cens = 0) )

  data02_24 <- lapply(0:1, function(o)
    make_model_data(formula = formula[[1]], aux_formula = formula[[2]], data = data, cens = 1, aux_cens = o))

  data02 <- dlist( list(data02_24, data02_13))

  data12 <- lapply(0:1, function(p) make_model_data(formula = formula[[3]], aux_formula = formula[[1]], data = data, cens = 1, aux_cens = p) ) # row subsetting etc.

  data <- list(data01, data02, data12)

  #----------------
  # Construct data
  #----------------

  #----- model frame stuff

  mf_stuff <- nruapply(seq_along(formula), function(f) {
    lapply(seq_along(data[[f]]), function(d) make_model_frame(formula[[f]]$tf_form, data[[f]][[d]])
    )
  })

  mf <- lapply(mf_stuff, function(m) m$mf)  # model frame
  mt <- lapply(mf_stuff, function(m) m$mt) # model terms

  #----- dimensions and response vectors

  # entry and exit times for each row of data
  t_beg <- lapply(mf, function(m) make_t(m, type = "beg") )# entry time
  t_end <- lapply(mf, function(m) make_t(m, type = "end") )# exit time
  t_upp <- lapply(mf, function(m) make_t(m, type = "upp") )# upper time for interval censoring

  # ensure no event or censoring times are zero (leads to degenerate
  # estimate for log hazard for most baseline hazards, due to log(0))
  lapply(t_end, function(t){
    check1 <- any(t < 0, na.rm = TRUE)
    if (check1)
      stop2("All event and censoring times must be greater than 0.")
  })

  # check1 <- any(t_end01 <= 0, na.rm = TRUE)
  # check2 <- any(t_upp01 <= 0, na.rm = TRUE)
  # if (check1 || check2)
  #   stop2("All event and censoring times must be greater than 0 for rate 0 -> 1.")

  # event indicator for each row of data
  status <- lapply(mf, function(m) make_d(m))

  lapply(status, function(s){
    if (any(s < 0 || s > 3))
      stop2("Invalid status indicator in formula.")
  })

  # delayed entry indicator for each row of data
  delayed <- lapply(t_beg, function(t)
    as.logical(!t == 0) )

  # time variables for stan
  t_event <- lapply(seq_along(t_end), function(i)
    t_end[[i]][status[[i]] == 1] ) # exact event time
  t_rcens <- lapply(seq_along(t_end), function(i)
    t_end[[i]][status[[i]] == 0] ) # right censoring time
  t_lcens <- lapply(seq_along(t_end), function(i)
    t_end[[i]][status[[i]] == 2] ) # left censoring time
  t_icenl <- lapply(seq_along(t_end), function(i)
    t_end[[i]][status[[i]] == 3] )  # lower limit of interval censoring time
  t_icenu <- lapply(seq_along(t_upp), function(i)
    t_upp[[i]][status[[i]] == 3] )   # upper limit of interval censoring time
  t_delay <- lapply(seq_along(t_beg), function(i)
    t_beg[[i]][delayed[[i]]] )

  # t_event01 <- t_end01[status01 == 1] # exact event time
  # t_lcens01 <- t_end01[status01 == 2] # left  censoring time
  # t_rcens01 <- t_end01[status01 == 0] # right censoring time
  # t_icenl01 <- t_end01[status01 == 3] # lower limit of interval censoring time
  # t_icenu01 <- t_upp01[status01 == 3] # upper limit of interval censoring time
  # t_delay01 <- t_beg01[delayed01]

  # time variables for stan
  t_event02 <- t_end02[status02 == 1] # exact event time
  t_lcens02 <- t_end02[status02 == 2] # left  censoring time
  t_rcens02 <- t_end02[status02 == 0] # right censoring time
  t_icenl02 <- t_end02[status02 == 3] # lower limit of interval censoring time
  t_icenu02 <- t_upp02[status02 == 3] # upper limit of interval censoring time
  t_delay02 <- t_beg02[delayed02]

  # time variables for stan
  t_event12 <- t_end12[status12 == 1] # exact event time
  t_lcens12 <- t_end12[status12 == 2] # left  censoring time
  t_rcens12 <- t_end12[status12 == 0] # right censoring time
  t_icenl12 <- t_end12[status12 == 3] # lower limit of interval censoring time
  t_icenu12 <- t_upp12[status12 == 3] # upper limit of interval censoring time
  t_delay12 <- t_beg12[delayed12]


  # calculate log crude event rate
  t_tmp01 <- sum(rowMeans(cbind(t_end01, t_upp01), na.rm = TRUE) - t_beg01)
  d_tmp01 <- sum(!status01 == 0)
  log_crude_event_rate01 = log(d_tmp01 / t_tmp01)

  t_tmp02 <- sum(rowMeans(cbind(t_end02, t_upp02), na.rm = TRUE) - t_beg02)
  d_tmp02 <- sum(!status02 == 0)
  log_crude_event_rate02 = log(d_tmp02 / t_tmp02)

  t_tmp12 <- sum(rowMeans(cbind(t_end12, t_upp12), na.rm = TRUE) - t_beg12)
  d_tmp12 <- sum(!status12 == 0)
  log_crude_event_rate12 = log(d_tmp12 / t_tmp12)

  # dimensions
  nevent01 <- sum(status01 == 1)
  nrcens01 <- sum(status01 == 0)
  nlcens01 <- sum(status01 == 2)
  nicens01 <- sum(status01 == 3)
  ndelay01 <- sum(delayed01)

  nevent02 <- sum(status02 == 1)
  nrcens02 <- sum(status02 == 0)
  nlcens02 <- sum(status02 == 2)
  nicens02 <- sum(status02 == 3)
  ndelay02 <- sum(delayed02)

  nevent12 <- sum(status12 == 1)
  nrcens12 <- sum(status12 == 0)
  nlcens12 <- sum(status12 == 2)
  nicens12 <- sum(status12 == 3)
  ndelay12 <- sum(delayed12)


  #----- baseline hazard

  ok_basehaz <- c("exp", "weibull", "gompertz", "ms", "bs")
  ok_basehaz_ops01 <- get_ok_basehaz_ops(basehaz01)
  basehaz01 <- handle_basehaz_surv(basehaz        = basehaz01,
                                 basehaz_ops    = basehaz_ops01,
                                 ok_basehaz     = ok_basehaz,
                                 ok_basehaz_ops = ok_basehaz_ops01,
                                 times          = t_end01,
                                 status         = status01,
                                 min_t          = min(t_beg01),
                                 max_t          = max(c(t_end01,t_upp01), na.rm = TRUE))
  nvars01 <- basehaz01$nvars # number of basehaz aux parameters

  # flag if intercept is required for baseline hazard
  has_intercept01   <- ai(has_intercept(basehaz01))


  ok_basehaz_ops02 <- get_ok_basehaz_ops(basehaz02)
  basehaz02 <- handle_basehaz_surv(basehaz        = basehaz02,
                                   basehaz_ops    = basehaz_ops02,
                                   ok_basehaz     = ok_basehaz,
                                   ok_basehaz_ops = ok_basehaz_ops02,
                                   times          = t_end02,
                                   status         = status02,
                                   min_t          = min(t_beg02),
                                   max_t          = max(c(t_end02,t_upp02), na.rm = TRUE))
  nvars02 <- basehaz02$nvars # number of basehaz aux parameters

  # flag if intercept is required for baseline hazard
  has_intercept02   <- ai(has_intercept(basehaz02))

  ok_basehaz_ops12 <- get_ok_basehaz_ops(basehaz12)
  basehaz12 <- handle_basehaz_surv(basehaz        = basehaz12,
                                   basehaz_ops    = basehaz_ops12,
                                   ok_basehaz     = ok_basehaz,
                                   ok_basehaz_ops = ok_basehaz_ops12,
                                   times          = t_end12,
                                   status         = status12,
                                   min_t          = min(t_beg12),
                                   max_t          = max(c(t_end12,t_upp12), na.rm = TRUE))
  nvars12 <- basehaz12$nvars # number of basehaz aux parameters

  # flag if intercept is required for baseline hazard
  has_intercept12   <- ai(has_intercept(basehaz12))


  has_quadrature01 <-  FALSE # not implemented
  has_quadrature02 <-  FALSE # not implemented
  has_quadrature12 <-  FALSE # not implemented
  #
  # if(has_quadrature01){
  #
  # }else{
  #   cpts01     <- rep(0,0)
  #   len_cpts01 <- 0L
  #   idx_cpts01 <- matrix(0,7,2)
  #
  #   if (!qnodes01 == 15) # warn user if qnodes is not equal to the default
  #     warning2("There is no quadrature required 0 -> 1 so 'qnodes' is being ignored.")
  # }
  # NOT IMPLEMENTED YET

  # ms = basis_matrix(times, basis = basehaz$basis, integrate = FALSE)

  #----- basis terms for baseline hazard

  if (has_quadrature01) {

    basis_cpts <- make_basis(cpts01, basehaz01)

  } else { # NOT IMPLEMENTED

    basis_event01  <- make_basis(t_event01, basehaz01)

    ibasis_event01 <- make_basis(t_event01, basehaz01, integrate = TRUE)
    ibasis_lcens01 <- make_basis(t_lcens01, basehaz01, integrate = TRUE)
    ibasis_rcens01 <- make_basis(t_rcens01, basehaz01, integrate = TRUE)
    ibasis_icenl01 <- make_basis(t_icenl01, basehaz01, integrate = TRUE)
    ibasis_icenu01 <- make_basis(t_icenu01, basehaz01, integrate = TRUE)
    ibasis_delay01 <- make_basis(t_delay01, basehaz01, integrate = TRUE)

  }

  if (has_quadrature02) {

    basis_cpts <- make_basis(cpts02, basehaz02)

  } else { # NOT IMPLEMENTED

    basis_event02  <- make_basis(t_event02, basehaz02)

    ibasis_event02 <- make_basis(t_event02, basehaz02, integrate = TRUE)
    ibasis_lcens02 <- make_basis(t_lcens02, basehaz02, integrate = TRUE)
    ibasis_rcens02 <- make_basis(t_rcens02, basehaz02, integrate = TRUE)
    ibasis_icenl02 <- make_basis(t_icenl02, basehaz02, integrate = TRUE)
    ibasis_icenu02 <- make_basis(t_icenu02, basehaz02, integrate = TRUE)
    ibasis_delay02 <- make_basis(t_delay02, basehaz02, integrate = TRUE)

  }

  if (has_quadrature12) {

    basis_cpts <- make_basis(cpts12, basehaz12)

  } else { # NOT IMPLEMENTED

    basis_event12  <- make_basis(t_event12, basehaz12)

    ibasis_event12 <- make_basis(t_event12, basehaz12, integrate = TRUE)
    ibasis_lcens12 <- make_basis(t_lcens12, basehaz12, integrate = TRUE)
    ibasis_rcens12 <- make_basis(t_rcens12, basehaz12, integrate = TRUE)
    ibasis_icenl12 <- make_basis(t_icenl12, basehaz12, integrate = TRUE)
    ibasis_icenu12 <- make_basis(t_icenu12, basehaz12, integrate = TRUE)
    ibasis_delay12 <- make_basis(t_delay12, basehaz12, integrate = TRUE)

  }


  #----- predictor matrices

  # time-fixed predictor matrix




  x_stuff01 <- make_x(formula[[1]]$tf_form, mf[[1]])
  x01          <- x_stuff01$x
  x_bar01      <- x_stuff01$x_bar
  x_centered01 <- x_stuff01$x_centered
  x_event01 <- keep_rows(x01, status01 == 1)
  x_lcens01 <- keep_rows(x01, status01 == 2)
  x_rcens01 <- keep_rows(x01, status01 == 0)
  x_icens01 <- keep_rows(x01, status01 == 3)
  x_delay01 <- keep_rows(x01, delayed01)
  K01 <- ncol(x01)


  x_stuff02 <- make_x(formula[[2]]$tf_form, mf[[2]])
  x02          <- x_stuff02$x
  x_bar02      <- x_stuff02$x_bar
  x_centered02 <- x_stuff02$x_centered
  x_event02 <- keep_rows(x02, status02 == 1)
  x_lcens02 <- keep_rows(x02, status02 == 2)
  x_rcens02 <- keep_rows(x02, status02 == 0)
  x_icens02 <- keep_rows(x02, status02 == 3)
  x_delay02 <- keep_rows(x02, delayed02)
  K02 <- ncol(x02)


  x_stuff12 <- make_x(formula[[3]]$tf_form, mf[[3]])
  x12          <- x_stuff12$x
  x_bar12      <- x_stuff12$x_bar
  x_centered12 <- x_stuff12$x_centered
  x_event12 <- keep_rows(x12, status12 == 1)
  x_lcens12 <- keep_rows(x12, status12 == 2)
  x_rcens12 <- keep_rows(x12, status12 == 0)
  x_icens12 <- keep_rows(x12, status12 == 3)
  x_delay12 <- keep_rows(x12, delayed12)
  K12 <- ncol(x12)

  standata <- nlist(
    K01, K02, K12,
    nvars01, nvars02, nvars12,
    x_bar01, x_bar02, x_bar12,
    has_intercept01, has_intercept02, has_intercept12,
    type01 = basehaz01$type, type02 = basehaz02$type, type12 = basehaz12$type,
    log_crude_event_rate01, log_crude_event_rate02, log_crude_event_rate12,

    nevent01       = if (has_quadrature01) 0L else nevent01,
    nlcens01       = if (has_quadrature01) 0L else nlcens01,
    nrcens01       = if (has_quadrature01) 0L else nrcens01,
    nicens01       = if (has_quadrature01) 0L else nicens01,
    ndelay01       = if (has_quadrature01) 0L else ndelay01,

    t_event01      = if (has_quadrature01) rep(0,0) else t_event01,
    t_lcens01      = if (has_quadrature01) rep(0,0) else t_lcens01,
    t_rcens01      = if (has_quadrature01) rep(0,0) else t_rcens01,
    t_icenl01      = if (has_quadrature01) rep(0,0) else t_icenl01,
    t_icenu01      = if (has_quadrature01) rep(0,0) else t_icenu01,
    t_delay01      = if (has_quadrature01) rep(0,0) else t_delay01,

    x_event01      = if (has_quadrature01) matrix(0,0,K01) else x_event01,
    x_lcens01      = if (has_quadrature01) matrix(0,0,K01) else x_lcens01,
    x_rcens01      = if (has_quadrature01) matrix(0,0,K01) else x_rcens01,
    x_icens01      = if (has_quadrature01) matrix(0,0,K01) else x_icens01,
    x_delay01      = if (has_quadrature01) matrix(0,0,K01) else x_delay01,

    basis_event01  = if (has_quadrature01) matrix(0,0,nvars01) else basis_event01,
    ibasis_event01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_event01,
    ibasis_lcens01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_lcens01,
    ibasis_rcens01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_rcens01,
    ibasis_icenl01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_icenl01,
    ibasis_icenu01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_icenu01,
    ibasis_delay01 = if (has_quadrature01) matrix(0,0,nvars01) else ibasis_delay01,

    nevent02       = if (has_quadrature02) 0L else nevent02,
    nlcens02       = if (has_quadrature02) 0L else nlcens02,
    nrcens02       = if (has_quadrature02) 0L else nrcens02,
    nicens02       = if (has_quadrature02) 0L else nicens02,
    ndelay02       = if (has_quadrature02) 0L else ndelay02,

    t_event02      = if (has_quadrature02) rep(0,0) else t_event02,
    t_lcens02      = if (has_quadrature02) rep(0,0) else t_lcens02,
    t_rcens02      = if (has_quadrature02) rep(0,0) else t_rcens02,
    t_icenl02      = if (has_quadrature02) rep(0,0) else t_icenl02,
    t_icenu02      = if (has_quadrature02) rep(0,0) else t_icenu02,
    t_delay02      = if (has_quadrature02) rep(0,0) else t_delay02,

    x_event02      = if (has_quadrature02) matrix(0,0,K02) else x_event02,
    x_lcens02      = if (has_quadrature02) matrix(0,0,K02) else x_lcens02,
    x_rcens02      = if (has_quadrature02) matrix(0,0,K02) else x_rcens02,
    x_icens02      = if (has_quadrature02) matrix(0,0,K02) else x_icens02,
    x_delay02      = if (has_quadrature02) matrix(0,0,K02) else x_delay02,

    basis_event02  = if (has_quadrature02) matrix(0,0,nvars02) else basis_event02,
    ibasis_event02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_event02,
    ibasis_lcens02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_lcens02,
    ibasis_rcens02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_rcens02,
    ibasis_icenl02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_icenl02,
    ibasis_icenu02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_icenu02,
    ibasis_delay02 = if (has_quadrature02) matrix(0,0,nvars02) else ibasis_delay02,

    nevent12       = if (has_quadrature12) 0L else nevent12,
    nlcens12       = if (has_quadrature12) 0L else nlcens12,
    nrcens12       = if (has_quadrature12) 0L else nrcens12,
    nicens12       = if (has_quadrature12) 0L else nicens12,
    ndelay12       = if (has_quadrature12) 0L else ndelay12,

    t_event12      = if (has_quadrature12) rep(0,0) else t_event12,
    t_lcens12      = if (has_quadrature12) rep(0,0) else t_lcens12,
    t_rcens12      = if (has_quadrature12) rep(0,0) else t_rcens12,
    t_icenl12      = if (has_quadrature12) rep(0,0) else t_icenl12,
    t_icenu12      = if (has_quadrature12) rep(0,0) else t_icenu12,
    t_delay12      = if (has_quadrature12) rep(0,0) else t_delay12,

    x_event12      = if (has_quadrature12) matrix(0,0,K12) else x_event12,
    x_lcens12      = if (has_quadrature12) matrix(0,0,K12) else x_lcens12,
    x_rcens12      = if (has_quadrature12) matrix(0,0,K12) else x_rcens12,
    x_icens12      = if (has_quadrature12) matrix(0,0,K12) else x_icens12,
    x_delay12      = if (has_quadrature12) matrix(0,0,K12) else x_delay12,

    basis_event12  = if (has_quadrature12) matrix(0,0,nvars12) else basis_event12,
    ibasis_event12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_event12,
    ibasis_lcens12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_lcens12,
    ibasis_rcens12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_rcens12,
    ibasis_icenl12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_icenl12,
    ibasis_icenu12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_icenu12,
    ibasis_delay12 = if (has_quadrature12) matrix(0,0,nvars12) else ibasis_delay12

)

  #----- priors and hyperparameters

  # valid priors
  ok_dists <- nlist("normal",
                    student_t = "t",
                    "cauchy",
                    "hs",
                    "hs_plus",
                    "laplace",
                    "lasso") # disallow product normal
  ok_intercept_dists <- ok_dists[1:3]
  ok_aux_dists       <- ok_dists[1:3]
  ok_smooth_dists    <- c(ok_dists[1:3], "exponential")

  # priors
  user_prior_stuff01 <- prior_stuff01 <-
    handle_glm_prior(prior01,
                     nvars = K01,
                     default_scale = 2.5,
                     link = NULL,
                     ok_dists = ok_dists)

  user_prior_intercept_stuff01 <- prior_intercept_stuff01 <-
    handle_glm_prior(prior_intercept01,
                     nvars = 1,
                     default_scale = 20,
                     link = NULL,
                     ok_dists = ok_intercept_dists)

  user_prior_aux_stuff01 <- prior_aux_stuff01 <-
    handle_glm_prior(prior_aux01,
                     nvars = basehaz01$nvars,
                     default_scale = get_default_aux_scale(basehaz01),
                     link = NULL,
                     ok_dists = ok_aux_dists)

  # stop null priors if prior_PD is TRUE
  if (prior_PD) {
    if (is.null(prior01))
      stop("'prior' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_intercept01) && has_intercept01)
      stop("'prior_intercept' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_aux01))
      stop("'prior_aux' cannot be NULL if 'prior_PD' is TRUE")
  }

  # priors
  user_prior_stuff02 <- prior_stuff02 <-
    handle_glm_prior(prior02,
                     nvars = K02,
                     default_scale = 2.5,
                     link = NULL,
                     ok_dists = ok_dists)

  user_prior_intercept_stuff02 <- prior_intercept_stuff02 <-
    handle_glm_prior(prior_intercept02,
                     nvars = 1,
                     default_scale = 20,
                     link = NULL,
                     ok_dists = ok_intercept_dists)

  user_prior_aux_stuff02 <- prior_aux_stuff02 <-
    handle_glm_prior(prior_aux02,
                     nvars = basehaz02$nvars,
                     default_scale = get_default_aux_scale(basehaz02),
                     link = NULL,
                     ok_dists = ok_aux_dists)

  # stop null priors if prior_PD is TRUE
  if (prior_PD) {
    if (is.null(prior02))
      stop("'prior' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_intercept02) && has_intercept02)
      stop("'prior_intercept' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_aux02))
      stop("'prior_aux' cannot be NULL if 'prior_PD' is TRUE")
  }

  # priors
  user_prior_stuff12 <- prior_stuff12 <-
    handle_glm_prior(prior12,
                     nvars = K12,
                     default_scale = 2.5,
                     link = NULL,
                     ok_dists = ok_dists)

  user_prior_intercept_stuff12 <- prior_intercept_stuff12 <-
    handle_glm_prior(prior_intercept12,
                     nvars = 1,
                     default_scale = 20,
                     link = NULL,
                     ok_dists = ok_intercept_dists)

  user_prior_aux_stuff12 <- prior_aux_stuff12 <-
    handle_glm_prior(prior_aux12,
                     nvars = basehaz12$nvars,
                     default_scale = get_default_aux_scale(basehaz12),
                     link = NULL,
                     ok_dists = ok_aux_dists)

  # stop null priors if prior_PD is TRUE
  if (prior_PD) {
    if (is.null(prior12))
      stop("'prior' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_intercept12) && has_intercept12)
      stop("'prior_intercept' cannot be NULL if 'prior_PD' is TRUE")
    if (is.null(prior_aux12))
      stop("'prior_aux' cannot be NULL if 'prior_PD' is TRUE")
  }

  # autoscaling of priors
  prior_stuff01           <- autoscale_prior(prior_stuff01, predictors = x01)
  prior_intercept_stuff01 <- autoscale_prior(prior_intercept_stuff01)
  prior_aux_stuff01       <- autoscale_prior(prior_aux_stuff01)

  prior_stuff02           <- autoscale_prior(prior_stuff02, predictors = x02)
  prior_intercept_stuff02 <- autoscale_prior(prior_intercept_stuff02)
  prior_aux_stuff02       <- autoscale_prior(prior_aux_stuff02)

  prior_stuff12           <- autoscale_prior(prior_stuff12, predictors = x12)
  prior_intercept_stuff12 <- autoscale_prior(prior_intercept_stuff12)
  prior_aux_stuff12       <- autoscale_prior(prior_aux_stuff12)

  # priors
  standata$prior_dist01              <- prior_stuff01$prior_dist
  standata$prior_dist_for_intercept01 <- prior_intercept_stuff01$prior_dist
  standata$prior_dist_for_aux01      <- prior_aux_stuff01$prior_dist

  standata$prior_dist02              <- prior_stuff02$prior_dist
  standata$prior_dist_for_intercept02 <- prior_intercept_stuff02$prior_dist
  standata$prior_dist_for_aux02      <- prior_aux_stuff02$prior_dist

  standata$prior_dist12              <- prior_stuff12$prior_dist
  standata$prior_dist_for_intercept12 <- prior_intercept_stuff12$prior_dist
  standata$prior_dist_for_aux12      <- prior_aux_stuff12$prior_dist


  # hyperparameters
  standata$prior_mean01               <- prior_stuff01$prior_mean
  standata$prior_scale01              <- prior_stuff01$prior_scale
  standata$prior_df01                 <- prior_stuff01$prior_df
  standata$prior_mean_for_intercept01 <- c(prior_intercept_stuff01$prior_mean)
  standata$prior_scale_for_intercept01 <- c(prior_intercept_stuff01$prior_scale)
  standata$prior_df_for_intercept01   <- c(prior_intercept_stuff01$prior_df)
  standata$prior_scale_for_aux01      <- prior_aux_stuff01$prior_scale
  standata$prior_df_for_aux01         <- prior_aux_stuff01$prior_df

  standata$prior_mean02               <- prior_stuff02$prior_mean
  standata$prior_scale02              <- prior_stuff02$prior_scale
  standata$prior_df02                 <- prior_stuff02$prior_df
  standata$prior_mean_for_intercept02 <- c(prior_intercept_stuff02$prior_mean)
  standata$prior_scale_for_intercept02 <- c(prior_intercept_stuff02$prior_scale)
  standata$prior_df_for_intercept02   <- c(prior_intercept_stuff02$prior_df)
  standata$prior_scale_for_aux02      <- prior_aux_stuff02$prior_scale
  standata$prior_df_for_aux02         <- prior_aux_stuff02$prior_df

  standata$prior_mean12               <- prior_stuff12$prior_mean
  standata$prior_scale12              <- prior_stuff12$prior_scale
  standata$prior_df12                 <- prior_stuff12$prior_df
  standata$prior_mean_for_intercept12 <- c(prior_intercept_stuff12$prior_mean)
  standata$prior_scale_for_intercept12 <- c(prior_intercept_stuff12$prior_scale)
  standata$prior_df_for_intercept12   <- c(prior_intercept_stuff12$prior_df)
  standata$prior_scale_for_aux12      <- prior_aux_stuff12$prior_scale
  standata$prior_df_for_aux12         <- prior_aux_stuff12$prior_df

  # any additional flags
  standata$prior_PD <- ai(prior_PD)


  #---------------
  # Prior summary
  #---------------

  prior_info01 <- summarize_jm_prior(
    user_priorEvent           = user_prior_stuff01,
    user_priorEvent_intercept = user_prior_intercept_stuff01,
    user_priorEvent_aux       = user_prior_aux_stuff01,
    adjusted_priorEvent_scale           = prior_stuff01$prior_scale,
    adjusted_priorEvent_intercept_scale = prior_intercept_stuff01$prior_scale,
    adjusted_priorEvent_aux_scale       = prior_aux_stuff01$prior_scale,
    e_has_intercept  = has_intercept01,
    e_has_predictors = K01 > 0,
    basehaz = basehaz01
  )

  prior_info02 <- summarize_jm_prior(
    user_priorEvent           = user_prior_stuff02,
    user_priorEvent_intercept = user_prior_intercept_stuff02,
    user_priorEvent_aux       = user_prior_aux_stuff02,
    adjusted_priorEvent_scale           = prior_stuff02$prior_scale,
    adjusted_priorEvent_intercept_scale = prior_intercept_stuff02$prior_scale,
    adjusted_priorEvent_aux_scale       = prior_aux_stuff02$prior_scale,
    e_has_intercept  = has_intercept02,
    e_has_predictors = K02 > 0,
    basehaz = basehaz02
  )

  prior_info12 <- summarize_jm_prior(
    user_priorEvent           = user_prior_stuff12,
    user_priorEvent_intercept = user_prior_intercept_stuff12,
    user_priorEvent_aux       = user_prior_aux_stuff12,
    adjusted_priorEvent_scale           = prior_stuff12$prior_scale,
    adjusted_priorEvent_intercept_scale = prior_intercept_stuff12$prior_scale,
    adjusted_priorEvent_aux_scale       = prior_aux_stuff12$prior_scale,
    e_has_intercept  = has_intercept12,
    e_has_predictors = K12 > 0,
    basehaz = basehaz12
  )

  #-----------
  # Fit model
  #-----------

  # obtain stan model code
  #stanfit  <- stanmodels$simple_competing_stan
  # stanfit <-  "src/stan_files/simple_competing_stan.stan"

  # specify parameters for stan to monitor
  # stanpars <- c(if (standata$has_intercept01) "alpha01",
  #               if (standata$K01)             "beta01",
  #               if (standata$nvars01)         "aux01",
  #               if (standata$has_intercept02) "alpha02",
  #               if (standata$K02)             "beta02",
  #               if (standata$nvars02)         "aux02",
  #               if (standata$has_intercept12) "alpha12",
  #               if (standata$K12)             "beta12",
  #               if (standata$nvars12)         "aux12")

  #
  #
  # # fit model using stan
  # if (algorithm == "sampling") { # mcmc
  #   args <- set_sampling_args(
  #     object = stanfit,
  #     data   = standata,
  #     pars   = stanpars,
  #     prior  = prior01,
  #     user_dots = list(...),
  #     user_adapt_delta = adapt_delta,
  #     show_messages = FALSE)
  #   stanfit <- do.call(rstan::sampling, args)
  # } else { # meanfield or fullrank vb
  #   args <- nlist(
  #     object = stanfit,
  #     data   = standata,
  #     pars   = stanpars,
  #     algorithm
  #   )
  #   args[names(dots)] <- dots
  #   stanfit <- do.call(rstan::vb, args)
  # }


  return(standata)
}
