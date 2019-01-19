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
idm_stan <- function(formula01,
                     formula02,
                     formula12,
                     data,
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
  basehaz_ops <- list(basehaz_ops01, basehaz_ops02, basehaz_ops12)

  if (missing(data) || !inherits(data, "data.frame"))
    stop("'data' must be a data frame.")

  dots      <- list(...)
  algorithm <- match.arg(algorithm)


  formula <- list(formula01, formula02, formula12)
  h <- length(formula)

  formula <- lapply(formula, function(f) parse_formula(f, data))

  data01 <- nruapply(0:1, function(o){
    lapply(0:1, function(p) make_model_data(formula = formula[[1]], aux_formula = formula[[2]], data = data, cens = p, aux_cens = o))
  }) # row subsetting etc.

  data02_13 <- lapply(0:1, function(o)
    make_model_data(formula = formula[[2]], aux_formula = formula[[1]], data = data, cens = o, aux_cens = 0) )

  data02_24 <- lapply(0:1, function(o)
    make_model_data(formula = formula[[2]], aux_formula = formula[[1]], data = data, cens = o, aux_cens = 1))

  data02 <-  list(data02_13[[1]], data02_24[[1]], data02_13[[2]], data02_24[[2]] )

  data12 <- lapply(0:1, function(p) make_model_data(formula = formula[[3]], aux_formula = formula[[1]], data = data, cens = p, aux_cens = 1) ) # row subsetting etc.

  data <- list(data01, data02, data12)

  #----------------
  # Construct data
  #----------------

  #----- model frame stuff

  mf_stuff <- lapply(seq_along(formula), function(f) {
    lapply(seq_along(data[[f]]), function(d) make_model_frame(formula[[f]]$tf_form, data[[f]][[d]])
    )
  })

  mf_stuff[[2]][[2]] <- make_model_frame(formula[[1]]$tf_form,
                                         data[[2]][[2]])
  mf_stuff[[2]][[4]] <- make_model_frame(formula[[1]]$tf_form,
                                         data[[2]][[4]])

  mf <- lapply(mf_stuff, function(m) lapply(m, function(n) n$mf) )  # model frame
  mt <- lapply(mf_stuff, function(m) lapply(m, function(n) n$mt) ) # model terms

  #----- dimensions and response vectors

  # entry and exit times for each row of data
  t_beg <- lapply(mf, function(m) lapply(m, function(n) make_t(n, type = "beg") ) )# entry time
  t_end <- lapply(mf, function(m) lapply(m, function(n) make_t(n, type = "end") ) )# exit time
  t_upp <- lapply(mf, function(m) lapply(m, function(n) make_t(n, type = "upp") ) )# upper time for interval censoring

  # ensure no event or censoring times are zero (leads to degenerate
  # estimate for log hazard for most baseline hazards, due to log(0))
  for(i in seq_along(dlist(t_end) ) ){
    check1 <- any(dlist(t_end)[[i]] <= 0, na.rm = TRUE)
    if (check1)
      stop2("All event and censoring times must be greater than 0.")
  }

  # event indicator for each row of data
  status <- lapply(mf, function(m) lapply(m, function(n) make_d(n)) )
  # correct status data02type3
  status[[2]][[2]] <- rep(0, length(status[[2]][[2]]) )
  status[[2]][[4]] <- rep(0, length(status[[2]][[4]]) )

  for(i in seq_along(dlist(status))){
    if (any(dlist(status)[[i]] < 0 || dlist(status)[[i]] > 3))
      stop2("Invalid status indicator in formula.")
  }

  # delayed entry indicator for each row of data
  delayed <- lapply(t_beg, function(t) lapply(t, function(i) as.logical(!i == 0) ) )

  # time variables for stan
  t_event <- lapply(seq_along(t_end), function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_end[[t]][[i]][status[[t]][[i]] == 1] ) ) # exact event time
  t_rcens <- lapply(seq_along(t_end),  function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_end[[t]][[i]][status[[t]][[i]] == 0] ) )# right censoring time
  t_lcens <- lapply(seq_along(t_end),  function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_end[[t]][[i]][status[[t]][[i]] == 2] ) )# left censoring time
  t_icenl <- lapply(seq_along(t_end),  function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_end[[t]][[i]][status[[t]][[i]] == 3] ) ) # lower limit of interval censoring time
  t_icenu <- lapply(seq_along(t_upp), function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_upp[[t]][[i]][status[[t]][[i]] == 3] )  ) # upper limit of interval censoring time
  t_delay <- lapply(seq_along(t_beg),  function(t) lapply(seq_along(t_end[[t]]), function(i)
    t_beg[[t]][[i]][delayed[[t]][[i]]] )   )

  # calculate log crude event rate
  t_tmp <- lapply(seq_along(t_end), function(t) sum(rowMeans(cbind(dlist(t_end[[t]]), dlist(t_upp[[t]]) ), na.rm = TRUE) - dlist(t_beg[[t]] )) )

  d_tmp <- lapply(status, function(s) sum(!dlist(s) == 0) )

  log_crude_event_rate <- lapply(seq_along(t_tmp), function(t) log(d_tmp[[t]] / t_tmp[[t]]))

  # dimensions
  nevent <- lapply(status, function(s) lapply(s, function(p) sum(p == 1)) )
  nrcens <- lapply(status,  function(s) lapply(s, function(p) sum(p == 0)) )
  nlcens <- lapply(status, function(s) lapply(s, function(p) sum(p == 2)) )
  nicens <- lapply(status, function(s) lapply(s, function(p) sum(p == 3)) )
  ndelay <- lapply(delayed, function(d) lapply(d, function(e) sum(e)) )

  #----- baseline hazard

  ok_basehaz <- c("exp", "weibull", "gompertz", "ms", "bs")
  basehaz <- list(basehaz01, basehaz02, basehaz12)
  ok_basehaz_ops <- lapply(basehaz, function(b) get_ok_basehaz_ops(b))

  basehaz <- lapply(seq_along(basehaz), function(b) lapply(seq_along(t_end[[b]]), function(i)
    sw( handle_basehaz_surv(basehaz = basehaz[[b]],
                        basehaz_ops = basehaz_ops[[b]],
                        ok_basehaz = ok_basehaz,
                        ok_basehaz_ops = ok_basehaz_ops[[b]],
                        times = t_end[[b]][[i]],
                        status = status[[b]][[i]],
                        min_t = min(t_beg[[b]][[i]]),
                        max_t = max(c(t_end[[b]][[i]], t_upp[[b]][[i]]), na.rm = TRUE ) )

    )))

  nvars <- lapply(basehaz, function(f) lapply(f, function(b) b$nvars) ) # number of basehaz aux parameters

  # flag if intercept is required for baseline hazard
  has_intercept   <- lapply(basehaz, function(f) lapply(f, function(b) ai(has_intercept(b)) ) )
  has_quadrature <- lapply(basehaz, function(f) lapply(f, function(b) FALSE)) #NOT IMPLEMENTED IN THIS RELEASE


  #----- basis terms for baseline hazard
  basis_cpts <- rep(list(list()), h)
  basis_event  <- rep(list(list()), h)
  ibasis_event <- rep(list(list()), h)
  ibasis_lcens <- rep(list(list()), h)
  ibasis_rcens <- rep(list(list()), h)
  ibasis_icenl <- rep(list(list()), h)
  ibasis_icenu <- rep(list(list()), h)
  ibasis_delay <- rep(list(list()), h)
  for(b in seq_along(basehaz) ){
    for(i in seq_along(basehaz[[b]]) ){
      if (has_quadrature[[b]][[i]]) {
        # NOT IMPLEMENTED
        #basis_cpts[[b]][[i]] <- make_basis(cpts[[b]], basehaz[[i]])

      } else {
        basis_event[[b]][[i]]  <- make_basis(t_event[[b]][[i]], basehaz[[b]][[i]])

        ibasis_event[[b]][[i]]  <- make_basis(t_event[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
        ibasis_lcens[[b]][[i]]  <- make_basis(t_lcens[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
        ibasis_rcens[[b]][[i]]  <- make_basis(t_rcens[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
        ibasis_icenl[[b]][[i]]  <- make_basis(t_icenl[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
        ibasis_icenu[[b]][[i]]  <- make_basis(t_icenu[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
        ibasis_delay[[b]][[i]]  <-make_basis(t_delay[[b]][[i]], basehaz[[b]][[i]], integrate = TRUE)
     }
    }
  }


  #----- predictor matrices

  # time-fixed predictor matrix
  x_stuff <- lapply(mf[[1]], function(m) make_x(formula[[1]]$tf_form, m) )
  x_stuff <- list(x_stuff, x_stuff, list(x_stuff[[2]], x_stuff[[4]]))

  x          <- lapply(x_stuff, function(m) lapply(m, function(n) n$x) )

  # column means of predictor matrix
  x_bar01 <- aa(colMeans(do.call(rbind, x[[1]])) )
  x_bar02 <- aa(colMeans(do.call(rbind, x[[2]])) )
  x_bar12 <- aa(colMeans(do.call(rbind, x[[3]])) )

  x_centered <-  lapply(x_stuff, function(m) lapply(m, function(n) n$x_centered  ))
  x_event <- lapply(seq_along(status), function(s) lapply( seq_along(status[[s]]), function(i) keep_rows(x_centered[[s]][[i]], status[[s]][[i]] == 1) ) )
  x_lcens <- lapply(seq_along(status), function(s) lapply( seq_along(status[[s]]), function(i)keep_rows(x_centered[[s]][[i]], status[[s]][[i]] == 2) ))
  x_rcens <- lapply(seq_along(status), function(s) lapply( seq_along(status[[s]]), function(i)keep_rows(x_centered[[s]][[i]], status[[s]][[i]] == 0) ))
  x_icens <- lapply(seq_along(status), function(s) lapply( seq_along(status[[s]]), function(i)keep_rows(x_centered[[s]][[i]], status[[s]][[i]] == 3) ))
  x_delay <- lapply(seq_along(status), function(s) lapply( seq_along(status[[s]]), function(i)keep_rows(x_centered[[s]][[i]], delayed[[s]][[i]]) ))
  K <- lapply(x, function(s) lapply( s, function(i) ncol(i) ))


  standata <- nlist(

    K01 = K[[1]][[1]], K02 = K[[2]][[1]], K12 = K[[3]][[1]],
    nvars01 = nvars[[1]][[1]] , nvars02 = nvars[[2]][[1]], nvars12 = nvars[[3]][[1]],

    x_bar01 = x_bar01,
    x_bar02 = x_bar02,
    x_bar12 = x_bar02,

    has_intercept01 = has_intercept[[1]][[1]], has_intercept02 = has_intercept[[2]][[1]], has_intercept12 = has_intercept[[3]][[1]],
    type01 = basehaz[[1]][[1]]$type, type02 = basehaz[[2]][[1]]$type, type12 = basehaz[[3]][[1]]$type,

    log_crude_event_rate01 = log_crude_event_rate[[1]],
    log_crude_event_rate02 = log_crude_event_rate[[2]],
    log_crude_event_rate12 = log_crude_event_rate[[3]],


    nevent01type2 = nevent[[1]][[2]],
    nevent01type4 = nevent[[1]][[4]],
    nevent02type3 = nevent[[2]][[3]],
    nevent12type4 = nevent[[3]][[2]],

    nrcens01type1 = nrcens[[1]][[1]],
    nrcens01type3 = nrcens[[1]][[3]],
    nrcens02type1 = nrcens[[2]][[1]],
    nrcens02type2 = nrcens[[2]][[2]],
    nrcens02type4 = nrcens[[2]][[4]],
    nrcens12type2 = nrcens[[3]][[1]],

    t_event01type2 = t_event[[1]][[2]],
    t_event01type4 = t_event[[1]][[4]],
    t_event02type3 = t_event[[2]][[3]],
    t_event12type4 = t_event[[3]][[2]],

    t_rcens01type1 = t_rcens[[1]][[1]],
    t_rcens01type3 = t_rcens[[1]][[3]],
    t_rcens02type1 = t_rcens[[2]][[1]],
    t_rcens02type2 = t_rcens[[2]][[2]],
    t_rcens02type4 = t_rcens[[2]][[4]],
    t_rcens12type2 = t_rcens[[3]][[1]],

    x_event01type2 = x_event[[1]][[2]],
    x_event01type4 = x_event[[1]][[4]],
    x_event02type3 = x_event[[2]][[3]],
    x_event12type4 = x_event[[3]][[2]],

    x_rcens01type1 = x_rcens[[1]][[1]],
    x_rcens01type3 = x_rcens[[1]][[3]],
    x_rcens02type1 = x_rcens[[2]][[1]],
    x_rcens02type2 = x_rcens[[2]][[2]],
    x_rcens02type4 = x_rcens[[2]][[4]],
    x_rcens12type2 = x_rcens[[3]][[1]],

    basis_event01type2 = basis_event[[1]][[2]],
    basis_event01type4 = basis_event[[1]][[4]],
    basis_event02type3 = basis_event[[2]][[3]],
    basis_event12type4 = basis_event[[3]][[2]],

    ibasis_event01type2 = ibasis_event[[1]][[2]],
    ibasis_event01type4 = ibasis_event[[1]][[4]],
    ibasis_event02type3 = ibasis_event[[2]][[3]],
    ibasis_event12type4 = ibasis_event[[3]][[2]],

    ibasis_rcens01type1 = ibasis_rcens[[1]][[1]],
    ibasis_rcens01type3 = ibasis_rcens[[1]][[3]],
    ibasis_rcens02type1 = ibasis_rcens[[2]][[1]],
    ibasis_rcens02type2 = ibasis_rcens[[2]][[2]],
    ibasis_rcens02type4 = ibasis_rcens[[2]][[4]],
    ibasis_rcens12type2 = ibasis_rcens[[3]][[1]]
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
                     nvars = standata$K01,
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
                     nvars = standata$nvars01,
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
                     nvars = standata$K02,
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
                     nvars = standata$nvars02,
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
                     nvars = standata$K12,
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
                     nvars = standata$nvars12,
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
  prior_stuff01           <- autoscale_prior(prior_stuff01, predictors = do.call(rbind, x[[1]]) )
  prior_intercept_stuff01 <- autoscale_prior(prior_intercept_stuff01)
  prior_aux_stuff01       <- autoscale_prior(prior_aux_stuff01)

  prior_stuff02           <- autoscale_prior(prior_stuff02, predictors = do.call(rbind, x[[2]]) )
  prior_intercept_stuff02 <- autoscale_prior(prior_intercept_stuff02)
  prior_aux_stuff02       <- autoscale_prior(prior_aux_stuff02)

  prior_stuff12           <- autoscale_prior(prior_stuff12, predictors = do.call(rbind, x[[3]]) )
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
    e_has_intercept  = standata$has_intercept01,
    e_has_predictors = standata$K01 > 0,
    basehaz = basehaz[[1]][[1]]
  )

  prior_info02 <- summarize_jm_prior(
    user_priorEvent           = user_prior_stuff02,
    user_priorEvent_intercept = user_prior_intercept_stuff02,
    user_priorEvent_aux       = user_prior_aux_stuff02,
    adjusted_priorEvent_scale           = prior_stuff02$prior_scale,
    adjusted_priorEvent_intercept_scale = prior_intercept_stuff02$prior_scale,
    adjusted_priorEvent_aux_scale       = prior_aux_stuff02$prior_scale,
    e_has_intercept  = standata$has_intercept02,
    e_has_predictors = standata$K02 > 0,
    basehaz = basehaz[[2]][[1]]
  )

  prior_info12 <- summarize_jm_prior(
    user_priorEvent           = user_prior_stuff12,
    user_priorEvent_intercept = user_prior_intercept_stuff12,
    user_priorEvent_aux       = user_prior_aux_stuff12,
    adjusted_priorEvent_scale           = prior_stuff12$prior_scale,
    adjusted_priorEvent_intercept_scale = prior_intercept_stuff12$prior_scale,
    adjusted_priorEvent_aux_scale       = prior_aux_stuff12$prior_scale,
    e_has_intercept  = standata$has_intercept12,
    e_has_predictors = standata$K12 > 0,
    basehaz = basehaz[[3]][[1]]
  )

  #-----------
  # Fit model
  #-----------

  # obtain stan model code
  stanfit  <- stanmodels$simple_MS

  # specify parameters for stan to monitor
  stanpars <- c(if (standata$has_intercept01) "alpha01",
                if (standata$K01)             "beta01",
                if (standata$nvars01)         "aux01",
                if (standata$has_intercept02) "alpha02",
                if (standata$K02)             "beta02",
                if (standata$nvars02)         "aux02",
                if (standata$has_intercept12) "alpha12",
                if (standata$K12)             "beta12",
                if (standata$nvars12)         "aux12")

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
