

# Return the spline basis for the given type of baseline hazard.
# 
# @param times A numeric vector of times at which to evaluate the basis.
# @param basehaz A list with info about the baseline hazard, returned by a 
#   call to 'handle_basehaz'.
# @param integrate A logical, specifying whether to calculate the integral of
#   the specified basis.
# @return A matrix.
make_basis <- function(times, basehaz, integrate = FALSE) {
  N <- length(times)
  K <- basehaz$nvars
  if (!N) { # times is NULL or empty vector
    return(matrix(0, 0, K))
  } 
  switch(basehaz$type_name,
         "exp"       = matrix(0, N, K), # dud matrix for Stan
         "weibull"   = matrix(0, N, K), # dud matrix for Stan
         "gompertz"  = matrix(0, N, K), # dud matrix for Stan
         "ms"        = basis_matrix(times, basis = basehaz$basis, integrate = integrate),
         "bs"        = basis_matrix(times, basis = basehaz$basis),
         "piecewise" = dummy_matrix(times, knots = basehaz$knots),
         stop2("Bug found: type of baseline hazard unknown."))
}


# Deal with priors
#
# @param prior A list
# @param nvars An integer indicating the number of variables
# @param default_scale Default value to use to scale if not specified by user
# @param link String naming the link function.
# @param ok_dists A list of admissible distributions.
handle_glm_prior <- function(prior, nvars, default_scale, link,
                             ok_dists = nlist("normal", student_t = "t", 
                                              "cauchy", "hs", "hs_plus", 
                                              "laplace", "lasso", "product_normal")) {
  if (!length(prior))
    return(list(prior_dist = 0L, prior_mean = as.array(rep(0, nvars)),
                prior_scale = as.array(rep(1, nvars)),
                prior_df = as.array(rep(1, nvars)), prior_dist_name = NA,
                global_prior_scale = 0, global_prior_df = 0,
                slab_df = 0, slab_scale = 0,
                prior_autoscale = FALSE))
  
  if (!is.list(prior)) 
    stop(sQuote(deparse(substitute(prior))), " should be a named list")
  
  prior_dist_name <- prior$dist
  prior_scale <- prior$scale
  prior_mean <- prior$location
  prior_df <- prior$df
  prior_mean[is.na(prior_mean)] <- 0
  prior_df[is.na(prior_df)] <- 1
  global_prior_scale <- 0
  global_prior_df <- 0
  slab_df <- 0
  slab_scale <- 0
  if (!prior_dist_name %in% unlist(ok_dists)) {
    stop("The prior distribution should be one of ",
         paste(names(ok_dists), collapse = ", "))
  } else if (prior_dist_name %in% 
             c("normal", "t", "cauchy", "laplace", "lasso", "product_normal")) {
    if (prior_dist_name == "normal") prior_dist <- 1L
    else if (prior_dist_name == "t") prior_dist <- 2L
    else if (prior_dist_name == "laplace") prior_dist <- 5L
    else if (prior_dist_name == "lasso") prior_dist <- 6L
    else if (prior_dist_name == "product_normal") prior_dist <- 7L
    prior_scale <- set_prior_scale(prior_scale, default = default_scale, 
                                   link = link)
  } else if (prior_dist_name %in% c("hs", "hs_plus")) {
    prior_dist <- ifelse(prior_dist_name == "hs", 3L, 4L)
    global_prior_scale <- prior$global_scale
    global_prior_df <- prior$global_df
    slab_df <- prior$slab_df
    slab_scale <- prior$slab_scale
  } else if (prior_dist_name %in% "exponential") {
    prior_dist <- 3L # only used for scale parameters so 3 not a conflict with 3 for hs
  }
  
  prior_df <- maybe_broadcast(prior_df, nvars)
  prior_df <- as.array(pmin(.Machine$double.xmax, prior_df))
  prior_mean <- maybe_broadcast(prior_mean, nvars)
  prior_mean <- as.array(prior_mean)
  prior_scale <- maybe_broadcast(prior_scale, nvars)
  
  nlist(prior_dist, 
        prior_mean, 
        prior_scale, 
        prior_df, 
        prior_dist_name, 
        global_prior_scale,
        global_prior_df,
        slab_df,
        slab_scale,
        prior_autoscale = isTRUE(prior$autoscale))
}

# Return the default scale parameter for 'prior_aux'.
#
# @param basehaz A list with info about the baseline hazard; see 'handle_basehaz'.
# @return A scalar.
get_default_aux_scale <- function(basehaz) {
  nm <- get_basehaz_name(basehaz)
  if (nm %in% c("weibull", "gompertz")) 2 else 20
}

# Check and set scale parameters for priors
#
# @param scale Value of scale parameter (can be NULL).
# @param default Default value to use if \code{scale} is NULL.
# @param link String naming the link function or NULL.
# @return If a probit link is being used, \code{scale} (or \code{default} if
#   \code{scale} is NULL) is scaled by \code{dnorm(0) / dlogis(0)}. Otherwise
#   either \code{scale} or \code{default} is returned.
set_prior_scale <- function(scale, default, link) {
  stopifnot(is.numeric(default), is.character(link) || is.null(link))
  if (is.null(scale)) 
    scale <- default
  if (isTRUE(link == "probit"))
    scale <- scale * dnorm(0) / dlogis(0)
  
  return(scale)
}
# Maybe broadcast 
#
# @param x A vector or scalar.
# @param n Number of replications to possibly make. 
# @return If \code{x} has no length the \code{0} replicated \code{n} times is
#   returned. If \code{x} has length 1, the \code{x} replicated \code{n} times
#   is returned. Otherwise \code{x} itself is returned.
maybe_broadcast <- function(x, n) {
  if (!length(x)) {
    rep(0, times = n)
  } else if (length(x) == 1L) {
    rep(x, times = n)
  } else {
    x
  }
}


# Check formula object
#
# @param formula The user input to the formula argument.
# @param needs_response A logical; if TRUE then formula must contain a LHS.
validate_formula <- function(formula, needs_response = TRUE) {
  
  if (!inherits(formula, "formula")) {
    stop2("'formula' must be a formula.")
  }
  
  if (needs_response) {
    len <- length(formula)
    if (len < 3) {
      stop2("'formula' must contain a response.")
    }
  }
  as.formula(formula)
}

# Extract LHS of a formula
#
# @param x A formula object
# @param as_formula Logical. If TRUE then the result is reformulated.
lhs <- function(x, as_formula = FALSE) {
  len <- length(x)
  if (len == 3L) {
    out <- x[[2L]]
  } else {
    out <- NULL
  }
  out
}

# Reformulate as LHS of a formula
#
# @param x A character string or expression object
# @param as_formula Logical. If TRUE then the result is reformulated.
reformulate_lhs <- function(x) {
  #x <- deparse(x, 500L)
  x <- formula(substitute(LHS ~ 1, list(LHS = x)))
  x
}

# Extract RHS of a formula
#
# @param x A formula object
# @param as_formula Logical. If TRUE then the result is reformulated.
rhs <- function(x, as_formula = FALSE) {
  len <- length(x)
  if (len == 3L) {
    out <- x[[3L]]
  } else {
    out <- x[[2L]]
  }
  out
}

# Reformulate as RHS of a formula
#
# @param x A formula object
# @param as_formula Logical. If TRUE then the result is reformulated.
reformulate_rhs <- function(x) {
  #x <- deparse(x, 500L)
  x <- formula(substitute(~ RHS, list(RHS = x)))
  x
}

# Check object is a Surv object with a valid type
#
# @param x A Surv object; the LHS of a formula evaluated in a data frame environment.
# @param ok_types A character vector giving the allowed types of Surv object.
validate_surv <- function(x, ok_types = c("right", "counting",
                                          "interval", "interval2")) {
  if (!inherits(x, "Surv"))
    stop2("LHS of 'formula' must be a 'Surv' object.")
  if (!attr(x, "type") %in% ok_types)
    stop2("Surv object type must be one of: ", comma(ok_types))
  x
}

# Return a data frame with NAs excluded
#
# @param formula The parsed model formula.
# @param data The user specified data frame.
make_model_data <- function(formula, aux_formula, data, cens, aux_cens ) {
  
  data <- data[data[aux_formula$dvar] == aux_cens, ]
  data <- data[data[formula$dvar] == cens, ]
  mf <- model.frame(formula$tf_form, data, na.action = na.pass)
  include <- apply(mf, 1L, function(row) !any(is.na(row)))
  
  
  data[include, , drop = FALSE]
}

# Parse the model formula
#
# @param formula The user input to the formula argument.
# @param data The user input to the data argument (i.e. a data frame).
parse_formula <- function(formula, data) {
  
  formula <- validate_formula(formula, needs_response = TRUE)
  
  lhs      <- lhs(formula) # full LHS of formula
  lhs_form <- reformulate_lhs(lhs)
  
  rhs        <- rhs(formula)         # RHS as expression
  rhs_form   <- reformulate_rhs(rhs) # RHS as formula
  rhs_terms  <- terms(rhs_form, specials = "tde")
  rhs_vars   <- rownames(attr(rhs_terms, "factors"))
  
  allvars <- all.vars(formula)
  allvars_form <- reformulate(allvars)
  
  surv <- eval(lhs, envir = data) # Surv object
  surv <- validate_surv(surv)
  type <- attr(surv, "type")
  
  if (type == "right") {
    tvar_beg <- NULL
    tvar_end <- as.character(lhs[[2L]])
    dvar     <- as.character(lhs[[3L]])
    min_t    <- 0
    max_t    <- max(surv[, "time"])
  } else if (type == "counting") {
    tvar_beg <- as.character(lhs[[2L]])
    tvar_end <- as.character(lhs[[3L]])
    dvar     <- as.character(lhs[[4L]])
    min_t    <- min(surv[, "start"])
    max_t    <- max(surv[, "stop"])
  } else if (type == "interval") {
    tvar_beg <- NULL
    tvar_end <- as.character(lhs[[2L]])
    dvar     <- as.character(lhs[[4L]])
    min_t    <- 0
    max_t    <-  max(surv[, c("time1", "time2")])
  } else if (type == "interval2") {
    tvar_beg <- NULL
    tvar_end <- as.character(lhs[[2L]])
    dvar     <- as.character(lhs[[3L]])
    min_t    <- 0
    max_t    <- max(surv[, c("time1", "time2")])
  }
  
  sel <- attr(rhs_terms, "specials")$tde
  
  if (!is.null(sel)) { # model has tde
    
    # # replace 'tde(x, ...)' in formula with 'x'
    # tde_oldvars <- rhs_vars
    # tde_newvars <- sapply(tde_oldvars, function(oldvar) {
    #   if (oldvar %in% rhs_vars[sel]) {
    #     tde <- function(newvar, ...) { # define tde function locally
    #       safe_deparse(substitute(newvar)) 
    #     }
    #     eval(parse(text = oldvar))
    #   } else oldvar
    # }, USE.NAMES = FALSE)
    # term_labels <- attr(rhs_terms, "term.labels")
    # for (i in sel) {
    #   sel_terms <- which(attr(rhs_terms, "factors")[i, ] > 0)
    #   for (j in sel_terms) {
    #     term_labels[j] <- gsub(tde_oldvars[i], 
    #                            tde_newvars[i], 
    #                            term_labels[j], 
    #                            fixed = TRUE)
    #   }
    # }
    # tf_form <- reformulate(term_labels, response = lhs)
    # 
    # # extract 'tde(x, ...)' from formula and construct 'bs(times, ...)'
    # tde_terms <- lapply(rhs_vars[sel], function(x) {
    #   tde <- function(vn, ...) { # define tde function locally
    #     dots <- list(...)
    #     ok_args <- c("df")
    #     if (!isTRUE(all(names(dots) %in% ok_args)))
    #       stop2("Invalid argument to 'tde' function. ",
    #             "Valid arguments are: ", comma(ok_args))
    #     df <- if (is.null(dots$df)) 3 else dots$df
    #     degree <- 3
    #     if (df == 3) {
    #       dots[["knots"]] <- numeric(0)
    #     } else {
    #       dx <- (max_t - min_t) / (df - degree + 1)
    #       dots[["knots"]] <- seq(min_t + dx, max_t - dx, dx)
    #     }
    #     dots[["Boundary.knots"]] <- c(min_t, max_t) 
    #     sub("^list\\(", "bs\\(times__, ", safe_deparse(dots))
    #   }
    #   tde_calls <- eval(parse(text = x))
    #   sel_terms <- which(attr(rhs_terms, "factors")[x, ] > 0)
    #   new_calls <- sapply(seq_along(sel_terms), function(j) {
    #     paste0(term_labels[sel_terms[j]], ":", tde_calls)
    #   })
    #   nlist(tde_calls, new_calls)
    # })
    # td_basis <- fetch(tde_terms, "tde_calls")
    # new_calls <- fetch_(tde_terms, "new_calls")
    # td_form <- reformulate(new_calls, response = NULL, intercept = FALSE)
    
    ## not implemented yet
    
  } else { # model doesn't have tde
    tf_form  <- formula
    td_form  <- NULL
    td_basis <- NULL
  }
  
  nlist(formula,
        lhs,
        rhs,
        lhs_form,
        rhs_form,
        tf_form,
        td_form,
        td_basis,
        fe_form = rhs_form, # no re terms accommodated yet
        re_form = NULL,     # no re terms accommodated yet
        allvars,
        allvars_form,
        tvar_beg,
        tvar_end,
        dvar,
        surv_type = attr(surv, "type"))
}

# Return the model frame
#
# @param formula The parsed model formula.
# @param data The model data frame.
make_model_frame <- function(formula, data, check_constant = TRUE) {
  
  # construct terms object from formula 
  Terms <- terms(formula)
  
  # construct model frame
  mf <- model.frame(Terms, data)
  
  # check no constant vars NOT IMPLEMENTED
  # if (check_constant)
  #   mf <- check_constant_vars(mf)

  # check for terms
  mt <- attr(mf, "terms")
  if (is.empty.model(mt)) 
    stop2("No intercept or predictors specified.")
  
  nlist(mf, mt)
}



# Return the response vector (time) for estimation
#
# @param model_frame The model frame.
# @param type The type of time variable to return:
#   "beg": the entry time for the row in the survival data,
#   "end": the exit  time for the row in the survival data,
#   "gap": the difference between entry and exit times,
#   "upp": if the row involved interval censoring, then the exit time
#          would have been the lower limit of the interval, and "upp" 
#          is the upper limit of the interval.
# @return A numeric vector.
make_t <- function(model_frame, type = c("beg", "end", "gap", "upp")) {
  
  type <- match.arg(type)
  resp <- if (survival::is.Surv(model_frame)) 
    model_frame else model.response(model_frame)
  surv <- attr(resp, "type")
  err  <- paste0("Bug found: cannot handle '", surv, "' Surv objects.")
  
  t_beg <- switch(surv,
                  "right"     = rep(0, nrow(model_frame)),
                  "interval"  = rep(0, nrow(model_frame)),
                  "interval2" = rep(0, nrow(model_frame)),
                  "counting"  = as.vector(resp[, "start"]),
                  stop(err))
  
  t_end <- switch(surv,
                  "right"     = as.vector(resp[, "time"]),
                  "interval"  = as.vector(resp[, "time1"]),
                  "interval2" = as.vector(resp[, "time1"]),
                  "counting"  = as.vector(resp[, "stop"]),
                  stop(err))
  
  t_upp <- switch(surv,
                  "right"     = rep(NaN, nrow(model_frame)),
                  "counting"  = rep(NaN, nrow(model_frame)),
                  "interval"  = as.vector(resp[, "time2"]),
                  "interval2" = as.vector(resp[, "time2"]),
                  stop(err))
  
  switch(type,
         "beg" = t_beg,
         "end" = t_end,
         "gap" = t_end - t_beg,
         "upp" = t_upp,
         stop("Bug found: cannot handle specified 'type'."))
}


# Return the response vector (status indicator)
#
# @param model_frame The model frame.
# @return A numeric vector.
make_d <- function(model_frame) {
  
  resp <- if (survival::is.Surv(model_frame)) 
    model_frame else model.response(model_frame)
  surv <- attr(resp, "type")
  err  <- paste0("Bug found: cannot handle '", surv, "' Surv objects.")
  
  switch(surv,
         "right"     = as.vector(resp[, "status"]),
         "interval"  = as.vector(resp[, "status"]),
         "interval2" = as.vector(resp[, "status"]),
         "counting"  = as.vector(resp[, "status"]),
         stop(err))
}


# Identify whether the type of baseline hazard requires an intercept in
# the linear predictor (NB splines incorporate the intercept into the basis).
#
# @param basehaz A list with info about the baseline hazard; see 'handle_basehaz'.
# @return A Logical.
has_intercept <- function(basehaz) {
  nm <- get_basehaz_name(basehaz)
  (nm %in% c("exp", "weibull", "gompertz"))
}



# Return the fe predictor matrix for estimation
#
# @param formula The parsed model formula.
# @param model_frame The model frame.
# @return A named list with the following elements:
#   x: the fe model matrix, not centred and without intercept.
#   x_bar: the column means of the model matrix.
#   x_centered: the fe model matrix, centered.
#   N,K: number of rows (observations) and columns (predictors) in the
#     fixed effects model matrix
make_x <- function(formula, model_frame, xlevs = NULL, check_constant = TRUE) {
  
  # uncentred predictor matrix, without intercept
  x <- model.matrix(formula, model_frame, xlevs = xlevs)
  x <- drop_intercept(x)
  
  # column means of predictor matrix
  x_bar <- aa(colMeans(x))
  
  # centered predictor matrix
  x_centered <- sweep(x, 2, x_bar, FUN = "-")
  
  # identify any column of x with < 2 unique values (empty interaction levels)
  sel <- (apply(x, 2L, n_distinct) < 2)
  if (check_constant && any(sel)) {
    cols <- paste(colnames(x)[sel], collapse = ", ")
    stop2("Cannot deal with empty interaction levels found in columns: ", cols)
  }
  
  nlist(x, x_centered, x_bar, N = NROW(x), K = NCOL(x))
}
