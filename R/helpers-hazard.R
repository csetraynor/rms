# Evaluate a spline basis matrix at the specified times
#
# @param time A numeric vector.
# @param basis Info on the spline basis.
# @param integrate A logical, should the integral of the basis be returned?
# @return A two-dimensional array.
basis_matrix <- function(times, basis, integrate = FALSE) {
  out <- predict(basis, times)
  if (integrate) {
    stopifnot(inherits(basis, "mSpline"))
    class(basis) <- c("matrix", "iSpline")
    out <- predict(basis, times)
  }
  aa(out)
}


handle_basehaz_mspline <-  function(basehaz, times, status){
  
  max_t <- max(times)
  
  df <- 6L # default df for M-splines, assuming an intercept is included
  
  tt <- times[status == 1] # uncensored event times
  if (is.null(knots) && !length(tt)) {
    warning2("No observed events found in the data. Censoring times will ",
             "be used to evaluate default knot locations for splines.")
    tt <- times
  }    
  
  bknots <- c(0, max_t)
  iknots <- get_iknots(tt, df = df, iknots = NULL)
  basis  <- get_basis(tt, iknots = iknots, bknots = bknots, type = "ms")      
  nvars  <- ncol(basis)  # number of aux parameters, basis terms
  
  
  nlist(type_name = basehaz, 
        type = basehaz_for_stan(basehaz), 
        nvars, 
        iknots, 
        bknots, 
        basis,
        df = nvars,
        user_df = nvars,
        knots = if (basehaz == "bs") iknots else c(bknots[1], iknots, bknots[2]),
        bs_basis = basis)
}


# Return the integer respresentation for the baseline hazard, used by Stan
#
# @param basehaz_name A character string, the type of baseline hazard.
# @return An integer, or NA if unmatched.
basehaz_for_stan <- function(basehaz_name) {
  switch(basehaz_name, 
         weibull   = 1L, 
         bs        = 2L,
         piecewise = 3L,
         ms        = 4L,
         exp       = 5L,
         gompertz  = 6L,
         NA)
}

# Return a vector with valid names for elements in the list passed to the
# 'basehaz_ops' argument of a 'stan_jm' or 'stan_surv' call
#
# @param basehaz_name A character string, the type of baseline hazard.
# @return A character vector, or NA if unmatched.
get_ok_basehaz_ops <- function(basehaz_name) {
  switch(basehaz_name,
         weibull   = c(),
         bs        = c("df", "knots"),
         piecewise = c("df", "knots"),
         ms        = c("df", "knots"),
         NA)
}


# Return a vector with internal knots for 'x', based on evenly spaced quantiles
#
# @param x A numeric vector.
# @param df The degrees of freedom. If specified, then 'df - degree - intercept'.
#   knots are placed at evenly spaced percentiles of 'x'. If 'iknots' is 
#   specified then 'df' is ignored.
# @return A numeric vector of internal knot locations, or NULL if there are
#   no internal knots.
get_iknots <- function(x, df = 6L, degree = 3L, iknots = NULL, intercept = TRUE) {
  
  # obtain number of internal knots
  if (is.null(iknots)) {
    nk <- df - degree - intercept
  } else {
    nk <- length(iknots)
  }
  
  # validate number of internal knots
  if (nk < 0) {
    stop2("Number of internal knots cannot be negative.")
  }
  
  # obtain default knot locations if necessary
  if (is.null(iknots)) {
    iknots <- qtile(x, nq = nk + 1)  # evenly spaced percentiles
  }
  
  # return internal knot locations, ensuring they are positive
  validate_positive_scalar(iknots)
  
  return(iknots)
}


get_basis <- function(x, iknots, bknots = range(x), 
                      degree = 3, intercept = TRUE, 
                      type = c("bs", "is", "ms")) {
  type <- match.arg(type)
  if (type == "bs") {
    out <- splines::bs(x, knots = iknots, Boundary.knots = bknots,
                       degree = degree, intercept = intercept)
  } else if (type == "is") {
    out <- splines2::iSpline(x, knots = iknots, Boundary.knots = bknots,
                             degree = degree, intercept = intercept)
  } else if (type == "ms") {
    out <- splines2::mSpline(x, knots = iknots, Boundary.knots = bknots,
                             degree = degree, intercept = intercept)
  } else {
    stop2("'type' is not yet accommodated.")
  }
  out
}

# Construct a list with information about the baseline hazard
#
# @param basehaz A string specifying the type of baseline hazard
# @param basehaz_ops A named list with elements df, knots 
# @param ok_basehaz A list of admissible baseline hazards
# @param times A numeric vector with eventtimes for each individual
# @param status A numeric vector with event indicators for each individual
# @param min_t Scalar, the minimum entry time across all individuals
# @param max_t Scalar, the maximum event or censoring time across all individuals
# @return A named list with the following elements:
#   type: integer specifying the type of baseline hazard, 1L = weibull,
#     2L = b-splines, 3L = piecewise.
#   type_name: character string specifying the type of baseline hazard.
#   user_df: integer specifying the input to the df argument
#   df: integer specifying the number of parameters to use for the 
#     baseline hazard.
#   knots: the knot locations for the baseline hazard.
#   bs_basis: The basis terms for the B-splines. This is passed to Stan
#     as the "model matrix" for the baseline hazard. It is also used in
#     post-estimation when evaluating the baseline hazard for posterior
#     predictions since it contains information about the knot locations
#     for the baseline hazard (this is implemented via splines::predict.bs). 
handle_basehaz_surv <- function(basehaz, 
                                basehaz_ops, 
                                ok_basehaz     = c("weibull", "bs", "piecewise"),
                                ok_basehaz_ops = c("df", "knots"),
                                times, 
                                status,
                                min_t, max_t) {
  
  if (!basehaz %in% ok_basehaz)
    stop2("'basehaz' should be one of: ", comma(ok_basehaz))
  
  if (!all(names(basehaz_ops) %in% ok_basehaz_ops))
    stop2("'basehaz_ops' can only include: ", comma(ok_basehaz_ops))
  
  if (basehaz == "exp") {
    
    bknots <- NULL # boundary knot locations
    iknots <- NULL # internal knot locations
    basis  <- NULL # spline basis
    nvars  <- 0L   # number of aux parameters, none
    
  } else if (basehaz == "gompertz") {
    
    bknots <- NULL # boundary knot locations
    iknots <- NULL # internal knot locations
    basis  <- NULL # spline basis
    nvars  <- 1L   # number of aux parameters, Gompertz scale
    
  } else if (basehaz == "weibull") {
    
    bknots <- NULL # boundary knot locations
    iknots <- NULL # internal knot locations
    basis  <- NULL # spline basis
    nvars  <- 1L   # number of aux parameters, Weibull shape
    
  } else if (basehaz == "bs") {
    
    df    <- basehaz_ops$df
    knots <- basehaz_ops$knots
    
    if (!is.null(df) && !is.null(knots))
      stop2("Cannot specify both 'df' and 'knots' for the baseline hazard.")
    
    if (is.null(df))
      df <- 6L # default df for B-splines, assuming an intercept is included
    # NB this is ignored if the user specified knots
    
    tt <- times[status == 1] # uncensored event times
    if (is.null(knots) && !length(tt)) {
      warning2("No observed events found in the data. Censoring times will ",
               "be used to evaluate default knot locations for splines.")
      tt <- times
    }
    
    if (!is.null(knots)) {
      if (any(knots < min_t))
        stop2("'knots' cannot be placed before the earliest entry time.")
      if (any(knots > max_t))
        stop2("'knots' cannot be placed beyond the latest event time.")
    } 
    
    bknots <- c(min_t, max_t)
    iknots <- get_iknots(tt, df = df, iknots = knots)
    basis  <- get_basis(tt, iknots = iknots, bknots = bknots, type = "bs")      
    nvars  <- ncol(basis)  # number of aux parameters, basis terms
    
  } else if (basehaz == "ms") {
    
    df    <- basehaz_ops$df
    knots <- basehaz_ops$knots
    
    if (!is.null(df) && !is.null(knots)) {
      stop2("Cannot specify both 'df' and 'knots' for the baseline hazard.")
    }
    
    tt <- times[status == 1] # uncensored event times
    if (is.null(df)) {
      df <- 6L # default df for M-splines, assuming an intercept is included
      # NB this is ignored if the user specified knots
    }
    
    tt <- times[status == 1] # uncensored event times
    if (is.null(knots) && !length(tt)) {
      warning2("No observed events found in the data. Censoring times will ",
               "be used to evaluate default knot locations for splines.")
      tt <- times
    }    
    
    if (!is.null(knots)) {
      if (any(knots < min_t))
        stop2("'knots' cannot be placed before the earliest entry time.")
      if (any(knots > max_t))
        stop2("'knots' cannot be placed beyond the latest event time.")
    }
    
    bknots <- c(min_t, max_t)
    iknots <- get_iknots(tt, df = df, iknots = knots)
    basis  <- get_basis(tt, iknots = iknots, bknots = bknots, type = "ms")      
    nvars  <- ncol(basis)  # number of aux parameters, basis terms
    
  } else if (basehaz == "piecewise") {
    
    df    <- basehaz_ops$df
    knots <- basehaz_ops$knots
    
    if (!is.null(df) && !is.null(knots)) {
      stop2("Cannot specify both 'df' and 'knots' for the baseline hazard.")
    }
    
    if (is.null(df)) {
      df <- 6L # default number of segments for piecewise constant
      # NB this is ignored if the user specified knots
    }
    
    if (is.null(knots) && !length(tt)) {
      warning2("No observed events found in the data. Censoring times will ",
               "be used to evaluate default knot locations for piecewise basehaz.")
      tt <- times
    }    
    
    if (!is.null(knots)) {
      if (any(knots < min_t))
        stop2("'knots' cannot be placed before the earliest entry time.")
      if (any(knots > max_t))
        stop2("'knots' cannot be placed beyond the latest event time.")
    }
    
    bknots <- c(min_t, max_t)
    iknots <- get_iknots(tt, df = df, iknots = knots)
    basis  <- NULL               # spline basis
    nvars  <- length(iknots) + 1 # number of aux parameters, dummy indicators
    
  }  
  
  nlist(type_name = basehaz, 
        type = basehaz_for_stan(basehaz), 
        nvars, 
        iknots, 
        bknots, 
        basis,
        df = nvars,
        user_df = nvars,
        knots = if (basehaz == "bs") iknots else c(bknots[1], iknots, bknots[2]),
        bs_basis = basis)
}

# Return the name of the baseline hazard
#
# @return A character string.
get_basehaz_name <- function(x) {
  if (is.character(x)) 
    return(x)
  if (is.stansurv(x))
    return(x$basehaz$type_name)
  if (is.stanjm(x))
    return(x$survmod$basehaz$type_name)
  if (is.character(x$type_name))
    return(x$type_name)
  stop("Bug found: could not resolve basehaz name.")
}
