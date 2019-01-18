stan_surv <- function(formula01,
                      formula12,
                      formula02,
                      data, 
                      basehaz         = "ms", 
                      basehaz_ops, 
                      qnodes          = 15, 
                      prior           = normal(), 
                      prior_intercept = normal(),
                      prior_aux       = normal(), 
                      prior_smooth    = exponential(autoscale = FALSE), 
                      prior_PD        = FALSE,
                      algorithm       = c("sampling", "meanfield", "fullrank"),
                      adapt_delta     = 0.95, ...) {
  #-----------------------------
  # Pre-processing of arguments
  #-----------------------------
  
  if (!requireNamespace("survival"))
    stop("the 'survival' package must be installed to use this function.")
  
  if (missing(basehaz_ops)) 
    basehaz_ops <- NULL
  if (missing(data) || !inherits(data, "data.frame"))
    stop("'data' must be a data frame.")
  
  dots      <- list(...)
  algorithm <- match.arg(algorithm)
  
  formula01   <- parse_formula(formula01, data)
  formula12   <- parse_formula(formula12, data)
  formula02   <- parse_formula(formula02, data)
  data      <- make_model_data(formula$tf_form, data) # row subsetting etc.
  #----------------
  # Construct data
  #----------------
  
  #----- model frame stuff
  
  mf_stuff01 <- make_model_frame(formula01$tf_form, data)
  mf_stuff12 <- make_model_frame(formula12$tf_form, data)
  mf_stuff02 <- make_model_frame(formula02$tf_form, data)
  
  mf01 <- mf_stuff01$mf # model frame
  mf12 <- mf_stuff12$mf # model frame
  mf02 <- mf_stuff02$mf # model frame
  
  mt01 <- mf_stuff01$mt # model terms
  mt12 <- mf_stuff12$mt # model terms
  mt02 <- mf_stuff02$mt # model terms
  
  #----- dimensions and response vectors
  
  # entry and exit times for each row of data
  t_beg01 <- make_t(mf01, type = "beg") # entry time
  t_end12 <- make_t(mf12, type = "end") # exit  time
  t_upp02 <- make_t(mf02, type = "upp") # upper time for interval censoring
  # ensure no event or censoring times are zero (leads to degenerate
  # estimate for log hazard for most baseline hazards, due to log(0))
  check1 <- any(t_end <= 0, na.rm = TRUE)
  check2 <- any(t_upp <= 0, na.rm = TRUE)
  if (check1 || check2)
    stop2("All event and censoring times must be greater than 0.")
  
}
  
  