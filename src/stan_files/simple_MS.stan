functions {

  /**
  * Log hazard for exponential distribution
  *
  * @param eta Vector, linear predictor
  * @return A vector
  */
  vector exponential_log_haz(vector eta) {
    return eta;
  }

  /**
  * Log hazard for Weibull distribution
  *
  * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param shape Real, Weibull shape
  * @return A vector
  */
  vector weibull_log_haz(vector eta, vector t, real shape) {
    vector[rows(eta)] res;
    res = log(shape) + (shape - 1) * log(t) + eta;
    return res;
  }

  /**
  * Log hazard for Gompertz distribution
  *
  * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param scale Real, Gompertz scale
  * @return A vector
  */
  vector gompertz_log_haz(vector eta, vector t, real scale) {
    vector[rows(eta)] res;
    res = scale * t + eta;
    return res;
  }



  /**
  * Log hazard for B-spline model
  *
  * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param coefs Vector, B-spline coefficients
  * @return A vector
  */
  vector bspline_log_haz(vector eta, matrix basis, vector coefs) {
    vector[rows(eta)] res;
    res = basis * coefs + eta;
    return res;
  }

  /**
  * Evaluate log survival or log CDF from the log hazard evaluated at
  * quadrature points and a corresponding vector of quadrature weights
  *
  * @param qwts Vector, the quadrature weights
  * @param log_hazard Vector, log hazard at the quadrature points
  * @param qnodes Integer, the number of quadrature points for each individual
  * @param N Integer, the number of individuals (ie. rows(log_hazard) / qnodes)
  * @return A vector
  */
  real quadrature_log_surv(vector qwts, vector log_hazard) {
    real res;
    res = - dot_product(qwts, exp(log_hazard)); // sum across all individuals
    return res;
  }

  vector quadrature_log_cdf(vector qwts, vector log_hazard, int qnodes, int N) {
    int M = rows(log_hazard);
    vector[M] hazard = exp(log_hazard);
    matrix[N,qnodes] qwts_mat = to_matrix(qwts,   N, qnodes);
    matrix[N,qnodes] haz_mat  = to_matrix(hazard, N, qnodes);
    vector[N] chaz = rows_dot_product(qwts_mat, haz_mat);
    vector[N] res;
    res = log(1 - exp(- chaz));
    return res;
  }

  vector quadrature_log_cdf2(vector qwts_lower, vector log_hazard_lower,
                             vector qwts_upper, vector log_hazard_upper,
                             int qnodes, int N) {
    int M = rows(log_hazard_lower);
    vector[M] hazard_lower = exp(log_hazard_lower);
    vector[M] hazard_upper = exp(log_hazard_upper);
    matrix[N,qnodes] qwts_lower_mat = to_matrix(qwts_lower,   N, qnodes);
    matrix[N,qnodes] qwts_upper_mat = to_matrix(qwts_upper,   N, qnodes);
    matrix[N,qnodes] haz_lower_mat  = to_matrix(hazard_lower, N, qnodes);
    matrix[N,qnodes] haz_upper_mat  = to_matrix(hazard_upper, N, qnodes);
    vector[N] chaz_lower = rows_dot_product(qwts_lower_mat, haz_lower_mat);
    vector[N] chaz_upper = rows_dot_product(qwts_upper_mat, haz_upper_mat);
    vector[N] surv_lower = exp(- chaz_lower);
    vector[N] surv_upper = exp(- chaz_upper);
    vector[N] res;
    res = log(surv_lower - surv_upper);
    return res;
  }


  /**
    * Log hazard for M-spline model
  *
    * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param coefs Vector, M-spline coefficients
  * @return A vector
  */
    vector mspline_log_haz(vector eta, matrix basis, vector coefs) {
      vector[rows(eta)] res;
      res = log(basis * coefs) + eta;
      return res;
    }

  /**
    * Cornish-Fisher expansion for standard normal to Student t
  *
    * See result 26.7.5 of
  * http://people.math.sfu.ca/~cbm/aands/page_949.htm
  *
    * @param z A scalar distributed standard normal
  * @param df A scalar degrees of freedom
  * @return An (approximate) Student t variate with df degrees of freedom
  */
    real CFt(real z, real df) {
      real z2 = square(z);
      real z3 = z2 * z;
      real z5 = z2 * z3;
      real z7 = z2 * z5;
      real z9 = z2 * z7;
      real df2 = square(df);
      real df3 = df2 * df;
      real df4 = df2 * df2;
      return z + (z3 + z) / (4 * df) + (5 * z5 + 16 * z3 + 3 * z) / (96 * df2)
      + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df3)
      + (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / (92160 * df4);
    }

  /**
    * Return the lower bound for the baseline hazard parameters
  *
    * @param type An integer indicating the type of baseline hazard
  * @return A real
  */
    real coefs_lb(int type) {
      real lb;
      if (type == 2) // B-splines, on log haz scale
      lb = negative_infinity();
      else if (type == 3) // piecewise constant, on log haz scale
      lb = negative_infinity();
      else
        lb = 0;
      return lb;
    }
  /**
    * Scale the primitive population level parameters based on prior information
  *
    * @param z_beta A vector of primitive parameters
  * @param prior_dist Integer, the type of prior distribution
  * @param prior_mean,prior_scale Vectors of mean and scale parameters
  *   for the prior distributions
  * @return A vector containing the population level parameters (coefficients)
  */
    vector make_beta(vector z_beta, int prior_dist, vector prior_mean,
                     vector prior_scale) {
      vector[rows(z_beta)] beta;
     // if (prior_dist == 0) beta = z_beta;
   //   else if (prior_dist == 1)
      beta = z_beta .* prior_scale + prior_mean;
      return beta;
    }

  /**
    * Log-prior for coefficients
  *
    * @param z_beta Vector of primative coefficients
  * @param prior_dist Integer, the type of prior distribution
  * @param prior_scale Real, scale for the prior distribution
  * @param prior_df Real, df for the prior distribution
  * @param global_prior_df Real, df for the prior for the global hs parameter
  * @param local Vector of hs local parameters
  * @param global Real, the global parameter
  * @param mix Vector of shrinkage parameters
  * @param one_over_lambda Real
  * @return Nothing
  */
    real beta_lp(vector z_beta, int prior_dist) {
     // if      (prior_dist == 1)
      target += normal_lpdf(z_beta | 0, 1);
      /* else prior_dist is 0 and nothing is added */
        return target();
    }

  /**
    * Log-prior for baseline hazard parameters
  *
    * @param aux_unscaled Vector (potentially of length 1) of unscaled
  *   auxiliary parameter(s)
  * @param dist Integer specifying the type of prior distribution
  * @param df Real specifying the df for the prior distribution
  * @return Nothing
  */
    real basehaz_lp(vector aux_unscaled, int dist) {
      if (dist > 0) {
        if (dist == 1)
          target += normal_lpdf(aux_unscaled | 0, 1);
            else
              target += exponential_lpdf(aux_unscaled | 1);
      }
      return target();
    }

  /**
  * Log-prior for intercept parameters
  *
  * @param gamma Real, the intercept parameter
  * @param dist Integer, the type of prior distribution
  * @param mean Real, mean of prior distribution
  * @param scale Real, scale for the prior distribution
  * @param df Real, df for the prior distribution
  * @return Nothing
  */
  real gamma_lp(real gamma, int dist, real mean, real scale) {
    if (dist == 1)  // normal
      target += normal_lpdf(gamma | mean, scale);
    //else if (dist == 2)  // student_t
     // target += student_t_lpdf(gamma | df, mean, scale);
    /* else dist is 0 and nothing is added */
    return target();
  }

  /**
    * Raise each element of x to the power of y
  *
    * @param x Vector
  * @param y Real, the power to raise to
  * @return vector
  */
    vector pow_vec(vector x, real y) {
      int N = rows(x);
      vector[N] res;
      for (n in 1:N)
        res[n] = pow(x[n], y);
      return res;
    }

  /**
    * Log survival and log CDF for exponential distribution
  *
    * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @return A vector
  */
    vector exponential_log_surv(vector eta, vector t) {
      vector[rows(eta)] res;
      res = - t .* exp(eta);
      return res;
    }

  vector exponential_log_cdf(vector eta, vector t) {
    vector[rows(eta)] res;
    res = log(1 - exp(-t .* exp(eta)));
    return res;
  }

  vector exponential_log_cdf2(vector eta, vector t_lower, vector t_upper) {
    int N = rows(eta);
    vector[N] exp_eta = exp(eta);
    vector[N] surv_lower = exp(-t_lower .* exp_eta);
    vector[N] surv_upper = exp(-t_upper .* exp_eta);
    vector[N] res;
    res = log(surv_lower - surv_upper);
    return res;
  }

  /**
    * Log survival and log CDF for Weibull distribution
  *
    * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param shape Real, Weibull shape
  * @return A vector
  */
    vector weibull_log_surv(vector eta, vector t, real shape) {
      vector[rows(eta)] res;
      res = - pow_vec(t, shape) .* exp(eta);
      return res;
    }

  vector weibull_log_cdf(vector eta, vector t, real shape) {
    vector[rows(eta)] res;
    res = log(1 - exp(- pow_vec(t, shape) .* exp(eta)));
    return res;
  }

  vector weibull_log_cdf2(vector eta, vector t_lower, vector t_upper, real shape) {
    int N = rows(eta);
    vector[N] exp_eta = exp(eta);
    vector[N] surv_lower = exp(- pow_vec(t_lower, shape) .* exp_eta);
    vector[N] surv_upper = exp(- pow_vec(t_upper, shape) .* exp_eta);
    vector[N] res;
    res = log(surv_lower - surv_upper);
    return res;
  }

  /**
    * Log survival and log CDF for Gompertz distribution
  *
    * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param scale Real, Gompertz scale
  * @return A vector
  */
    vector gompertz_log_surv(vector eta, vector t, real scale) {
      vector[rows(eta)] res;
      res = inv(scale) * -(exp(scale * t) - 1) .* exp(eta);
      return res;
    }

  vector gompertz_log_cdf(vector eta, vector t, real scale) {
    vector[rows(eta)] res;
    res = log(1 - exp(inv(scale) * -(exp(scale * t) - 1) .* exp(eta)));
    return res;
  }

  vector gompertz_log_cdf2(vector eta, vector t_lower, vector t_upper, real scale) {
    int N = rows(eta);
    real inv_scale = inv(scale);
    vector[N] exp_eta = exp(eta);
    vector[N] surv_lower = exp(inv_scale * -(exp(scale * t_lower) - 1) .* exp_eta);
    vector[N] surv_upper = exp(inv_scale * -(exp(scale * t_upper) - 1) .* exp_eta);
    vector[N] res;
    res = log(surv_lower - surv_upper);
    return res;
  }

  /**
    * Log survival and log CDF for M-spline model
  *
    * @param eta Vector, linear predictor
  * @param t Vector, event or censoring times
  * @param coefs Vector, M-spline coefficients
  * @return A vector
  */
    vector mspline_log_surv(vector eta, matrix ibasis, vector coefs) {
      vector[rows(eta)] res;
      res = - (ibasis * coefs) .* exp(eta);
      return res;
    }

  vector mspline_log_cdf(vector eta, matrix ibasis, vector coefs) {
    vector[rows(eta)] res;
    res = log(1 - exp(-(ibasis * coefs) .* exp(eta)));
    return res;
  }

  vector mspline_log_cdf2(vector eta, matrix ibasis_lower, matrix ibasis_upper, vector coefs) {
    int N = rows(eta);
    vector[N] exp_eta = exp(eta);
    vector[N] surv_lower = exp(-(ibasis_lower * coefs) .* exp_eta);
    vector[N] surv_upper = exp(-(ibasis_upper * coefs) .* exp_eta);
    vector[N] res;
    res = log(surv_lower - surv_upper);
    return res;
  }

}

data {
  //int h;
  //vector[h] type;

  int<lower=0> nevent01type2;
  int<lower=0> nevent01type4;
  int<lower=0> nevent02type3;
  int<lower=0> nevent12type4;

  int<lower=0> nrcens01type1;
  int<lower=0> nrcens01type3;
  int<lower=0> nrcens02type1;
  int<lower=0> nrcens02type2;
  int<lower=0> nrcens02type4;
  int<lower=0> nrcens12type2;


  int<lower=0> K01;
  int<lower=0> K02;
  int<lower=0> K12;



  // log crude event rate (used for centering log baseline hazard)
  real log_crude_event_rate01;
  real log_crude_event_rate02;
  real log_crude_event_rate12;

  // response and time variables
  vector[nevent01type2] t_event01type2;  // time of events
  vector[nevent01type4] t_event01type4;  // time of events
  vector[nevent02type3] t_event02type3;  // time of events
  vector[nevent12type4] t_event12type4;  // time of events


  vector[nrcens01type1] t_rcens01type1;  // time of right censoring
  vector[nrcens01type3] t_rcens01type3;  // time of right censoring
  vector[nrcens02type1] t_rcens02type1;  // time of right censoring
  vector[nrcens02type2] t_rcens02type2;  // time of right censoring
  vector[nrcens02type4] t_rcens02type4;  // time of right censoring
  vector[nrcens12type2] t_rcens12type2;  // time of right censoring

  vector[K01] x_bar01;           // predictor means
  vector[K02] x_bar02;           // predictor means
  vector[K12] x_bar12;           // predictor means


  matrix[nevent01type2, K01] x_event01type2;
  matrix[nevent01type4, K01] x_event01type4;
  matrix[nevent02type3, K02] x_event02type3;
  matrix[nevent12type4, K12] x_event12type4;


  matrix[nrcens01type1, K01] x_rcens01type1;
  matrix[nrcens01type3, K01] x_rcens01type3;
  matrix[nrcens02type1, K02] x_rcens02type1;
  matrix[nrcens02type2, K02] x_rcens02type2;
  matrix[nrcens02type4, K02] x_rcens02type4;
  matrix[nrcens12type2, K12] x_rcens12type2;


  // num. aux parameters for baseline hazard
  int<lower=0> nvars01;
  int<lower=0> nvars02;
  int<lower=0> nvars12;

  matrix[nevent01type2, nvars01] basis_event01type2;
  matrix[nevent01type4, nvars01] basis_event01type4;
  matrix[nevent02type3, nvars02] basis_event02type3;
  matrix[nevent12type4, nvars12] basis_event12type4;

  matrix[nevent01type2, nvars01] ibasis_event01type2;
  matrix[nevent01type4, nvars01] ibasis_event01type4;
  matrix[nevent02type3, nvars02] ibasis_event02type3;
  matrix[nevent12type4, nvars12] ibasis_event12type4;


  matrix[nrcens01type1, nvars01] ibasis_rcens01type1;
  matrix[nrcens01type3, nvars01] ibasis_rcens01type3;
  matrix[nrcens02type1, nvars02] ibasis_rcens02type1;
  matrix[nrcens02type2, nvars02] ibasis_rcens02type2;
  matrix[nrcens02type4, nvars02] ibasis_rcens02type4;
  matrix[nrcens12type2, nvars12] ibasis_rcens12type2;

  // baseline hazard type:
    //   1 = weibull
  //   2 = B-splines
  //   3 = piecewise
  //   4 = M-splines
  //   5 = exponential
  //   6 = gompertz
  int<lower=1,upper=7> type01;
  int<lower=1,upper=7> type02;
  int<lower=1,upper=7> type12;

  // flags
  int<lower=0,upper=1> has_intercept01; // basehaz requires intercept
  int<lower=0,upper=1> has_intercept02; // basehaz requires intercept
  int<lower=0,upper=1> has_intercept12; // basehaz requires intercept
  int<lower=0,upper=1> prior_PD;      // draw only from prior predictive dist.

  // prior family:
    //   0 = none
  //   1 = normal
  //   2 = student_t
  int<lower=0,upper=2> prior_dist01;
  int<lower=0,upper=2> prior_dist02;
  int<lower=0,upper=2> prior_dist12;

  vector[K01]           prior_mean01;
  vector[K02]           prior_mean02;
  vector[K12]           prior_mean12;
  vector<lower=0>[K01]  prior_scale01;
  vector<lower=0>[K02]  prior_scale02;
  vector<lower=0>[K12]  prior_scale12;

  // prior family:
  //   0 = none
  //   1 = normal
  //   2 = student_t
  int<lower=0,upper=2> prior_dist_for_intercept01;
  int<lower=0,upper=2> prior_dist_for_intercept02;
  int<lower=0,upper=2> prior_dist_for_intercept12;

  // hyperparameters (intercept), set to 0 if there is no prior
  real                prior_mean_for_intercept01;
  real<lower=0>       prior_scale_for_intercept01;
  real                prior_mean_for_intercept02;
  real<lower=0>       prior_scale_for_intercept02;
  real                prior_mean_for_intercept12;
  real<lower=0>       prior_scale_for_intercept12;

  // prior family:
    //   0 = none
  //   1 = normal
  //   2 = student_t
  //   3 = exponential
  int<lower=0,upper=3> prior_dist_for_aux01;
  int<lower=0,upper=3> prior_dist_for_aux02;
  int<lower=0,upper=3> prior_dist_for_aux12;

  // hyperparameters (basehaz pars), set to 0 if there is no prior
  vector<lower=0>[nvars01] prior_scale_for_aux01;
  vector<lower=0>[nvars02] prior_scale_for_aux02;
  vector<lower=0>[nvars12] prior_scale_for_aux12;

}


parameters {
  // log hazard ratios
  vector[K01] z_beta01;
  vector[K02] z_beta02;
  vector[K12] z_beta12;

  // unscaled basehaz parameters
  //   M-spline model: nvars = number of basis terms, ie. spline coefs
  vector<lower=0>[nvars01] z_coefs01;
  vector<lower=0>[nvars02] z_coefs02;
  vector<lower=0>[nvars12] z_coefs12;

  // intercept
  real gamma01[has_intercept01];
  real gamma02[has_intercept02];
  real gamma12[has_intercept12];
}

transformed parameters {

  // log hazard ratios
  vector[K01] beta01;
  vector[K02] beta02;
  vector[K12] beta12;

  // basehaz parameters
  vector[nvars01] coefs01;
  vector[nvars02] coefs02;
  vector[nvars12] coefs12;

  // define log hazard ratios
  if (K01 > 0) {
    beta01 = make_beta(z_beta01, prior_dist01, prior_mean01,
                       prior_scale01);
  }
  if (K02 > 0) {
    beta02 = make_beta(z_beta02, prior_dist02, prior_mean02,
                       prior_scale02);
  }
  if (K12 > 0) {
    beta12 = make_beta(z_beta12, prior_dist12, prior_mean12,
                       prior_scale12);
  }

  // define basehaz parameters
  if (nvars01 > 0) {
    coefs01 = z_coefs01 .* prior_scale_for_aux01;
  }
  if (nvars02 > 0) {
    coefs02 = z_coefs02 .* prior_scale_for_aux02;
  }
  if (nvars12 > 0) {
    coefs12 = z_coefs12 .* prior_scale_for_aux12;
  }
}

model {
  // linear predictor
  vector[nrcens01type1] eta_rcens01type1;  // time of right censoring
  vector[nrcens01type3] eta_rcens01type3;  // time of right censoring

  vector[nevent01type2] eta_event01type2;  // time of events
  vector[nevent01type4] eta_event01type4;  // time of events

  vector[nrcens02type1] eta_rcens02type1;  // time of right censoring
  vector[nrcens02type2] eta_rcens02type2;  // time of right censoring
  vector[nevent02type3] eta_event02type3;  // time of events
  vector[nrcens02type4] eta_rcens02type4;  // time of right censoring for type 4

  vector[nrcens12type2] eta_rcens12type2;  // time of right censoring
  vector[nevent12type4] eta_event12type4;  // time of events

  // define linear predictor
  if (K01 > 0) {
    if(nevent01type2 > 0)  eta_event01type2 = x_event01type2 * beta01;
    if(nevent01type4 > 0)  eta_event01type4 = x_event01type4 * beta01;
    if(nrcens01type1 > 0)  eta_rcens01type1 = x_rcens01type1 * beta01;
    if(nrcens01type3 > 0)  eta_rcens01type3 = x_rcens01type3 * beta01;
  } else {
    if(nevent01type2 > 0)  eta_event01type2 = rep_vector(0.0, nevent01type2);
    if(nevent01type4 > 0)  eta_event01type4 = rep_vector(0.0, nevent01type4);
    if(nrcens01type1 > 0)  eta_rcens01type1 = rep_vector(0.0, nrcens01type1);
    if(nrcens01type3 > 0)  eta_rcens01type3 = rep_vector(0.0, nrcens01type3);
  }

  if (K02 > 0) {
    if(nevent02type3 > 0)  eta_event02type3 = x_event02type3 * beta02;
    if(nrcens02type1 > 0)  eta_rcens02type1 = x_rcens02type1 * beta02;
    if(nrcens02type2 > 0)  eta_rcens02type2 = x_rcens02type2 * beta02;
    if(nrcens02type4 > 0)  eta_rcens02type4 = x_rcens02type4 * beta02;
  } else {
    if(nevent02type3 > 0)  eta_event02type3 = rep_vector(0.0, nevent02type3);
    if(nrcens02type1 > 0)  eta_rcens02type1 = rep_vector(0.0, nrcens02type1);
    if(nrcens02type2 > 0)  eta_rcens02type2 = rep_vector(0.0, nrcens02type2);
    if(nrcens02type4 > 0)  eta_rcens02type4 = rep_vector(0.0, nrcens02type4);
  }

  if (K12 > 0){
    if(nevent12type4 > 0)  eta_event12type4 = x_event12type4 * beta12;
    if(nrcens12type2 > 0)  eta_rcens12type2 = x_rcens12type2 * beta12;
  } else {
    if(nevent12type4 > 0)  eta_event12type4 =  rep_vector(0.0, nevent12type4);
    if(nrcens12type2 > 0)  eta_rcens12type2 =  rep_vector(0.0, nrcens12type2);
  }

  // add intercept
  if (has_intercept01 == 1) {
    if(nevent01type2 > 0)  eta_event01type2 += gamma01[1];
    if(nevent01type4 > 0)  eta_event01type4 += gamma01[1];
    if(nrcens01type1 > 0)  eta_rcens01type1 += gamma01[1];
    if(nrcens01type3 > 0)  eta_rcens01type3 += gamma01[1];
  }
  if (has_intercept02 == 1) {
    if(nevent02type3 > 0)  eta_event02type3 += gamma02[1];
    if(nrcens02type1 > 0)  eta_rcens02type1 += gamma02[1];
    if(nrcens02type2 > 0)  eta_rcens02type2 += gamma02[1];
    if(nrcens02type4 > 0)  eta_rcens02type4 += gamma02[1];
  }
  if (has_intercept12 == 1) {
    if(nevent12type4 > 0)  eta_event12type4 += gamma12[1];
    if(nrcens12type2 > 0)  eta_rcens12type2 += gamma12[1];
  }

  // add on log crude event rate (helps to center intercept)
    if(nevent01type2 > 0)  eta_event01type2 += log_crude_event_rate01;
    if(nevent01type4 > 0)  eta_event01type4 += log_crude_event_rate01;
    if(nrcens01type1 > 0)  eta_rcens01type1 += log_crude_event_rate01;
    if(nrcens01type3 > 0)  eta_rcens01type3 += log_crude_event_rate01;

    if(nevent02type3 > 0)  eta_event02type3 += log_crude_event_rate02;
    if(nrcens02type1 > 0)  eta_rcens02type1 += log_crude_event_rate02;
    if(nrcens02type2 > 0)  eta_rcens02type2 += log_crude_event_rate02;
    if(nrcens02type4 > 0)  eta_rcens02type4 += log_crude_event_rate02;

    if(nevent12type4 > 0)  eta_event12type4 += log_crude_event_rate12;
    if(nrcens12type2 > 0)  eta_rcens12type2 += log_crude_event_rate12;

    if(type01 == 1){ // weibull
      real shape01 = coefs01[1];
      if (nrcens01type1 > 0) target +=  weibull_log_surv(eta_rcens01type1, t_rcens01type1, shape01);
      if (nevent01type2 > 0) target +=  weibull_log_haz(eta_event01type2, t_event01type2, shape01);
      if (nevent01type2 > 0) target +=  weibull_log_surv(eta_event01type2, t_event01type2, shape01);
      if (nrcens01type3 > 0) target +=  weibull_log_surv(eta_rcens01type3, t_rcens01type3, shape01);
      if (nevent01type4 > 0) target +=  weibull_log_haz(eta_event01type4, t_event01type4, shape01);
      if (nevent01type4 > 0) target +=  weibull_log_surv(eta_event01type4, t_event01type4, shape01);
    } else if(type01 == 4){ // M-splines
      if (nrcens01type1 > 0) target +=  mspline_log_surv(eta_rcens01type1, ibasis_rcens01type1, coefs01);
      if (nevent01type2 > 0) target +=  mspline_log_haz(eta_event01type2,  basis_event01type2, coefs01);
      if (nevent01type2 > 0) target +=  mspline_log_surv(eta_event01type2,  ibasis_event01type2, coefs01);
      if (nrcens01type3 > 0) target +=  mspline_log_surv(eta_rcens01type3, ibasis_rcens01type3, coefs01);
      if (nevent01type4 > 0) target +=  mspline_log_haz(eta_event01type4,  basis_event01type4, coefs01);
      if (nevent01type4 > 0) target +=  mspline_log_surv(eta_event01type4,  ibasis_event01type4, coefs01);
    } else if (type02 == 05) { // exponential model
      if (nrcens01type1 > 0) target +=  exponential_log_surv(eta_rcens01type1, t_rcens01type1);
      if (nevent01type2 > 0) target +=  exponential_log_haz(eta_event01type2);
      if (nevent01type2 > 0) target +=  exponential_log_surv(eta_event01type2, t_event01type2);
      if (nrcens01type3 > 0) target +=  exponential_log_surv(eta_rcens01type3, t_rcens01type3);
      if (nevent01type4 > 0) target +=  exponential_log_haz(eta_event01type4);
      if (nevent01type4 > 0) target +=  exponential_log_surv(eta_event01type4, t_event01type4);
  } else {
      reject("Bug found: invalid baseline hazard 01 (without quadrature).");
    }

  if(type02 == 1){ // weibull
    real shape02 = coefs02[1];
    if (nrcens02type1 > 0) target +=  weibull_log_surv(eta_rcens02type1, t_rcens02type1, shape02);
    if (nrcens02type2 > 0) target +=  weibull_log_surv(eta_rcens02type2, t_rcens02type2, shape02);
    if (nevent02type3 > 0) target +=  weibull_log_haz(eta_event02type3, t_event02type3, shape02);
    if (nevent02type3 > 0) target +=  weibull_log_surv(eta_event02type3, t_event02type3, shape02);
    if (nrcens02type4 > 0) target +=  weibull_log_surv(eta_rcens02type4, t_rcens02type4, shape02);
  } else if(type02 == 4){ // M-splines
    if (nrcens02type1 > 0) target +=  mspline_log_surv(eta_rcens02type1, ibasis_rcens02type1, coefs02);
    if (nrcens02type2 > 0) target +=  mspline_log_surv(eta_rcens02type2, ibasis_rcens02type2, coefs02);
    if (nevent02type3 > 0) target +=  mspline_log_haz(eta_event02type3, basis_event02type3, coefs02);
    if (nevent02type3 > 0) target +=  mspline_log_surv(eta_event02type3, ibasis_event02type3, coefs02);
    if (nrcens02type4 > 0) target +=  mspline_log_surv(eta_rcens02type4,  ibasis_rcens02type4, coefs02);
  } else if (type02 == 05) { // exponential model
  if (nrcens02type1 > 0) target +=  exponential_log_surv(eta_rcens02type1, t_rcens02type1);
    if (nrcens02type2 > 0) target +=  exponential_log_surv(eta_rcens02type2, t_rcens02type2);
    if (nevent02type3 > 0) target +=  exponential_log_haz(eta_event02type3);
    if (nevent02type3 > 0) target +=  exponential_log_surv(eta_event02type3, t_event02type3);
    if (nrcens02type4 > 0) target +=  exponential_log_surv(eta_rcens02type4, t_rcens02type4);
  } else {
    reject("Bug found: invalid baseline hazard 02 (without quadrature).");
  }

  if(type12 == 1){ // weibull
    real shape12 = coefs12[1];
    if (nrcens12type2 > 0) target +=  weibull_log_surv(eta_rcens12type2, t_rcens12type2, shape12);
    if (nevent12type4 > 0) target +=  weibull_log_haz(eta_event12type4, t_event12type4, shape12);
    if (nevent12type4 > 0) target +=  weibull_log_surv(eta_event12type4, t_event12type4, shape12);
  } else if(type12 == 4){ // M-splines
    if (nrcens12type2 > 0) target +=  mspline_log_surv(eta_rcens12type2, ibasis_rcens12type2, coefs12);
    if (nevent12type4 > 0) target +=  mspline_log_haz(eta_event12type4,  basis_event12type4, coefs12);
    if (nevent12type4 > 0) target +=  mspline_log_surv(eta_event12type4,  ibasis_event12type4, coefs12);
  } else if (type12 == 05) { // exponential model
  if (nrcens12type2 > 0) target +=  exponential_log_surv(eta_rcens12type2, t_rcens12type2);
    if (nevent12type4 > 0) target +=  exponential_log_haz(eta_event12type4);
    if (nevent12type4 > 0) target +=  exponential_log_surv(eta_event12type4, t_event12type4);
  } else {
    reject("Bug found: invalid baseline hazard 12 (without quadrature).");
  }


  //-------- log priors

  // log priors for coefficients
  if (K01 > 0) {
    real dummy = beta_lp(z_beta01, prior_dist01);
  }

  if (K02 > 0) {
    real dummy = beta_lp(z_beta02, prior_dist02);
  }

  if (K12 > 0) {
    real dummy = beta_lp(z_beta12, prior_dist12);
  }

  // log prior for intercept
  if (has_intercept01 == 1) {
    real dummy = gamma_lp(gamma01[1], prior_dist_for_intercept01,
                          prior_mean_for_intercept01, prior_scale_for_intercept01);
  }

  if (has_intercept02 == 1) {
    real dummy = gamma_lp(gamma02[1], prior_dist_for_intercept02,
                          prior_mean_for_intercept02, prior_scale_for_intercept02);
  }

  if (has_intercept12 == 1) {
    real dummy = gamma_lp(gamma12[1], prior_dist_for_intercept12,
                          prior_mean_for_intercept12, prior_scale_for_intercept12);
  }
  // log priors for baseline hazard parameters
  if (nvars01 > 0) {
    real dummy = basehaz_lp(z_coefs01, prior_dist_for_aux01);
  }

  if (nvars02 > 0) {
    real dummy = basehaz_lp(z_coefs02, prior_dist_for_aux02);
  }

  if (nvars12 > 0) {
    real dummy = basehaz_lp(z_coefs12, prior_dist_for_aux12);
  }
}


generated quantities {
  // tranformations to adjust for:
    //   - centering of covariates and
  //   - centering of baseline hazard around the crude event rate

  real alpha01[has_intercept01]; // transformed intercept
  real alpha02[has_intercept02]; // transformed intercept
  real alpha12[has_intercept12]; // transformed intercept

  vector[nvars01] aux01;              // transformed baseline hazard parameters
  vector[nvars02] aux02;              // transformed baseline hazard parameters
  vector[nvars12] aux12;              // transformed baseline hazard parameters


  if (type01 == 4) { // m-splines
    aux01 = coefs01 * exp(log_crude_event_rate01 - dot_product(x_bar01, beta01));
  } else { // exp, weibull, gompertz
    aux01 = coefs01;
    alpha01[1] = log_crude_event_rate01 - dot_product(x_bar01, beta01) + gamma01[1];
  }
  if (type02 == 4) { // m-splines
    aux02 = coefs02 * exp(log_crude_event_rate02 - dot_product(x_bar02, beta02)); // m-splines
  } else { // exp, weibull, gompertz
    aux02 = coefs02;
    alpha02[1] = log_crude_event_rate02 - dot_product(x_bar02, beta02) + gamma02[1];
  }
  if (type12 == 4) { // m-splines
    aux12 = coefs12 * exp(log_crude_event_rate12 - dot_product(x_bar12, beta12)); // m-splines
  } else { // exp, weibull, gompertz
    aux12 = coefs12;
    alpha12[1] = log_crude_event_rate12 - dot_product(x_bar12, beta12) + gamma12[1];
  }
}

