## Benchmark: robustrolling vs slider vs RcppRoll + stable vs fast.
##
## Usage:
##   Rscript benchmarks/bench_r.R
##
## Requires: bench, slider, RcppRoll

library(robustrolling)
library(slider)
library(RcppRoll)
library(bench)

set.seed(42)

SIZES  <- c(10000L, 100000L, 1000000L)
WINDOW <- 100L
REPS   <- 10L

med_ms <- function(fns, reps = REPS) {
  vapply(fns, function(f) {
    bm <- bench::mark(f(), iterations = reps, check = FALSE, memory = FALSE)
    as.numeric(bm$median) * 1000
  }, numeric(1))
}

run_vs_libs <- function(n) {
  x <- as.double(rnorm(n))
  w <- WINDOW

  cases <- list(
    rolling_max = list(
      robustrolling = function() rolling_max(x, w),
      slider        = function() slide_dbl(x, max, .before = w - 1L, .complete = TRUE),
      RcppRoll      = function() roll_max(x, w, fill = NA)
    ),
    rolling_min = list(
      robustrolling = function() rolling_min(x, w),
      slider        = function() slide_dbl(x, min, .before = w - 1L, .complete = TRUE),
      RcppRoll      = function() roll_min(x, w, fill = NA)
    ),
    rolling_mean = list(
      robustrolling = function() rolling_mean(x, w),
      slider        = function() slide_dbl(x, mean, .before = w - 1L, .complete = TRUE),
      RcppRoll      = function() roll_mean(x, w, fill = NA)
    ),
    rolling_variance = list(
      robustrolling = function() rolling_variance(x, w),
      slider        = function() slide_dbl(x, var, .before = w - 1L, .complete = TRUE),
      RcppRoll      = function() roll_var(x, w, fill = NA)
    ),
    rolling_median = list(
      robustrolling = function() rolling_median(x, w),
      slider        = function() slide_dbl(x, median, .before = w - 1L, .complete = TRUE),
      RcppRoll      = function() roll_median(x, w, fill = NA)
    )
  )

  rows <- lapply(names(cases), function(nm) {
    meds <- med_ms(cases[[nm]])
    data.frame(
      name        = nm,
      our_ms      = meds[["robustrolling"]],
      slider_ms   = meds[["slider"]],
      RcppRoll_ms = meds[["RcppRoll"]],
      vs_slider   = meds[["slider"]]   / meds[["robustrolling"]],
      vs_RcppRoll = meds[["RcppRoll"]] / meds[["robustrolling"]],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

run_unique_metrics <- function(n) {
  x <- as.double(rnorm(n))
  y <- as.double(rnorm(n))
  w <- WINDOW

  cases <- list(
    rolling_skewness = list(robustrolling = function() rolling_skewness(x, w)),
    rolling_kurtosis = list(robustrolling = function() rolling_kurtosis(x, w)),
    rolling_cov      = list(robustrolling = function() rolling_cov(x, y, w)),
    rolling_cor      = list(robustrolling = function() rolling_cor(x, y, w))
  )

  rows <- lapply(names(cases), function(nm) {
    meds <- med_ms(cases[[nm]])
    data.frame(name = nm, our_ms = meds[["robustrolling"]],
               stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

run_stable_vs_fast <- function(n) {
  x <- as.double(rnorm(n))
  w <- WINDOW

  cases <- list(
    `mean (assume_finite)` = list(
      stable = function() rolling_mean(x, w),
      fast   = function() rolling_mean(x, w, assume_finite = TRUE)
    ),
    variance = list(
      stable = function() rolling_variance(x, w),
      fast   = function() rolling_variance(x, w, method = "fast")
    ),
    skewness = list(
      stable = function() rolling_skewness(x, w),
      fast   = function() rolling_skewness(x, w, method = "fast")
    ),
    kurtosis = list(
      stable = function() rolling_kurtosis(x, w),
      fast   = function() rolling_kurtosis(x, w, method = "fast")
    )
  )

  rows <- lapply(names(cases), function(nm) {
    meds <- med_ms(cases[[nm]])
    data.frame(
      name      = nm,
      stable_ms = meds[["stable"]],
      fast_ms   = meds[["fast"]],
      speedup   = meds[["stable"]] / meds[["fast"]],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fmt_n <- function(n) formatC(n, format = "d", big.mark = ",")
flag  <- function(v) ifelse(!is.na(v) & v >= 1.0, "x", " ")

print_vs_libs <- function(n, df) {
  cat(sprintf("\n  n = %s   window = %d   (median of %d runs)\n",
              fmt_n(n), WINDOW, REPS))
  cat(sprintf("  %-20s %14s %10s %10s %10s %10s\n",
              "Function", "robustrolling", "slider", "RcppRoll",
              "vs slider", "vs RcppRoll"))
  cat("  ", strrep("-", 78), "\n", sep = "")
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    cat(sprintf("  %-20s %10.2f ms %9.2f ms %9.2f ms %7.2fx %s %7.2fx %s\n",
                r$name, r$our_ms, r$slider_ms, r$RcppRoll_ms,
                r$vs_slider,   flag(r$vs_slider),
                r$vs_RcppRoll, flag(r$vs_RcppRoll)))
  }
}

print_unique_metrics <- function(n, df) {
  cat(sprintf("\n  n = %s   window = %d   (median of %d runs)\n",
              fmt_n(n), WINDOW, REPS))
  cat(sprintf("  %-20s %14s\n", "Function", "robustrolling"))
  cat("  ", strrep("-", 36), "\n", sep = "")
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    cat(sprintf("  %-20s %10.2f ms\n", r$name, r$our_ms))
  }
}

print_stable_vs_fast <- function(n, df) {
  cat(sprintf("\n  n = %s   window = %d   (median of %d runs)\n",
              fmt_n(n), WINDOW, REPS))
  cat(sprintf("  %-22s %12s %10s %9s\n",
              "Function", "stable", "fast", "speedup"))
  cat("  ", strrep("-", 57), "\n", sep = "")
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    cat(sprintf("  %-22s %8.2f ms  %7.2f ms  %6.2fx %s\n",
                r$name, r$stable_ms, r$fast_ms, r$speedup, flag(r$speedup)))
  }
}

## ── rolling_median dispatch sweep ─────────────────────────────────────────────

MEDIAN_WINDOWS <- c(10L, 50L, 100L, 200L, 300L, 400L, 500L,
                    600L, 700L, 800L, 1000L, 2000L, 5000L)
MEDIAN_N       <- 500000L
NAN_FRAC       <- 0.15

which_algo <- function(w, nan_hint) {
  if (nan_hint) {
    if (w <= 1500L) "FlatMedian" else "TwoHeapMedian"
  } else {
    if (w <= 600L)  "FlatMedian"
    else if (w <= 2000L) "MultisetMedian"
    else "TwoHeapMedian"
  }
}

run_median_sweep <- function() {
  set.seed(42)
  data_clean <- as.double(rnorm(MEDIAN_N))
  data_nan   <- data_clean
  data_nan[sample(MEDIAN_N, floor(MEDIAN_N * NAN_FRAC))] <- NA_real_

  rows <- lapply(MEDIAN_WINDOWS, function(w) {
    time_ms <- function(x, expect_nan) {
      bm <- bench::mark(
        rolling_median(x, w, min_periods = 0L, expect_nan = expect_nan),
        iterations = 3L, check = FALSE, memory = FALSE
      )
      as.numeric(bm$median) * 1000
    }
    data.frame(
      window          = w,
      clean_algo      = which_algo(w, FALSE),
      clean_ms        = time_ms(data_clean, FALSE),
      nan_algo        = which_algo(w, TRUE),
      nan_ms          = time_ms(data_nan,   TRUE),
      nan_default_ms  = time_ms(data_nan,   FALSE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

print_median_sweep <- function(df) {
  cat(sprintf("\n  %-8s  %-15s  %10s    %-15s  %10s  %10s\n",
              "window", "clean algo", "clean ms",
              "nan algo", "nan ms", "nan(def) ms"))
  cat("  ", strrep("-", 80), "\n", sep = "")
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    cat(sprintf("  %-8d  %-15s  %8.1f ms    %-15s  %8.1f ms  %8.1f ms\n",
                r$window,
                r$clean_algo, r$clean_ms,
                r$nan_algo,   r$nan_ms, r$nan_default_ms))
  }
  cat("\n  nan ms       = expect_nan=TRUE  (NaN-robust dispatch path)\n")
  cat("  nan(def) ms  = expect_nan=FALSE (default path on NaN-heavy data)\n")
}

cat("robustrolling vs slider vs RcppRoll\n")
cat(strrep("=", 80), "\n")
for (n in SIZES) {
  df <- run_vs_libs(n)
  print_vs_libs(n, df)
}

cat("\n\nunique metrics (no slider/RcppRoll equivalent)\n")
cat(strrep("=", 80), "\n")
for (n in SIZES) {
  df <- run_unique_metrics(n)
  print_unique_metrics(n, df)
}

cat("\n\nstable vs fast — prefix-sum acceleration\n")
cat(strrep("=", 59), "\n")
for (n in SIZES) {
  df <- run_stable_vs_fast(n)
  print_stable_vs_fast(n, df)
}

cat("\n\nrolling_median dispatch sweep  (n = 500 000, NaN fraction = 15%)\n")
cat(strrep("=", 80), "\n")
df_sweep <- run_median_sweep()
print_median_sweep(df_sweep)

cat("\n")
