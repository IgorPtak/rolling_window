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

cat("\n")
