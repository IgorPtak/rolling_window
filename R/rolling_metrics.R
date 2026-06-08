.check_window <- function(window_size) {
  if (!is.finite(window_size) || window_size < 1L)
    stop("window_size must be a positive finite integer.", call. = FALSE)
}

.check_min_periods <- function(min_periods, window_size) {
  mp <- suppressWarnings(as.integer(min_periods))
  if (is.na(mp) || mp < 0L || mp > window_size)
    stop("min_periods must be an integer in [0, window_size].", call. = FALSE)
  mp
}

#' @title Rolling Sample Variance
#'
#' @description
#' Computes the rolling sample variance over a numeric vector.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size} (pandas
#'   semantics). Positions with fewer non-\code{NA} values yield \code{NA}.
#' @param method \code{"stable"} (default) uses the Welford online algorithm.
#'   \code{"fast"} uses a prefix-sum approach (faster, but susceptible to
#'   catastrophic cancellation when values are large and variance is small).
#'
#' @return
#' A numeric vector with rolling sample variance values. Entries are
#' \code{NA} when fewer than \code{min_periods} non-\code{NA} observations are
#' present in the window, and \code{NaN} when variance is undefined (fewer
#' than two values).
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4))
#' rolling_variance(x, 3L)
rolling_variance <- function(x, window_size, min_periods = window_size,
                             method = "stable") {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  if (method == "fast")
    .Call("rolling_variance_fast_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
  else
    .Call("rolling_variance_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
}

#' @title Rolling Maximum
#'
#' @description
#' Computes the rolling maximum over a numeric vector using a monotonic deque.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return
#' A numeric vector with rolling maximum values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_max(x, 3L)
rolling_max <- function(x, window_size, min_periods = window_size) {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_max_c", x, as.integer(window_size), as.integer(mp),
        PACKAGE = "robustrolling")
}

#' @title Rolling Minimum
#'
#' @description
#' Computes the rolling minimum over a numeric vector using a monotonic deque.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#'
#' @return
#' A numeric vector with rolling minimum values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_min(x, 3L)
rolling_min <- function(x, window_size, min_periods = window_size) {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_min_c", x, as.integer(window_size), as.integer(mp),
        PACKAGE = "robustrolling")
}

#' @title Rolling Median
#'
#' @description
#' Computes the rolling median over a numeric vector. Automatically selects the
#' fastest algorithm based on window size and expected NaN density: FlatMedian
#' (sorted vector) for small windows, MultisetMedian (red-black tree) for
#' medium windows, or TwoHeapMedian (lazy-deletion heaps) for large windows or
#' NaN-heavy data.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#' @param expect_nan Logical. If \code{TRUE}, hints that the input may contain
#'   many \code{NA} values; switches dispatch thresholds to NaN-robust paths
#'   (avoids MultisetMedian's iterator-shift degradation). Defaults to
#'   \code{FALSE}.
#'
#' @return
#' A numeric vector with rolling median values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 3, 2, 5, 4))
#' rolling_median(x, 3L)
rolling_median <- function(x, window_size, min_periods = window_size,
                           expect_nan = FALSE) {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_median_c", x, as.integer(window_size), as.integer(mp),
        as.logical(expect_nan), PACKAGE = "robustrolling")
}

#' @title Rolling Mean
#'
#' @description
#' Computes the rolling mean over a numeric vector.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#' @param assume_finite If \code{TRUE}, assumes the input contains no
#'   \code{NA} values and uses a faster SIMD prefix-sum path. Passing
#'   \code{TRUE} when \code{NA}s are present produces incorrect results.
#'   Defaults to \code{FALSE}.
#'
#' @return A numeric vector with rolling mean values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4))
#' rolling_mean(x, 3L)
rolling_mean <- function(x, window_size, min_periods = window_size,
                         assume_finite = FALSE) {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_mean_c", x, as.integer(window_size), as.integer(mp),
        as.logical(assume_finite), PACKAGE = "robustrolling")
}

#' @title Rolling Skewness
#'
#' @description
#' Computes the rolling adjusted Fisher-Pearson skewness over a numeric vector.
#' Requires at least 3 non-\code{NA} observations per window.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#' @param method \code{"stable"} (default) uses Terriberry's online algorithm.
#'   \code{"fast"} uses a prefix-sum approach (faster, but susceptible to
#'   catastrophic cancellation when values are large and variance is small).
#'
#' @return A numeric vector with rolling skewness values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' rolling_skewness(x, 3L)
rolling_skewness <- function(x, window_size, min_periods = window_size,
                             method = "stable") {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  if (method == "fast")
    .Call("rolling_skewness_fast_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
  else
    .Call("rolling_skewness_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
}

#' @title Rolling Kurtosis
#'
#' @description
#' Computes the rolling excess kurtosis (Fisher) over a numeric vector.
#' Requires at least 4 non-\code{NA} observations per window.
#'
#' @param x A numeric vector of type double.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of non-\code{NA} observations required in
#'   a window to return a result. Defaults to \code{window_size}.
#' @param method \code{"stable"} (default) uses Terriberry's online algorithm.
#'   \code{"fast"} uses a prefix-sum approach (faster, but susceptible to
#'   catastrophic cancellation when values are large and variance is small).
#'
#' @return A numeric vector with rolling excess kurtosis values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' rolling_kurtosis(x, 4L)
rolling_kurtosis <- function(x, window_size, min_periods = window_size,
                             method = "stable") {
  x <- as.double(x)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  if (method == "fast")
    .Call("rolling_kurtosis_fast_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
  else
    .Call("rolling_kurtosis_c", x, as.integer(window_size), as.integer(mp),
          PACKAGE = "robustrolling")
}

#' @title Rolling Covariance
#'
#' @description
#' Computes the rolling sample covariance (ddof=1) between two numeric vectors.
#'
#' @param x A numeric vector of type double.
#' @param y A numeric vector of type double, same length as \code{x}.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of valid (non-\code{NA}) pairs required.
#'   Defaults to \code{window_size}.
#'
#' @return A numeric vector with rolling covariance values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' y <- as.double(c(2, 4, 6, 8, 10))
#' rolling_cov(x, y, 3L)
rolling_cov <- function(x, y, window_size, min_periods = window_size) {
  x <- as.double(x)
  y <- as.double(y)
  if (length(x) != length(y)) stop("x and y must have the same length.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_cov_c", x, y, as.integer(window_size), as.integer(mp),
        PACKAGE = "robustrolling")
}

#' @title Rolling Correlation
#'
#' @description
#' Computes the rolling Pearson correlation between two numeric vectors.
#'
#' @param x A numeric vector of type double.
#' @param y A numeric vector of type double, same length as \code{x}.
#' @param window_size Positive integer window length.
#' @param min_periods Minimum number of valid (non-\code{NA}) pairs required.
#'   Defaults to \code{window_size}.
#'
#' @return A numeric vector with rolling correlation values.
#'
#' @export
#'
#' @examples
#' x <- as.double(c(1, 2, 3, 4, 5))
#' y <- as.double(c(2, 4, 6, 8, 10))
#' rolling_cor(x, y, 3L)
rolling_cor <- function(x, y, window_size, min_periods = window_size) {
  x <- as.double(x)
  y <- as.double(y)
  if (length(x) != length(y)) stop("x and y must have the same length.", call. = FALSE)
  .check_window(window_size)
  mp <- .check_min_periods(min_periods, window_size)
  .Call("rolling_cor_c", x, y, as.integer(window_size), as.integer(mp),
        PACKAGE = "robustrolling")
}
