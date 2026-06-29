# The CHAP model Architecture in R

## Load the rtorch package
suppressPackageStartupMessages({
  library(torch)
})


## Define CNN and BILSTM modules based on the CHAP architecture. 
## The Conv2dSame module implements "same" padding behavior for convolutional layers, ensuring that the output spatial dimensions are the same as the input when stride is 1. 
## The CNNModel defines the convolutional layers, and the CNNBiLSTMModel combines the CNN with a bidirectional LSTM for sequence modeling.

Conv2dSame <- nn_module(
  "Conv2dSame",
  initialize = function(
    in_channels,
    out_channels,
    kernel_size,
    stride = c(1L, 1L),
    dilation = c(1L, 1L),
    groups = 1L,
    bias = TRUE
  ) {
    as_pair_int <- function(v) {
      v <- as.integer(v)
      if (length(v) == 1L) return(c(v, v))
      if (length(v) == 2L) return(v)
      stop("Expected a scalar or length-2 vector.")
    }

    kernel_size <- as.integer(kernel_size)
    stride <- as_pair_int(stride)
    dilation <- as_pair_int(dilation)
    groups <- as.integer(groups)

    if (length(kernel_size) != 2L) stop("kernel_size must contain height and width values")
    
    if (in_channels %% groups != 0L) stop("in_channels must be divisible by groups")

    self$kernel_size <- kernel_size
    self$stride <- stride
    self$dilation <- dilation
    self$groups <- groups
    self$padding <- c(0L, 0L)

    self$weight <- nn_parameter(torch_empty(
      as.integer(out_channels),
      as.integer(in_channels / groups),
      kernel_size[[1]],
      kernel_size[[2]]
    ))
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))

    if (bias) {
      fan_in <- as.integer(in_channels / groups) * kernel_size[[1]] * kernel_size[[2]]
      bound <- 1 / sqrt(fan_in)
      self$bias <- nn_parameter(torch_empty(as.integer(out_channels)))
      nn_init_uniform_(self$bias, -bound, bound)
    } else {
      self$bias <- NULL
    }
  },
  calc_same_pad = function(input_size, kernel_size, stride, dilation) {
    as.integer(max(
      (ceiling(input_size / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size,
      0
    ))
  },
  forward = function(x) {
    input_height <- as.integer(x$size(3))
    input_width <- as.integer(x$size(4))

    pad_h <- self$calc_same_pad(
      input_height,
      self$kernel_size[[1]],
      self$stride[[1]],
      self$dilation[[1]]
    )

    pad_w <- self$calc_same_pad(
      input_width,
      self$kernel_size[[2]],
      self$stride[[2]],
      self$dilation[[2]]
    )

    if (pad_h > 0L || pad_w > 0L) {
      x <- nnf_pad(
        x,
        pad = as.integer(c(
          floor(pad_w / 2),
          pad_w - floor(pad_w / 2),
          floor(pad_h / 2),
          pad_h - floor(pad_h / 2)
        )),
        mode = "constant",
        value = 0
      )
    }

    nnf_conv2d(
      x,
      self$weight,
      bias = self$bias,
      stride = self$stride,
      padding = self$padding,
      dilation = self$dilation,
      groups = self$groups
    )
  }
)

CNNModel <- nn_module(
  "CNNModel",
  initialize = function(amp_factor = 1L) {
    self$amp_factor <- as.integer(amp_factor)

    self$conv1 <- Conv2dSame(
      in_channels = 1L,
      out_channels = 32L * self$amp_factor,
      kernel_size = c(5L, 3L),
      stride = c(2L, 1L)
    )
    self$conv2 <- Conv2dSame(
      in_channels = 32L * self$amp_factor,
      out_channels = 64L * self$amp_factor,
      kernel_size = c(5L, 1L),
      stride = c(2L, 1L)
    )
    self$conv3 <- Conv2dSame(
      in_channels = 64L * self$amp_factor,
      out_channels = 128L * self$amp_factor,
      kernel_size = c(5L, 1L),
      stride = c(2L, 1L)
    )
    self$conv4 <- Conv2dSame(
      in_channels = 128L * self$amp_factor,
      out_channels = 256L * self$amp_factor,
      kernel_size = c(5L, 1L),
      stride = c(2L, 1L)
    )
    self$conv5 <- Conv2dSame(
      in_channels = 256L * self$amp_factor,
      out_channels = 256L * self$amp_factor,
      kernel_size = c(5L, 1L),
      stride = c(2L, 1L)
    )

    self$fc <- nn_linear(
      in_features = 256L * self$amp_factor * 3L * 4L,
      out_features = 256L * self$amp_factor
    )
  },
  forward = function(x) {
    x <- nnf_relu(self$conv1(x))
    x <- nnf_relu(self$conv2(x))
    x <- nnf_relu(self$conv3(x))
    x <- nnf_relu(self$conv4(x))
    x <- nnf_relu(self$conv5(x))
    x <- x$reshape(c(x$size(1), -1))
    x <- self$fc(x)
    x
  }
)

CNNBiLSTMModel <- nn_module(
  "CNNBiLSTMModel",
  initialize = function(amp_factor, bi_lstm_win_size, num_classes = 2L) {
    self$amp_factor <- as.integer(amp_factor)
    self$bi_lstm_win_size <- as.integer(bi_lstm_win_size)
    self$num_classes <- as.integer(num_classes)
    self$hidden_size <- 128L

    if (self$num_classes != 2L) {
      stop("The Python CHAP implementation is binary-only (num_classes == 2).")
    }

    self$cnn_model <- CNNModel(amp_factor = self$amp_factor)

    self$bil_lstm <- nn_lstm(
      input_size = 256L * self$amp_factor,
      hidden_size = self$hidden_size,
      bidirectional = TRUE,
      batch_first = TRUE
    )

    self$fc_bilstm <- nn_linear(
      in_features = 2L * self$hidden_size,
      out_features = 1L
    )
  },
  forward = function(x) {
    cnn_output <- self$cnn_model(x)
    cnn_output <- cnn_output$view(c(
      -1L,
      self$bi_lstm_win_size,
      256L * self$amp_factor
    ))
    lstm_output <- self$bil_lstm(cnn_output)[[1]]
    fc_output <- self$fc_bilstm(lstm_output)
    fc_output$view(c(-1L, self$bi_lstm_win_size))
  }
)