Guess the correlation tutorial for classification
================
Isaac Parakati
February 28, 2021

Code taken from <https://torch.mlverse.org/start/guess_the_correlation/>
This example uses a GPU.

``` r
knitr::clean_cache()
```

    ## NULL

Get the packages

``` r
library(torch)
library(torchvision)
```

Get the dataset. I modified this step so the dataset is loaded to the
project’s directory instead of a temp file. I want to be able to repeat
this analysis later on if I want to.

``` r
## remotes::install_github("mlverse/torchdatasets")
library(torchdatasets)

train_indices <- 1:10000
val_indices <- 10001:15000
test_indices <- 15001:20000

add_channel_dim <- function(img) img$unsqueeze(1)
crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, 
                                          width = 130)
binarize <- function(tensor) torch_round(torch_abs(tensor))

root <- file.path(getwd(), "correlation")

train_ds <- guess_the_correlation_dataset(
  # where to unpack
  root = root,
  # additional preprocessing 
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # binarize target data
  target_transform = binarize,
  # don't take all data, but just the indices we pass in
  indexes = train_indices,
  download = FALSE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # binarize target data
  target_transform = binarize,
  indexes = val_indices,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # binarize target data
  target_transform = binarize,
  indexes = test_indices,
  download = FALSE
)

length(train_ds)
```

    ## [1] 10000

``` r
length(valid_ds)
```

    ## [1] 5000

``` r
length(test_ds)
```

    ## [1] 5000

``` r
train_ds[1]
```

    ## $x
    ## torch_tensor
    ## (1,.,.) = 
    ##  Columns 1 to 9  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
    ## ... [the output was truncated (use n=-1 to disable)]
    ## [ CPUFloatType{1,130,130} ]
    ## 
    ## $y
    ## torch_tensor
    ## 0
    ## [ CPUFloatType{} ]
    ## 
    ## $id
    ## [1] "arjskzyc"

Work with batches

``` r
train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

length(train_dl)
```

    ## [1] 157

``` r
batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

dim(batch$x)
```

    ## [1]  64   1 130 130

``` r
dim(batch$y)
```

    ## [1] 64

``` r
par(mfrow = c(8,8), mar = rep(0, 4))

images <- as.array(batch$x$squeeze(2))

images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})
```

![](guess-the-correlation-gpu-classify_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
batch$y %>% as.numeric() %>% round(digits = 2)
```

    ##  [1] 0 0 1 0 0 1 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0
    ## [39] 0 0 0 1 0 1 1 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 1 0 1 0

``` r
valid_dl <- dataloader(valid_ds, batch_size = 64)
length(valid_dl)
```

    ## [1] 79

``` r
test_dl <- dataloader(test_ds, batch_size = 64)
length(test_dl)
```

    ## [1] 79

Create the model.

``` r
torch_manual_seed(777)

net <- nn_module(
  
  "corr-cnn",
  
  initialize = function() {
    
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2()
  }
)

model <- net()

## need to move model parameters to GPU
model <- model$to(device = "cuda")

## testing model's output from a GPU
model(batch$x$to(device = "cuda"))
```

    ## torch_tensor
    ## 0.01 *
    ## -2.9203
    ##  -2.9768
    ##  -3.0900
    ##  -2.9248
    ##  -2.8599
    ##  -2.8964
    ##  -2.9364
    ##  -3.0511
    ##  -2.9233
    ##  -2.9334
    ##  -2.8690
    ##  -2.9784
    ##  -3.1091
    ##  -2.8357
    ##  -2.9905
    ##  -3.1016
    ##  -2.9113
    ##  -2.8658
    ##  -2.9664
    ##  -3.0294
    ##  -2.8656
    ##  -2.8263
    ##  -2.7438
    ##  -2.9003
    ##  -2.9372
    ##  -2.8215
    ##  -2.8455
    ##  -2.9079
    ##  -3.0489
    ## ... [the output was truncated (use n=-1 to disable)]
    ## [ CUDAFloatType{64,1} ]

Train the network.

``` r
optimizer <- optim_adam(model$parameters)


train_batch <- function(b) {
  
  optimizer$zero_grad()
  
  # get predictions after moving data to GPU
  output <- model(b$x$to(device = "cuda"))
  
  # calculate loss after moving data to GPU
  loss <- nnf_binary_cross_entropy_with_logits(output, b$y$unsqueeze(2)$to(device = "cuda"))
  
  # have gradients get calculated        
  loss$backward()
  
  # have gradients get applied
  optimizer$step()
  
  loss$item()
  
}

valid_batch <- function(b) {
  
  output <- model(b$x$to(device = "cuda"))
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = "cuda"))
  loss$item()
  
}
```

``` r
num_epochs <- 10

(training_start_time <- Sys.time())
```

    ## [1] "2021-02-28 21:46:55 CST"

``` r
for (epoch in 1:num_epochs) {
  
  # don't forget to do this
  model$train()
  
  train_losses <- c()
  
  ## The original GPU tutorial executed train_batch(b) 
  ## and didn't use an iterator. I found out the train_batch(b) generates
  ## an error when done using a GPU. Using an iterator works. There's probably
  ## an issue with pulling elements out of enumerate. I made use the iterator
  ## approach for training, validation, and testing.
  train_dl_iter <- dataloader_make_iter(train_dl)
  
  for (b in enumerate(train_dl)) {
    
    loss <- train_batch(dataloader_next(train_dl_iter))
    train_losses <- c(train_losses, loss)
    
  }
  
  # don't forget to do this either
  model$eval()
  
  valid_losses <- c()
  
  valid_dl_iter <- dataloader_make_iter(valid_dl)
  
  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(dataloader_next(valid_dl_iter))
    valid_losses <- c(valid_losses, loss)
    
  }
  
  cat(sprintf("\nLoss at epoch %d: training: %1.5f, validation: %1.5f\n", epoch, mean(train_losses), mean(valid_losses)))
  
}
```

    ## 
    ## Loss at epoch 1: training: 0.53699, validation: 3.86134
    ## 
    ## Loss at epoch 2: training: 0.33510, validation: 12.67120
    ## 
    ## Loss at epoch 3: training: 0.22370, validation: 17.09427
    ## 
    ## Loss at epoch 4: training: 0.17169, validation: 20.40405
    ## 
    ## Loss at epoch 5: training: 0.13679, validation: 32.11780
    ## 
    ## Loss at epoch 6: training: 0.11670, validation: 34.85388
    ## 
    ## Loss at epoch 7: training: 0.09212, validation: 55.67270
    ## 
    ## Loss at epoch 8: training: 0.07350, validation: 78.15397
    ## 
    ## Loss at epoch 9: training: 0.06118, validation: 86.72958
    ## 
    ## Loss at epoch 10: training: 0.04447, validation: 128.74101

``` r
(training_end_time <- Sys.time())
```

    ## [1] "2021-02-28 22:08:42 CST"

``` r
training_end_time - training_start_time
```

    ## Time difference of 21.78899 mins

Evaluate performance

``` r
model$eval()

batch_size <- 64

test_batch <- function(b) {
  
  output <- model(b$x$to(device = "cuda"))
  loss <- nnf_binary_cross_entropy_with_logits(output, b$y$unsqueeze(2)$to(device = "cuda"))
  
  loss <- loss$to(device = "cpu")
  output <- output$to(device = "cpu")

  
  prob <- torch_sigmoid(output)
  class_ <- torch_round(prob)
  correct <- (torch_sum(class_ == b$y$unsqueeze(2)))
  test_accuracy <<- c(test_accuracy, correct$item()/batch_size)
  test_losses <<- c(test_losses, loss$item())
  
}

test_losses <- c()
test_accuracy <- c()

test_dl_iter <- dataloader_make_iter(test_dl)

for (b in enumerate(test_dl)) {
  test_batch(dataloader_next(test_dl_iter))
}

mean(test_losses)
```

    ## [1] 0.2349497

``` r
mean(test_accuracy)
```

    ## [1] 0.9171282
