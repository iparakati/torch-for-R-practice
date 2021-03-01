Guess the correlation tutorial for regularization
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

root <- file.path(getwd(), "correlation")

train_ds <- guess_the_correlation_dataset(
  # where to unpack
  root = root,
  # additional preprocessing 
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # don't take all data, but just the indices we pass in
  indexes = train_indices,
  download = FALSE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = val_indices,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
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
    ## -0.45781
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

![](guess-the-correlation-gpu-bigger_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
batch$y %>% as.numeric() %>% round(digits = 2)
```

    ##  [1]  0.46  0.25  0.84  0.35 -0.44  0.55 -0.86  0.33 -0.60 -0.50  0.49  0.78
    ## [13] -0.10  0.00  0.75  0.25  0.01  0.05  0.07 -0.77  0.36  0.07  0.51  0.41
    ## [25]  0.03 -0.60  0.05  0.59 -0.66 -0.82  0.74 -0.26 -0.78  0.81 -0.33  0.39
    ## [37] -0.77 -0.75  0.72  0.47 -0.28 -0.72 -0.74 -0.32  0.67  0.85  0.26  0.04
    ## [49] -0.52 -0.61 -0.46 -0.71  0.66 -0.58  0.35 -0.51  0.50  0.50 -0.04  0.47
    ## [61]  0.22  0.34  0.32 -0.33

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
    
    self$bn1 <- nn_batch_norm2d(num_features = 32)
    self$bn2 <- nn_batch_norm2d(num_features = 64)
    self$bn3 <- nn_batch_norm2d(num_features = 128)
    self$bn4 <- nn_batch_norm1d(num_features = 128)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      self$bn1() %>%
      nnf_avg_pool2d(2) %>%
      nnf_dropout(p = 0.2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      self$bn2() %>%
      nnf_avg_pool2d(2) %>%
      nnf_dropout(p = 0.2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      self$bn3() %>%
      nnf_avg_pool2d(2) %>%
      nnf_dropout(p = 0.2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$bn4() %>%
      
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
    ##  0.2554
    ## -0.5426
    ## -0.8490
    ##  0.4856
    ## -0.7270
    ## -0.1510
    ## -1.5257
    ## -0.1678
    ##  0.4253
    ## -0.2672
    ##  0.1398
    ##  0.8793
    ## -0.5544
    ## -0.3299
    ##  1.3957
    ## -0.4985
    ## -0.9125
    ##  0.3814
    ##  0.7367
    ## -0.6382
    ## -0.0368
    ##  0.0985
    ##  0.2331
    ##  0.8415
    ##  0.6899
    ##  0.0313
    ##  0.9464
    ##  0.2838
    ##  0.5293
    ## -0.7270
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
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = "cuda"))
  
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

    ## [1] "2021-02-28 22:11:52 CST"

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
    ## Loss at epoch 1: training: 0.15016, validation: 0.05264
    ## 
    ## Loss at epoch 2: training: 0.01455, validation: 0.69606
    ## 
    ## Loss at epoch 3: training: 0.01047, validation: 1.33029
    ## 
    ## Loss at epoch 4: training: 0.00839, validation: 1.85657
    ## 
    ## Loss at epoch 5: training: 0.00808, validation: 0.19443
    ## 
    ## Loss at epoch 6: training: 0.00761, validation: 0.46781
    ## 
    ## Loss at epoch 7: training: 0.00720, validation: 0.83494
    ## 
    ## Loss at epoch 8: training: 0.00675, validation: 1.16890
    ## 
    ## Loss at epoch 9: training: 0.00625, validation: 1.55401
    ## 
    ## Loss at epoch 10: training: 0.00533, validation: 0.20523

``` r
(training_end_time <- Sys.time())
```

    ## [1] "2021-02-28 22:35:09 CST"

``` r
training_end_time - training_start_time
```

    ## Time difference of 23.2698 mins

Evaluate performance

``` r
model$eval()

test_batch <- function(b) {
  
  output <- model(b$x$to(device = "cuda"))
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = "cuda"))
  
  output <- output$to(device = "cpu")
  loss <- loss$to(device = "cpu")
  
  preds <<- c(preds, output %>% as.numeric())
  targets <<- c(targets, b$y %>% as.numeric())
  test_losses <<- c(test_losses, loss$item())
  
}

test_losses <- c()
preds <- c()
targets <- c()

test_dl_iter <- dataloader_make_iter(test_dl)

for (b in enumerate(test_dl)) {
  test_batch(dataloader_next(test_dl_iter))
}

mean(test_losses)
```

    ## [1] 0.002868379

``` r
df <- data.frame(preds = preds, targets = targets)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")
```

![](guess-the-correlation-gpu-bigger_files/figure-gfm/r%20model_test-1.png)<!-- -->
