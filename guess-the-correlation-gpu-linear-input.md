Guess the correlation tutorial for linear inputs
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
  transform = function(img) crop_axes(img) %>% torch_flatten(),
  # don't take all data, but just the indices we pass in
  indexes = train_indices,
  download = FALSE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% torch_flatten(),
  indexes = val_indices,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% torch_flatten(),
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
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ##  0.9999
    ## ... [the output was truncated (use n=-1 to disable)]
    ## [ CPUFloatType{16900} ]
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

    ## [1]    64 16900

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

![](guess-the-correlation-gpu-linear-input_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
batch$y %>% as.numeric() %>% round(digits = 2)
```

    ##  [1]  0.59  0.40 -0.35  0.07  0.09 -0.15  0.35 -0.21  0.44 -0.19 -0.22  0.68
    ## [13]  0.28  0.83 -0.67  0.26  0.35  0.62  0.58  0.03  0.75  0.21  0.04 -0.35
    ## [25]  0.93 -0.04  0.01  0.38  0.29 -0.14 -0.55  0.47  0.63  0.58  0.31  0.10
    ## [37] -0.38  0.53 -0.81  0.75  0.03  0.10  0.67  0.17 -0.44  0.39 -0.66 -0.07
    ## [49]  0.39 -0.19  0.61 -0.40  0.89 -0.16 -0.32 -0.16 -0.33 -0.55  0.29  0.25
    ## [61] -0.36  0.67 -0.46  0.11

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
    
    self$fc1 <- nn_linear(in_features = 130 * 130, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 256)
    self$fc3 <- nn_linear(in_features = 256, out_features = 1)
    
  },
  
  forward = function(x) {
    
    
    x %>% 
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2() %>%
      nnf_relu() %>%
      
      self$fc3() 
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
    ##  5.1926
    ##   0.6758
    ##   2.2072
    ##   1.3977
    ##   1.9829
    ##   2.3058
    ##   0.0033
    ##   1.8258
    ##   0.9488
    ##   3.3743
    ##   1.2755
    ##   1.1685
    ##  -0.4162
    ##   1.1861
    ##   3.6677
    ##   1.9597
    ##   4.9967
    ##   4.0826
    ##   0.8928
    ##   3.2274
    ##   6.2029
    ##   2.8717
    ##  -0.4884
    ##   1.3333
    ##   0.4179
    ##   1.2422
    ##   3.6225
    ##  -0.0387
    ##   3.2433
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

    ## [1] "2021-02-28 21:30:59 CST"

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
    ## Loss at epoch 1: training: 0.53061, validation: 0.22396
    ## 
    ## Loss at epoch 2: training: 0.14334, validation: 0.07502
    ## 
    ## Loss at epoch 3: training: 0.07136, validation: 0.04994
    ## 
    ## Loss at epoch 4: training: 0.03985, validation: 0.02068
    ## 
    ## Loss at epoch 5: training: 0.03923, validation: 0.07782
    ## 
    ## Loss at epoch 6: training: 0.02985, validation: 0.08823
    ## 
    ## Loss at epoch 7: training: 0.02325, validation: 0.03306
    ## 
    ## Loss at epoch 8: training: 0.02134, validation: 0.05385
    ## 
    ## Loss at epoch 9: training: 0.02483, validation: 0.01086
    ## 
    ## Loss at epoch 10: training: 0.01548, validation: 0.01007

``` r
(training_end_time <- Sys.time())
```

    ## [1] "2021-02-28 21:45:14 CST"

``` r
training_end_time - training_start_time
```

    ## Time difference of 14.25417 mins

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

    ## [1] 0.01009272

``` r
df <- data.frame(preds = preds, targets = targets)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")
```

![](guess-the-correlation-gpu-linear-input_files/figure-gfm/r%20model_test-1.png)<!-- -->

``` r
# worse fit, though fast training
```
