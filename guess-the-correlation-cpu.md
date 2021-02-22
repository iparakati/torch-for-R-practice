Guess the correlation CPU tutorial
================
Isaac Parakati
February 21, 2021

Code taken from <https://torch.mlverse.org/start/guess_the_correlation/>
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

![](guess-the-correlation-cpu_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
batch$y %>% as.numeric() %>% round(digits = 2)
```

    ##  [1]  0.13  0.15 -0.61 -0.38  0.48 -0.21 -0.01  0.21  0.61  0.41 -0.36 -0.11
    ## [13] -0.28 -0.73  0.79 -0.63  0.41 -0.50 -0.56 -0.73  0.43  0.16  0.02 -0.13
    ## [25]  0.42  0.04 -0.27 -0.40  0.54  0.40  0.21  0.46 -0.46  0.55 -0.24  0.70
    ## [37]  0.18 -0.56  0.47 -0.37 -0.07 -0.32  0.67 -0.18  0.75 -0.42  0.80 -0.03
    ## [49] -0.67 -0.46  0.19  0.35  0.32 -0.30 -0.36 -0.10 -0.75 -0.40  0.06 -0.08
    ## [61]  0.26 -0.57  0.07  0.02

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

model(batch$x)
```

    ## torch_tensor
    ## 0.01 *
    ## -2.8521
    ##  -2.9868
    ##  -2.6846
    ##  -2.8730
    ##  -2.8848
    ##  -2.8769
    ##  -2.9812
    ##  -2.8435
    ##  -2.9203
    ##  -2.8870
    ##  -2.8189
    ##  -2.9743
    ##  -2.9884
    ##  -2.8120
    ##  -2.9381
    ##  -2.8874
    ##  -2.9715
    ##  -2.8666
    ##  -2.8628
    ##  -2.8949
    ##  -2.7987
    ##  -3.0809
    ##  -2.8667
    ##  -2.9093
    ##  -2.9452
    ##  -2.8090
    ##  -2.8354
    ##  -2.6924
    ##  -2.9558
    ## ... [the output was truncated (use n=-1 to disable)]
    ## [ CPUFloatType{64,1} ]

Train the network.

``` r
optimizer <- optim_adam(model$parameters)

train_batch <- function(b) {
  
  optimizer$zero_grad()
  
  # get predictions
  output <- model(b$x)
  
  # calculate loss
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  
  # have gradients get calculated        
  loss$backward()
  
  # have gradients get applied
  optimizer$step()
  
  loss$item()
  
}

valid_batch <- function(b) {
  
  output <- model(b$x)
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  loss$item()
  
}
```

``` r
num_epochs <- 10

for (epoch in 1:num_epochs) {
  
  # don't forget to do this
  model$train()
  
  train_losses <- c()
  
  for (b in enumerate(train_dl)) {
    
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
    
  }
  
  # don't forget to do this either
  model$eval()
  
  valid_losses <- c()
  
  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(b)
    valid_losses <- c(valid_losses, loss)
    
  }
  
  cat(sprintf("\nLoss at epoch %d: training: %1.5f, validation: %1.5f\n", epoch, mean(train_losses), mean(valid_losses)))
  
}
```

    ## 
    ## Loss at epoch 1: training: 0.05826, validation: 0.02098
    ## 
    ## Loss at epoch 2: training: 0.01120, validation: 0.00774
    ## 
    ## Loss at epoch 3: training: 0.00616, validation: 0.00506
    ## 
    ## Loss at epoch 4: training: 0.00431, validation: 0.00374
    ## 
    ## Loss at epoch 5: training: 0.00287, validation: 0.00292
    ## 
    ## Loss at epoch 6: training: 0.00199, validation: 0.00211
    ## 
    ## Loss at epoch 7: training: 0.00163, validation: 0.00159
    ## 
    ## Loss at epoch 8: training: 0.00120, validation: 0.00140
    ## 
    ## Loss at epoch 9: training: 0.00099, validation: 0.00130
    ## 
    ## Loss at epoch 10: training: 0.00088, validation: 0.00127

Evaluate performance

``` r
model$eval()

test_batch <- function(b) {
  
  output <- model(b$x)
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  
  preds <<- c(preds, output %>% as.numeric())
  targets <<- c(targets, b$y %>% as.numeric())
  test_losses <<- c(test_losses, loss$item())
  
}

test_losses <- c()
preds <- c()
targets <- c()

for (b in enumerate(test_dl)) {
  test_batch(b)
}

mean(test_losses)
```

    ## [1] 0.001383291

``` r
df <- data.frame(preds = preds, targets = targets)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")
```

![](guess-the-correlation-cpu_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
