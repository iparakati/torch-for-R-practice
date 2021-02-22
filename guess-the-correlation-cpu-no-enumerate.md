Guess the correlation CPU tutorial
================
Isaac Parakati
February 21, 2021

Code taken from <https://torch.mlverse.org/start/guess_the_correlation/>

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

![](guess-the-correlation-cpu-no-enumerate_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
batch$y %>% as.numeric() %>% round(digits = 2)
```

    ##  [1] -0.18 -0.31  0.52 -0.65  0.94  0.75  0.59  0.48 -0.15  0.24  0.10 -0.87
    ## [13] -0.39 -0.69  0.33  0.09 -0.06  0.73 -0.19 -0.24  0.38  0.83  0.58  0.54
    ## [25]  0.26  0.31  0.29 -0.43  0.35 -0.62  0.79  0.47 -0.40  0.11  0.38 -0.16
    ## [37] -0.50  0.62 -0.13 -0.18  0.27  0.18 -0.47 -0.23  0.41 -0.42  0.55 -0.29
    ## [49]  0.56 -0.56 -0.13 -0.32  0.46 -0.12 -0.45  0.63 -0.23  0.11  0.33  0.39
    ## [61] -0.08 -0.82 -0.82  0.52

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
    ## -2.8892
    ##  -2.8839
    ##  -2.8416
    ##  -2.7818
    ##  -2.8159
    ##  -2.9318
    ##  -2.7535
    ##  -2.9824
    ##  -2.9415
    ##  -2.8536
    ##  -2.8606
    ##  -2.8808
    ##  -2.8956
    ##  -2.7244
    ##  -2.9402
    ##  -2.8365
    ##  -3.0419
    ##  -3.0426
    ##  -2.8467
    ##  -2.8137
    ##  -2.7828
    ##  -3.0827
    ##  -2.9770
    ##  -2.8405
    ##  -3.0103
    ##  -2.9116
    ##  -2.9399
    ##  -2.8925
    ##  -2.9322
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

(training_start_time <- Sys.time())
```

    ## [1] "2021-02-22 10:35:44 CST"

``` r
for (epoch in 1:num_epochs) {
  
  # don't forget to do this
  model$train()
  
  train_losses <- c()
  
  ## The original CPU tutorial executed train_batch(b) 
  ## and didn't use an iterator. I found out the train_batch(b) generates
  ## an error when done using a GPU. Using an iterator works. There's probably
  ## an issue with pulling elements out of enumerate when using a GPU. 
  ## I used the iterator approach for training, validation, and testing.
  
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

``` r
(training_end_time <- Sys.time())
```

    ## [1] "2021-02-22 11:15:06 CST"

``` r
training_end_time - training_start_time
```

    ## Time difference of 39.35921 mins

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

test_dl_iter <- dataloader_make_iter(test_dl)

for (b in enumerate(test_dl)) {
  test_batch(dataloader_next(test_dl_iter))
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

![](guess-the-correlation-cpu-no-enumerate_files/figure-gfm/r%20model_test-1.png)<!-- -->
