
#' ---
#' title: "Guess the correlation tutorial for regularization"
#' author: "Isaac Parakati"
#' date: "February 28, 2021"
#' output: github_document
#' ---


#' Code taken from https://torch.mlverse.org/start/guess_the_correlation/
#' This example uses a GPU.

knitr::clean_cache()

#' Get the packages

library(torch)
library(torchvision)

#' Get the dataset. I modified this step so the dataset is loaded to
#' the project's directory instead of a temp file. I want to be able to
#' repeat this analysis later on if I want to.

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
length(valid_ds)
length(test_ds)

train_ds[1]


#' Work with batches

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

length(train_dl)


batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

dim(batch$x)
dim(batch$y)

par(mfrow = c(8,8), mar = rep(0, 4))

images <- as.array(batch$x$squeeze(2))

images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})

batch$y %>% as.numeric() %>% round(digits = 2)

valid_dl <- dataloader(valid_ds, batch_size = 64)
length(valid_dl)

test_dl <- dataloader(test_ds, batch_size = 64)
length(test_dl)

#' Create the model.

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

#' Train the network.

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

#+ r model_train, cache = TRUE
num_epochs <- 10

(training_start_time <- Sys.time())

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

(training_end_time <- Sys.time())

training_end_time - training_start_time

#' Evaluate performance

#+ r model_test, cache = TRUE
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

df <- data.frame(preds = preds, targets = targets)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")



