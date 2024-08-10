#!/bin/bash

# Define the arrays of argument values
learning_rates=("1e-5" "5e-5" "1e-4")
masking_ratios=(".75" ".9")
epochs=("25")
batch_sizes=("100" "250")
contrast_loss_weights=("0.01" "1")
mae_loss_weights=("0" "1")

# Loop through each combination of arguments and call the original script with sbatch
for lr in "${learning_rates[@]}"
do
  for masking_ratio in "${masking_ratios[@]}"
  do
    for epoch in "${epochs[@]}"
    do
      for batch_size in "${batch_sizes[@]}"
      do
        for contrast_loss_weight in "${contrast_loss_weights[@]}"
        do
          for mae_loss_weight in "${mae_loss_weights[@]}"
          do
            # Command to submit the job
            sbatch ./original_script.sh $lr $masking_ratio $epoch $batch_size $contrast_loss_weight $mae_loss_weight
          done
        done
      done
    done
  done
done
