
lr_schedule = {
    "class":CyclicLR,
    "optimizer_args":{
        "epochs":num_epochs,
        "max_lr":0.3,
        "steps_per_epoch":steps_per_epoch_train(dataset, split_train_val, batch_size)
    }
}
