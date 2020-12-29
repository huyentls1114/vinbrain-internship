
lr_schedule = {
    "class":CyclicLR,
    "optimizer_args":{
        "epochs":num_epochs,
        "max_lr":0.3,
        "steps_per_epoch":int(train_length/batch_size)
    }
}
