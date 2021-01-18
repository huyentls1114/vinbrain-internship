def find_lr(trainer, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trainer.train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    trainer.optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    i = 0
    for data in trainer.train_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
        trainer.optimizer.zero_grad()
        outputs = trainer.net(inputs)
        loss = trainer.criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.dettach[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        trainer.optimizer.step()
        #Update the lr for the next step
        lr *= mult
        trainer.optimizer.param_groups[0]['lr'] = lr
        print(i, lr, avg_loss)
        i+=1
    return log_lrs, losses