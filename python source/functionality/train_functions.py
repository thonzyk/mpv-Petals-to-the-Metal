# Learning Rate Schedule for Fine Tuning #
def exponential_lr(epoch,
                   start_lr=0.00001, min_lr=0.00001, max_lr=0.00005,
                   rampup_epochs=5, sustain_epochs=0,
                   exp_decay=0.8):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay ** (epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr

    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)
