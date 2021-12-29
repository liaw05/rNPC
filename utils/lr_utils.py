import math


def adjust_lr_cos(optimizer, epoch, max_epochs=300, warm_epoch=10, lr_init=1e-3):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    if epoch < warm_epoch:
        cur_lr = lr_init * (epoch*(1.0-0.1)/warm_epoch + 0.1)
    else:
        cur_lr = lr_init * 0.5 * (1. + math.cos(math.pi * (epoch-warm_epoch)/ (max_epochs-warm_epoch)))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = lr_init
        else:
            param_group['lr'] = cur_lr