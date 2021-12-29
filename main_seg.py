import logging
import os
import torch
from utils import exp_utils as exp_utils
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader


def val(net,diceloss,val_dataloader,epoch,my_logging):
    '''
    验证
    :param model：模型
    :param val_dataloader：验证的数据集
    :param epoch：训练时的epoch
    return best_dice,平均dice,平均asd,平均loss
    '''
    net= net.eval()

    dice_total = []
    loss_total = []

    pbar = tqdm(total=len(val_dataloader), unit='batches')
    pbar.set_description('Epoch: [{}/{} ({:.0f}%)]'.format(epoch+1,cf.num_epochs, 100. *(epoch+1)/cf.num_epochs))
    my_logging.info('Epoch: [{}/{} ({:.0f}%)]'.format(epoch+1,cf.num_epochs, 100. *(epoch+1)/cf.num_epochs))

    with torch.no_grad():
        for i, (batch_data) in enumerate(val_dataloader):
            inputs, targets,dict_data = batch_data[0],batch_data[1],batch_data[2]
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = net(inputs)

            loss = diceloss(outputs, targets)
            dice = 1.0 - loss

            loss_total.append(loss.item())
            dice_total.append(dice.item())

            pbarinfo = 'Val[Loss: {:.6f} Aver_loss: {:.6f} Dice: {:.6f} Aver_Dice: {:.6f} lr: {}]'.format(loss.item(),np.mean(loss_total), dice.item(),np.mean(dice_total),optimizer.param_groups[0]['lr'])
            file_name = dict_data[0]
            my_logging.info(str(i+1)+'/'+str(len(val_dataloader))+' '+file_name+' '+pbarinfo)
            pbar.update()
            pbar.set_postfix_str(pbarinfo)

        pbar.close()

        aver_dice = np.mean(dice_total)
        aver_loss = np.mean(loss_total)

        return aver_dice,aver_loss


def train(net, optimizer, diceloss,train_dataloader,val_dataloader,lr_scheduler,my_tensorboard,my_logging):
    '''
    训练
    :param model：模型
    :param optimizer：优化器
    :param train_dataloader：训练的数据集
    :param val_dataloader：验证的数据集
    :param args：命令行解析对象
    :param lr_scheduler：学习率调整策略
    :param diceloss：损失函数
    '''

    iters = len(train_dataloader)
    val_dices = []
    trained_epochs = []
    for epoch in range(args.start_epoch, cf.num_epochs):
        net.train()
        if cf.lr_adjustment_strategy == 'consine':
            exp_utils.adjust_lr_cos_seg(optimizer, epoch, cf.num_epochs, 5, cf.init_lr)

        pbar = tqdm(total=len(train_dataloader), unit='batches')
        pbar.set_description('Epoch: [{}/{} ({:.0f}%)]'.format(epoch+1,cf.num_epochs, 100. *(epoch+1)/cf.num_epochs))
        my_logging.info('Epoch: [{}/{} ({:.0f}%)]'.format(epoch+1,cf.num_epochs, 100. *(epoch+1)/cf.num_epochs))

        train_dice_total = []
        train_loss_total = []

        for i, (batch_data) in enumerate(train_dataloader):
            inputs, targets = batch_data[0],batch_data[1]
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = net(inputs)
            loss = diceloss(outputs, targets)
            train_dice = 1.0 - loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            train_dice_total.append(train_dice.item())
            train_loss_total.append(loss.item())
           
            pbarinfo = 'Train[Loss: {:.6f} Aver_loss: {:.6f} Dice: {:.6f} Aver_Dice: {:.6f} lr: {}]'.format(loss.item(), np.mean(train_loss_total),train_dice.item(),np.mean(train_dice_total),optimizer.param_groups[0]['lr'])
            
            my_logging.info(str(i+1)+'/'+str(iters)+' '+pbarinfo)
            pbar.update()
            pbar.set_postfix_str(pbarinfo)

        pbar.close()

        val_aver_dice, val_aver_loss = val(net,diceloss,val_dataloader,epoch,my_logging)
        val_dices.append(val_aver_dice)
        print('best_dice is {:.4f} in epoch {}'.format(max(val_dices),val_dices.index(max(val_dices))+1))
        my_logging.info('best_dice is {:.4f} in epoch {}'.format(max(val_dices),val_dices.index(max(val_dices))+1))


        # 保存最好的前5个模型
        trained_epochs.append(epoch+1)
        exp_utils.model_select_save_seg(cf, val_dices, net, epoch+1, trained_epochs)

        # 保存最新的模型
        exp_utils.save_model_seg(cf, net, optimizer, epoch, val_aver_dice)

        # 指标可视化
        my_tensorboard.add_scalars('train', {'loss':np.mean(train_loss_total),'dice':np.mean(train_dice_total)}, (epoch + 1))
        my_tensorboard.add_scalars('val', {'loss':val_aver_loss,'dice':val_aver_dice}, (epoch + 1))
        my_tensorboard.add_scalar('lr', optimizer.param_groups[0]['lr'], (epoch + 1))

        if cf.lr_adjustment_strategy != 'cosine':
            lr_scheduler.step()

def getArgs():
    '''
    命令行解析对象
    :param version：版本，也是本次训练产生的所有文件的根目录
    :param description：版本的描述
    :param epochs：max epochs
    :param arch：网络模型
    :param batch_size：batch size
    :param resume：是否恢复训练
    :param save_model_path：保存模型的路径
    :param log_dir：日志路径
    return 命令行解析对象
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument('--gpus', default="2")

    parse.add_argument("--start_epoch", type=int, default=0)
    parse.add_argument("--resume", type=bool, default=False,help='Whether to resume training')

    parse.add_argument('--net', type=str, default='UNet',help='UNet')
    parse.add_argument("--batch_size", type=int, default=1)

    parse.add_argument('--use_stored_settings', action='store_true', default=False,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parse.add_argument('--exp_source', type=str, default='./experiments/exp_rnpc_seg',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    parse.add_argument('--stored_source', type=str, default='',
                        help='specifies, from which source experiment to load configs, data_loader and model.')

    args = parse.parse_args()

    return args


if __name__ =="__main__":

    args = getArgs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    assert use_gpu

    cf = exp_utils.prep_exp_ddp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=True)

    my_tensorboard = exp_utils.get_tensorboard_seg(cf)
    my_logging = exp_utils.getLog_seg(cf)

    model = exp_utils.import_module('model', cf.model_path)
    net = model.Net()
    net = nn.DataParallel(net)
    net = net.cuda()

    dataset = exp_utils.import_module('dataset', cf.dataset_path)

    train_data = dataset.DataCustom('train',cf)
    train_dataloaders = DataLoader(train_data, batch_size=args.batch_size,shuffle=True,num_workers=8,pin_memory=True)

    val_data = dataset.DataCustom('val',cf)
    val_dataloaders = DataLoader(val_data, batch_size=1,shuffle=False,num_workers=8,pin_memory=True)

    diceloss = exp_utils.getLoss_seg()
    optimizer = exp_utils.getOptim_seg(cf, net)
    lr_scheduler = exp_utils.getLr_scheduler_seg(cf, optimizer)
    

    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s' % \
          (args.net, cf.num_epochs, args.batch_size))
    my_logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' % \
          (args.net, cf.num_epochs, args.batch_size))
    print('**************************')


    if args.resume:
        args.start_epoch, best_dice = exp_utils.load_checkpoint_seg(cf,net, optimizer=None, is_fine_tuning=False)
        if best_dice is not None:
            print('checkpoint eopch {},best_dice {}'.format(args.start_epoch,best_dice))
            my_logging.info('checkpoint eopch {},best_dice {}'.format(args.start_epoch,best_dice))
        else:
            print('checkpoint不存在，从头开始训练')
            my_logging.info('checkpoint不存在，从头开始训练')


    train(net, optimizer, diceloss,train_dataloaders,val_dataloaders,lr_scheduler,my_tensorboard,my_logging)
