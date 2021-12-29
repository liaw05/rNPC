import argparse
import os
import torch
from utils import evaluator_utils as evaluator_utils
from utils import exp_utils as exp_utils
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from scipy.ndimage.interpolation import zoom
from utils.evaluator_utils import Dice_test, ASD_test
        

def test(infer,net,test_dataloaders,my_logging,save_predict=True):
    net.eval()
    dice_total = []
    asd_total = []
    pbar = tqdm(total=len(test_dataloaders), unit='batches')
    with torch.no_grad():
        for i, (batch_data) in enumerate(test_dataloaders):
            inputs,targets,dict_data,origin_shape,crop_shape,zxy,spacing,origin,direction = batch_data[0],batch_data[1],batch_data[2],batch_data[3],batch_data[4],batch_data[5],batch_data[6],batch_data[7],batch_data[8]
            file_name = dict_data[0].replace('.nii.gz','')
           
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = net(inputs)

            # reize到crop时的大小进行计算
            

            outputs = torch.squeeze(outputs).cpu().numpy()
            targets = torch.squeeze(targets).cpu().numpy()
            assert len(targets.shape) == 3

            origin_shape = (origin_shape[0].item(),origin_shape[1].item(),origin_shape[2].item())
            crop_shape = (crop_shape[0].item(),crop_shape[1].item(),crop_shape[2].item())

            shape_scale = np.array(crop_shape) / targets.shape
            outputs = zoom(outputs, shape_scale, order=0)
            targets = zoom(targets, shape_scale, order=0)

            spacing = (spacing[0].item(),spacing[1].item(),spacing[2].item())
            origin = (origin[0].item(),origin[1].item(),origin[2].item())
            direction = (direction[0].item(),direction[1].item(),direction[2].item(),
                            direction[3].item(),direction[4].item(),direction[5].item(),
                            direction[6].item(),direction[7].item(),direction[8].item())
          
            outputs, targets = torch.from_numpy(outputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))
            
            dice = Dice_test(outputs, targets)
            asd = ASD_test(outputs, targets,spacing)
            dice_total.append(dice.item())
            asd_total.append(asd)
            pbarinfo = 'Test[Dice: {:.6f} Aver_Dice: {:.6f} ASD: {:.6f} Aver_ASD {:.6f}]'.format(dice.item(),np.mean(dice_total),asd,np.nanmean(asd_total))
            
            my_logging.info(file_name+' '+pbarinfo)
            pbar.update()
            pbar.set_postfix_str(pbarinfo)
            if save_predict ==True:
                evaluator_utils.save_nii_seg(cf,infer,outputs, targets,dict_data,origin_shape,spacing,origin,direction,zxy)
               
        pbar.close()
        print('****',infer,'****')
        print('Aver_Dice {:.6f} Aver_ASD {:.6f}'.format(np.mean(dice_total),np.nanmean(asd_total)))
        my_logging.info('Aver_Dice {:.6f} Aver_ASD {:.6f}'.format(np.mean(dice_total),np.nanmean(asd_total)))



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

    cf = exp_utils.prep_exp_ddp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=False)

    my_logging = exp_utils.getLog_seg(cf)

    # model = exp_utils.getModel_seg(args)

    model = exp_utils.import_module('model', cf.model_path)
    net = model.Net()
    net = nn.DataParallel(net)

    net = net.cuda()
    dataset = exp_utils.import_module('dataset', cf.dataset_path)
    

    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s' % \
          (args.net, cf.num_epochs, args.batch_size))
    my_logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' % \
          (args.net, cf.num_epochs, args.batch_size))
    print('**************************')


    net.load_state_dict(torch.load(cf.checkpoint_path))  # 载入训练好的模型

    for infer in ['val','test','SYSUCC_test','test_ex_anony']:
        infer_data = dataset.DataCustom(infer,cf)
        infer_dataloaders = DataLoader(infer_data, batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
        test(infer,net,infer_dataloaders, my_logging,save_predict=True)