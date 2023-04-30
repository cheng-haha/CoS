from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from configs import args
from utils.setup import GetMOS 
import torch.nn as nn
import os
from utils.metric import Timer
from tqdm import trange
import torch
from loss.SupCon import SupConLoss
from utils.augmentations import gen_aug
from utils.logger import record_result 
import numpy as np

def MetaTrain(args):
    if args.mode in ['CoSBase' ]:
        if args.model.find('CoS') == -1:
            return Stand_train
        else:
            return CoS_train
    else:
        raise NotImplementedError("{} Mode is not implemented".format(args.mode))


def get_aug_data(data):
    if  args.data_aug  and  (not args.data_aug2) :
        aug_data = gen_aug( data.squeeze(1) , ssh_type= args.data_aug ).unsqueeze(1)
        inputs   = torch.cat( [data,aug_data] , dim=0 )
    elif args.data_aug  and  args.data_aug2 :
        aug1     = gen_aug( data.squeeze(1) , ssh_type= args.data_aug  ).unsqueeze(1)
        aug2     = gen_aug( data.squeeze(1) , ssh_type= args.data_aug2 ).unsqueeze(1)
        inputs   = torch.cat( [aug1,aug2] , dim=0 )
    else:
        '''fully supervised without data augmentation'''
        inputs   = data
    return inputs



def CoS_train( train_set,valid_set,logger):
    train_loader    = DataLoader( train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
    val_loader      = DataLoader( valid_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
    model, optimizer, scheduler = GetMOS( )
    criterion_cls    = nn.CrossEntropyLoss()
    contra_criterion = SupConLoss( temperature=args.temperature )

    state= {}
    state['best_acc'] = 0.0
    # logging
    logger.info(args)
    print("==>training...")
    timer = Timer()
    Test_losses ,Acc_tests, F1_tests = [],[],[]
    for epoch in trange(args.epochs, desc='Training_epoch'):
        model.train()
        total_num, total_loss , sum_c_loss  = 0, 0, 0
        label_list, predicted_list = [], []
        for idx, (data, labels) in enumerate(train_loader):
            timer.start()
            bsz            = labels.size(0)
            inputs         = get_aug_data(data)
            inputs, labels = inputs.cuda().float(), labels.cuda().long()
            res            = model( inputs )
            outputs , feat_list = res['output'] , res['feat_list']
            outputs = outputs[:bsz]
            loss    = criterion_cls( outputs , labels )
            c_loss  = 0
            for index in range( len( feat_list ) ):
                features = feat_list[index]
                f1, f2   = torch.split( features, [bsz, bsz], dim=0 )
                features = torch.cat( [ f1.unsqueeze(1), f2.unsqueeze(1) ], dim=1 )
                if args.supervision:    # SupCon
                    c_loss  += contra_criterion( features, labels ) * args.lam
                else:                   # SimCLR
                    c_loss  += contra_criterion( features ) * args.lam

            loss += c_loss
            sum_c_loss += c_loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_( model.parameters(), 100 )
            optimizer.step()

            timer.stop()
            total_loss += loss.detach().item() 
            with torch.no_grad():
                _, predicted = torch.max( outputs, 1 )
                label_list.append( labels )
                predicted_list.append( predicted )
            total_num = total_num + len( labels )
                
        scheduler.step()
        labels    = torch.cat( label_list ).cpu().detach().numpy()
        predicted = torch.cat( predicted_list ).cpu().detach().numpy()
        acc_train = ( predicted == labels ).sum() / total_num
        f1_train  = metrics.f1_score( labels, predicted, average='weighted' )
        logger.info( 'Epoch:[{}/{}] - loss:{:.7f}, ConLoss:{:.7f}, train@Acc: {:.5f}, train@F1: {:.5f}'\
            .format( epoch, args.epochs, (total_loss-sum_c_loss)/len(train_loader) , sum_c_loss/len(train_loader),
                     acc_train, f1_train ) )
        Test_loss , Acc_test, F1_test,state = evaluate(model, logger=logger, eval_loader = val_loader , epoch =  epoch , not_valid=False,state=state)

        for elem , save_res in zip((Test_loss,Acc_test, F1_test),(Test_losses,Acc_tests, F1_tests)):
            save_res.append(elem)
    return timer.sum(),Test_losses, Acc_tests, F1_tests



def Stand_train(train_set,valid_set,logger):
    if args.mode.find('semi') != -1 :
        train_set = train_set[0]
    train_loader = DataLoader( train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
    val_loader   = DataLoader( valid_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )

    model, optimizer, scheduler = GetMOS(opt_type=args.opt_type)
    
    criterion_cls       = nn.CrossEntropyLoss()
    state = {}
    state['best_acc']   = 0.0
    # logging
    logger.info( args )
    print( f"==>training..." )
    timer = Timer()
    Test_losses ,Acc_tests, F1_tests = [],[],[]
    for epoch in trange( args.epochs, desc='Training_epoch' ):
        model.train()
        total_num, total_loss = 0, 0
        label_list, predicted_list = [], []
        for idx, (inputs, label) in enumerate(train_loader):
            timer.start()
            inputs, label  = inputs.cuda().float(), label.cuda().long()
            output         = model(inputs)['output']
            loss = criterion_cls(output, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_( model.parameters(), 100 )
            optimizer.step()

            timer.stop()
            total_loss += loss.detach().item() 
            with torch.no_grad():
                _, predicted = torch.max( output, 1 )
                label_list.append(label)
                predicted_list.append( predicted )
            total_num = total_num + len( label )
        
        scheduler.step()
        label     = torch.cat(label_list).cpu().detach().numpy()
        predicted = torch.cat(predicted_list).cpu().detach().numpy()
        
        acc_train = ( predicted == label ).sum() / total_num
        f1_train  = metrics.f1_score( label, predicted, average='weighted' )
        logger.info( 'Epoch:[{}/{}] - loss:{:.7f}, train@Acc: {:.5f}, train@F1: {:.5f}'\
            .format( epoch, args.epochs, total_loss/len(train_loader) , 
                     acc_train, f1_train ) )
        Test_loss , Acc_test, F1_test, state = evaluate(model, logger=logger, eval_loader = val_loader , epoch =  epoch , not_valid=False, state=state)

        for elem , save_res in zip((Test_loss,Acc_test, F1_test),(Test_losses,Acc_tests, F1_tests)):
            save_res.append(elem)
    
    return timer.sum(),Test_losses, Acc_tests, F1_tests


def evaluate(model,logger, eval_loader, epoch, not_valid=True, mode='best', no_dict_ouput = False, state = {}):
    if not_valid:
        model.load_state_dict(torch.load(os.path.join(args.save_folder, mode + '.pth')), strict=False)
    model.eval()

    criterion_cls = nn.CrossEntropyLoss()

    total_loss , corect_num , total_num, f1_test  = 0.0 , 0 , 0 , 0.0 
    label_list, predicted_list = [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(eval_loader):
            model.eval()
            data, label = data.cuda().float(), label.cuda().long()
            if no_dict_ouput:
                output = model(data)
            else:
                output = model(data)['output']
            total_loss += criterion_cls(output, label).detach().item()
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            label          = label.cpu().detach().numpy()
            predicted      = predicted.cpu().detach().numpy()
            corect_num    += (predicted == label).sum()
            total_num     += len(label)
    batch_loss     = total_loss / len(eval_loader)
    ALL_label      = torch.cat(label_list).cpu().detach().numpy()
    ALL_predicted  = torch.cat(predicted_list).cpu().detach().numpy()
    acc_test       = corect_num/ total_num
    f1_test        = metrics.f1_score(ALL_label, ALL_predicted, average='weighted')
    if not_valid:
        logger.info('=> test@Acc: {:.5f}%, test@F1: {:.5f}'.format(acc_test, f1_test))
        c_mat  = confusion_matrix(label, predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat,record_flag = -1)
    else:
        logger.info('=> valid@Acc: {:.5f}, valid@F1: {:.5f}'.format(acc_test, f1_test))
        logger.info('=> cls_loss: {:.7f}'.format(batch_loss))

        if acc_test > state['best_acc']:
                # metrics for mesuring performance of model
                state['best_acc'] = acc_test
                state['best_f1'] = f1_test
                # calculate best confusion matrix
                state['cmt'] = confusion_matrix(ALL_label, ALL_predicted)

                state['best_epoch'] = epoch
                torch.save( model.state_dict(), os.path.join(args.save_folder, 'best.pth') )
            
    return total_loss, acc_test, f1_test , state
