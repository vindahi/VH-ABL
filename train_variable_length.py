import argparse
import time
import torch.nn.functional as F
from model import PCH
from utils import *
from data import *
from lossss import MultiSimilarityLoss
import numpy as np


def train(args, dset):
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]
    logName = args.dataset + '_' + str(args.nbit) + '_variable_length_exp'
    log = logger(logName)
    log.info('Training Stage for Variable Length Experiment...')
    log.info('Base hash length: %d, Output hash length: %d', args.nbit, args.output_nbit)
    
    loss_l2 = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    model = PCH(args=args)
    model.train().cuda()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr}])

    start_time = time.time() * 1000

    MSL = MultiSimilarityLoss()

    train_loader = data.DataLoader(my_dataset(dset.I_tr, dset.T_tr, dset.L_tr),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    

    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):
            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())
            
            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()
            label = label.cuda()
            

            aff_label = torch.Tensor(aff_label).cuda()

            optimizer.zero_grad()
            H, pred = model(img_feat, txt_feat, label)
            H_norm = F.normalize(H)

            center = model.centroids.to(dtype=torch.float32).cuda()

            cen_loss = center_loss(center)

            code_center = H.mm(center.t())
            constr_loss = bce_loss(code_center, label)

            clf_loss = loss_l2(torch.sigmoid(pred), label)

            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)

            code_cen_loss = code_center_loss(H, center, label)

            loss = clf_loss * args.param_clf + similarity_loss * args.param_sim + cen_loss * args.param_cen + code_cen_loss * args.param_it + constr_loss * args.param_sup
            
            loss.backward()
            optimizer.step()
            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:
                log.info('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-S: %.4f, Loss-CEN: %.4f, Loss-IT: %.4f, Loss-SUP: %.4f'
                          % (epoch + 1, args.epochs, loss.item(),
                             clf_loss.item() * args.param_clf,
                             similarity_loss.item() * args.param_sim,
                             cen_loss.item() * args.param_cen,
                             code_cen_loss.item() * args.param_it,
                             constr_loss.item() * args.param_sup))  

    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    log.info('Training Time: %.4f' % (elapsed))

    return model


def eval_adaptive_truncation(model, dset, args):
    """
    使用自适应截断机制进行评估：计算全局比特重要性排序，然后截取前output_nbit位作为最终的哈希码
    """
    model.eval()
    logName = args.dataset + '_' + str(args.nbit) + '_adaptive_truncation_exp'
    log = logger(logName)
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    # 计算全局比特重要性排序
    log.info('Computing global bit importance ranking...')
    retrieval_loader_full = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)
    
    global_ranking = model.compute_global_bit_importance(retrieval_loader_full)
    log.info(f'Global bit importance ranking computed. Top 10 important bits: {global_ranking[:10]}')

    retrieval_loader = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)

    retrievalP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(retrieval_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        label = label.cuda()
        # 使用自适应截断获取指定长度的哈希码
        H, _ = model(img_feat, txt_feat, label, target_length=args.output_nbit)
        retrievalP.append(H.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    # 此时H已经是目标长度，无需进一步截断
    retrievalCode = np.sign(retrievalH)

    end_time = time.time() * 1000
    retrieval_time = end_time - start_time

    log.info('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    valP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(val_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        label = label.cuda()
        # 使用自适应截断获取指定长度的哈希码
        H, _ = model(img_feat, txt_feat, label, target_length=args.output_nbit)
        valP.append(H.data.cpu().numpy())

    valH = np.concatenate(valP)
    # 此时H已经是目标长度，无需进一步截断
    valCode = np.sign(valH)

    end_time = time.time() * 1000
    query_time = end_time - start_time
    log.info('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))
    
    if args.save_flag:
        map = calculate_map(qu_B=valCode.astype(np.int8), re_B=retrievalCode.astype(np.int8), qu_L=dset.L_te, re_L=dset.L_db)
        log.info('[Adaptive Truncation MAP - Base Length: %d -> Output Length: %d] %.4f' % (args.nbit, args.output_nbit, map))
        
        # 如果需要保存结果用于后续绘图
        if hasattr(args, 'results_file') and args.results_file:
            with open(args.results_file, 'a') as f:
                f.write('%d,%d,%.4f\n' % (args.nbit, args.output_nbit, map))
                
        if isinstance(valCode, torch.Tensor):
            valCode_np = valCode.cpu().numpy().astype(np.int8)
        else:
            valCode_np = valCode.astype(np.int8)

        if isinstance(retrievalCode, torch.Tensor):
            retrievalCode_np = retrievalCode.cpu().numpy().astype(np.int8)
        else:
            retrievalCode_np = retrievalCode.astype(np.int8)

    return map


def run_variable_length_experiment(args, dset):
    """
    运行可变长度实验：使用不同的基础长度训练模型，但输出固定长度的哈希码
    """
    base_lengths = [2**i for i in range(2, 11)]  # [2^2, 2^3, ..., 2^10] = [4, 8, ..., 1024]
    output_length = args.output_nbit  # 固定输出长度为16位
    
    logName = args.dataset + '_variable_length_exp'
    log = logger(logName)
    log.info("Starting variable length experiment with adaptive truncation...")
    log.info(f"Base lengths to test: {base_lengths}")
    log.info(f"Fixed output length: {output_length}")
    
    results = {}
    
    # 创建结果文件并写入头部
    args.results_file = f"{args.dataset}_variable_length_results.csv"
    with open(args.results_file, 'w') as f:
        f.write('base_length,output_length,map\n')  # CSV头部
    
    for base_len in base_lengths:
        log.info(f"Training model with base length: {base_len}")
        
        # 临时修改参数用于当前实验
        original_nbit = args.nbit
        args.nbit = base_len
        
        # 训练模型
        model = train(args, dset)
        
        # 使用自适应截断评估
        map_score = eval_adaptive_truncation(model, dset, args)
        
        results[base_len] = map_score
        log.info(f"Base length {base_len} -> Output {output_length}-bit MAP: {map_score:.4f}")
        
        # 恢复原始参数
        args.nbit = original_nbit
        
    log.info("Variable length experiment completed.")
    log.info("Results:")
    for base_len, map_score in results.items():
        log.info(f"Base length: {base_len}, Output {output_length}-bit MAP: {map_score:.4f}")
    
    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--model', type=str, default='FSFH', help='Use GMMH.')
    parser.add_argument('--self_paced',  type=bool, default='True', help='--self_paced learning schedule')
    parser.add_argument('--epochs', type=int, default=140, help='Number of student epochs to train.')
    parser.add_argument('--epochs_pre', type=int, default=100, help='Epoch to learn the hashcode.')
    parser.add_argument('--nbit', type=int, default=128, help='Base hash code length during training')
    parser.add_argument('--output_nbit', type=int, default=16, help='Output hash code length for evaluation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.8, help='')
    parser.add_argument('--mlpdrop', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--loss', type=str, default='p2p', help='different kinds of loss')
    parser.add_argument('--temp', type=float, default=0.3, help='temperature')
    parser.add_argument("--tau",type=float,default=0.2)

    parser.add_argument('--nhead', type=int, default=1, help='"nhead" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--trans_act', type=str, default='gelu', help='"activation" in Transformer.')

    
    ## Data params
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1386)

    ## Net latent dimension params
    # COCO: 128 Flickr: 256
    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')
    

    ## Loss params
    parser.add_argument('--param_dn', type=float, default=0.000001)
    parser.add_argument('--param_qmi', type=float, default=0.000001)
    parser.add_argument('--param_clf', type=float, default=1)
    parser.add_argument('--param_sim', type=float, default=1)
    parser.add_argument('--param_cluster', type=float, default=0.01)
    parser.add_argument('--param_it', type=float, default=0.01)
    parser.add_argument('--param_sup', type=float, default=0.0001)
    parser.add_argument('--param_cen', type=float, default=0.01)

    ## Flag params
    parser.add_argument('--save_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--experiment_type', type=str, default='variable_length', 
                        choices=['standard', 'variable_length'], 
                        help='Type of experiment to run')
    args = parser.parse_args()

    seed_setting(args.seed)

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))

    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]

    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)

    if args.experiment_type == 'variable_length':
        # 运行可变长度实验
        run_variable_length_experiment(args, dset)
    else:
        # 运行标准实验
        model = train(args, dset)
        eval_adaptive_truncation(model, dset, args)