import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from datasets.custom_dataset import CustomDataset
from datasets.contrastive_viewgenerator import custom_transform
from metric.accuracy import mIoU
import matplotlib.pyplot as plt
from SimCLR import SimCLR_Loss

def trainer3_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = CustomDataset(root_dir=args.root_path, transform=custom_transform)
    print("The length of train set is: {}".format(len(db_train)))

    def split_train_valid(data):
        train_size=int((0.9 * len(db_train)))
        valid_size= len(db_train)-train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(db_train, [train_size, valid_size])
        return train_dataset, valid_dataset

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # get one item in data
    train_dataset=split_train_valid(db_train)[0]
    valid_dataset=split_train_valid(db_train)[1]

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    validloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    contrastive_loss = SimCLR_Loss(batch_size, temperature=0.5)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0

    best_performance = 0.0
    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)

    loss_history={'train':[], 'val':[]}
    accuracy_history={'train':[], 'val':[]}

    def train_epoch(dataloader,max_epochs, mode='train'):
        iter_num=0
        max_iterations = max_epochs * len(dataloader)
        logging.info("{} iterations per epoch. {} max iterations ".format(len(dataloader), max_iterations))

        running_loss = 0.0
        running_accuracy = 0.0

        model.train() if mode =='train' else model.eval()

        for i_batch, (sampled_batch, pos_batch) in enumerate(dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            pos_image_batch= pos_batch['image']
            image_batch, label_batch, pos_image_batch = image_batch.cuda(), label_batch.cuda(), pos_image_batch.cuda()

            
            with torch.set_grad_enabled(mode == 'train'):
                #print(image_batch.shape)
                #print(pos_image_batch.shape)
                outputs, outputs2 = model(image_batch)
                outputs_pos, outputs_pos2 = model(pos_image_batch)
                # print(outputs.shape) # 24,14,224,224
                # print(outputs2.shape)
                # print(outputs_pos2.shape)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss_cl = contrastive_loss(outputs2,outputs_pos2)
                loss = 1* loss_ce + 1* loss_dice + 1*loss_cl
                accuracy = mIoU(pred_mask=outputs,mask=label_batch)

                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                    iter_num=iter_num +1
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                running_loss += loss.item()
                running_accuracy += accuracy

                if iter_num % 20 == 0 and mode == 'train':
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image(f'{mode}/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image(f'{mode}/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image(f'{mode}/GroundTruth', labs, iter_num)

        average_loss = running_loss / len(dataloader)
        average_accuracy = running_accuracy / len(dataloader)

        writer.add_scalar(f'info/{mode}_loss', average_loss, iter_num)
        writer.add_scalar(f'info/{mode}_accuracy', average_accuracy, iter_num)

        return average_loss, average_accuracy

    for epoch_num in iterator:
        # Training
        train_loss, train_accuracy = train_epoch(trainloader, max_epoch, mode='train')
        loss_history['train'].append(train_loss)
        accuracy_history['train'].append(train_accuracy)

        # Validation
        val_loss, val_accuracy = train_epoch(validloader, max_epoch, mode='val')
        loss_history['val'].append(val_loss)
        accuracy_history['val'].append(val_accuracy)

        logging.info('Epoch %d : Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f' % (epoch_num, train_loss, train_accuracy, val_loss, val_accuracy))

        # Save the model checkpoint
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    # plot
    epochs = range(1, max_epoch + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history['train'], label='Train')
    plt.plot(epochs, loss_history['val'], label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history['train'], label='Train')
    plt.plot(epochs, accuracy_history['val'], label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    writer.close()
    return "Training Finished!"