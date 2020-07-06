import torch
import torch.optim as optim
import torch.nn as nn
import model_no_class as model_no
import adversarial1 as ad
import numpy as np
import os
import argparse
from data_list import ImageList
import pre_process as prep
import math

torch.set_num_threads(1)

def test_target(loader, model):
    with torch.no_grad():
        start_test = True
        iter_val = [iter(loader['val'+str(i)]) for i in range(10)]
        for i in range(len(loader['val0'])):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            labels = labels[:, 0]
            outputs = []
            for j in range(10):
                _, output = model(inputs[j])
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer


def entropy_loss_func(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def strategy_progressive(iter_num, initial_smooth, final_smooth, max_iter_num, strategy):
    if strategy == 'e':
        lambda_p = 2 / (1 + math.exp(-10 * iter_num / max_iter_num)) -1
    elif strategy == 'l':
        lambda_p = iter_num / max_iter_num
    elif strategy == 's':
        lambda_p = iter_num // (max_iter_num // 10) * 0.1
    elif strategy == 'x':
        lambda_p = math.pow(2,(iter_num / max_iter_num)) - 1
    else:
        lambda_p = 2 / (1 + math.exp(-10 * iter_num / max_iter_num)) - 1
    smooth = initial_smooth + (final_smooth - initial_smooth) * lambda_p
    return smooth


class discriminator(nn.Module):
    def __init__(self, feature_len):
        super(discriminator, self).__init__()
        self.Bilinear = nn.Bilinear(feature_len, cate_all[0], 1024, bias=True)
        self.Bilinear.weight.data.normal_(0, 0.01)
        self.Bilinear.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(feature_len, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(2048, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())

    def forward(self, x, y):
        f1 = self.Bilinear(x, y)
        f1 = self.relu(f1)
        f1 = self.dropout(f1)
        f2 = self.fc1(x)
        f = torch.cat((f1, f2), dim=1)
        f = self.fc2_3(f)
        return f

class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return(activations)


class fine_net(nn.Module):
    def __init__(self, feature_len):
        super(fine_net,self).__init__()
        self.model_fc = model_no.Resnet50Fc()
        self.bottleneck_0 = nn.Linear(feature_len, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = predictor(256, cate_all[0])

    def forward(self,x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        logits = self.classifier_layer(out_bottleneck)
        return(out_bottleneck, logits)


class coarse_net(nn.Module):
    def __init__(self):
        super(coarse_net,self).__init__()
        self.model_fc = model_no.Resnet50Fc()

        self.bottleneck = nn.Linear(2048, 256)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck, nn.ReLU(), nn.Dropout(0.5))

        self.classifier_layer_1 = nn.Linear(256, cate_all[1])
        self.classifier_layer_1.weight.data.normal_(0, 0.01)
        self.classifier_layer_1.bias.data.fill_(0.0)

        self.classifier_layer_2 = nn.Linear(256, cate_all[2])
        self.classifier_layer_2.weight.data.normal_(0, 0.01)
        self.classifier_layer_2.bias.data.fill_(0.0)

        self.classifier_layer_3 = nn.Linear(256, cate_all[3])
        self.classifier_layer_3.weight.data.normal_(0, 0.01)
        self.classifier_layer_3.bias.data.fill_(0.0)


    def forward(self,x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        logits_1 = self.classifier_layer_1(out_bottleneck)
        logits_2 = self.classifier_layer_2(out_bottleneck)
        logits_3 = self.classifier_layer_3(out_bottleneck)
        return [logits_1, logits_2, logits_3]


class coarse_extractor(nn.Module):
    def __init__(self):
        super(coarse_extractor,self).__init__()
        self.model_fc = model_no.Resnet50Fc()
        self.bottleneck = nn.Linear(2048, 256)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck, nn.ReLU(), nn.Dropout(0.5))

    def forward(self,x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        return out_bottleneck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Learning')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--entropy_source', type=float, nargs='?', default=0, help="target dataset")
    parser.add_argument('--entropy_target', type=float, nargs='?', default=0.01, help="target dataset")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--initial_smooth', type=float, nargs='?', default=0.9, help="target dataset")
    parser.add_argument('--final_smooth', type=float, nargs='?', default=0.1, help="target dataset")
    parser.add_argument('--max_iteration', type=float, nargs='?', default=12500, help="target dataset")
    parser.add_argument('--smooth_stratege', type=str, nargs='?', default='e', help="smooth stratege")

    args = parser.parse_args()


    # device assignment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # file paths and domains
    file_path = {
        "pai":"./dataset_list/cub200_drawing_multi.txt",
        "cub":"./dataset_list/cub200_2011_multi.txt"
    }

    if args.source == "p":
        dataset_source = file_path["pai"]
    else:
        dataset_source = file_path["cub"]

    if args.target == "p":
        dataset_target = dataset_test = file_path["pai"]
    else:
        dataset_target = dataset_test = file_path["cub"]


    # dataset load
    batch_size = {"train": 36, "val": 36, "test": 4}
    for i in range(10):
        batch_size["val" + str(i)] = 4

    dataset_loaders = {}

    dataset_list = ImageList(open(dataset_source).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["train"] = torch.utils.data.DataLoader(dataset_list, batch_size=36, shuffle=True, num_workers=4)

    dataset_list = ImageList(open(dataset_target).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["val"] = torch.utils.data.DataLoader(dataset_list, batch_size=36, shuffle=True, num_workers=4)

    dataset_list = ImageList(open(dataset_test).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=4)

    prep_dict_test = prep.image_test_10crop(resize_size=256, crop_size=224)
    for i in range(10):
        dataset_list = ImageList(open(dataset_test).readlines(), transform=prep_dict_test["val" + str(i)])
        dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=4)


    # fine-grained categories and coarse-grained categories
    cate_all = [200, 122, 38, 14, 1, 1, 1]
    num_coarse_cate_sel = 3

    fine_coarse_map = []
    with open('./dataset_list/cub_labels_map.txt', 'r') as file_map:
        line = file_map.readline()
        while line:
            line_list = line.strip().split(' ')
            fine_coarse_map.append([int(line_list[i]) for i in range(7)])
            line = file_map.readline()


    # network construction
    feature_len = 2048
    # fine-grained feature extractor + fine-grained label predictor
    my_fine_net = fine_net(feature_len)
    my_fine_net = my_fine_net.to(device)
    my_fine_net.train(True)
    # domain discriminator
    my_discriminator = discriminator(256)
    my_discriminator = my_discriminator.to(device)
    my_discriminator.train(True)
    # gradient reversal layer
    my_grl = ad.AdversarialLayer(max_iter = args.max_iteration)
    # coarse-grained feature extractor + coarse-grained label predictor
    my_coarse_extractor = coarse_extractor()
    my_coarse_extractor = my_coarse_extractor.to(device)
    my_coarse_extractor.train(True)
    my_coarse_predictor = []
    for i in range(num_coarse_cate_sel):
        my_coarse_predictor.append(predictor(256, cate_all[i+1]))
        my_coarse_predictor[i].to(device)
        my_coarse_predictor[i].train(True)


    # criterion and optimizer
    criterion = {
        "classifier": nn.CrossEntropyLoss(),
        "kl_loss": nn.KLDivLoss(size_average=False),
        "adversarial": nn.BCELoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, my_fine_net.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, my_fine_net.bottleneck_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_fine_net.classifier_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_discriminator.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_coarse_extractor.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, my_coarse_extractor.bottleneck_layer.parameters()), "lr": 1}
    ]
    for i in range(num_coarse_cate_sel):
        optimizer_dict.append({"params": filter(lambda p: p.requires_grad, my_coarse_predictor[i].parameters()), "lr": 1})
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])


    # losses
    train_fine_cross_loss = 0.0
    train_coarse_cross_loss = 0.0
    train_transfer_loss = 0.0
    train_entropy_loss_source = 0.0
    train_entropy_loss_target = 0.0
    train_total_loss = 0.0

    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])


    for iter_num in range(1, args.max_iteration + 1):
        my_fine_net.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["val"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, labels_target = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.to(device)

        fine_labels_source_cpu = labels_source[:, 0].view(-1, 1)
        coarse_labels_source = []
        labels_source = labels_source.to(device)
        for i in range(num_coarse_cate_sel):
            coarse_labels_source.append(labels_source[:, i+1])
        domain_labels = torch.from_numpy(np.array([[1], ] * batch_size["train"] + [[0], ] * batch_size["train"])).float()
        domain_labels = domain_labels.to(device)

        features_btnk, logits_fine = my_fine_net(inputs)
        logits_fine_source = logits_fine.narrow(0, 0, batch_size["train"])
        fine_labels_onehot = torch.zeros(logits_fine_source.size()).scatter_(1, fine_labels_source_cpu, 1)
        fine_labels_onehot = fine_labels_onehot.to(device)

        features_btnk_coarse = my_coarse_extractor(inputs.narrow(0, 0, batch_size["train"]))
        logits_coarse = []
        for i in range(num_coarse_cate_sel):
            logits_coarse.append(my_coarse_predictor[i](features_btnk_coarse))
        logits_coarse_detach = []
        logits_coarse_detach_extended = []
        for i in range(len(logits_coarse)):
            logits_coarse_detach.append(logits_coarse[i].detach())
            logits_coarse_detach_extended.append(torch.zeros(logits_fine_source.size()))

        for i in range(len(logits_coarse_detach)):
            for j in range(len(fine_coarse_map)):
                logits_coarse_detach_extended[i][:, j] = logits_coarse_detach[i][:, fine_coarse_map[j][i+1]]
        for i in range(len(logits_coarse_detach_extended)):
            logits_coarse_detach_extended[i] = nn.Softmax(dim=1)(logits_coarse_detach_extended[i]).to(device)

        lambda_progressive = strategy_progressive(iter_num, args.initial_smooth, args.final_smooth, args.max_iteration, args.smooth_stratege)
        for i in range(1, len(logits_coarse_detach_extended)):
            logits_coarse_detach_extended[0] += logits_coarse_detach_extended[i]
        labels_onehot_smooth = (1 - lambda_progressive) * fine_labels_onehot + lambda_progressive * (logits_coarse_detach_extended[0]) / num_coarse_cate_sel

        fine_classifier_loss = criterion["kl_loss"](nn.LogSoftmax(dim=1)(logits_fine_source), labels_onehot_smooth)
        fine_classifier_loss = fine_classifier_loss / batch_size["train"]
        for i in range(len(logits_coarse)):
            if i == 0:
                coarse_classifier_loss = criterion["classifier"](logits_coarse[i], coarse_labels_source[i])
            else:
                coarse_classifier_loss += criterion["classifier"](logits_coarse[i], coarse_labels_source[i])
        classifier_loss = fine_classifier_loss + coarse_classifier_loss

        domain_predicted = my_discriminator(my_grl(features_btnk), nn.Softmax(dim=1)(logits_fine).detach())
        transfer_loss = nn.BCELoss()(domain_predicted, domain_labels)

        entropy_loss_source = entropy_loss_func(nn.Softmax(dim=1)(logits_fine.narrow(0, 0, batch_size["train"])))
        entropy_loss_target = entropy_loss_func(nn.Softmax(dim=1)(logits_fine.narrow(0, batch_size["train"], batch_size["train"])))
        total_loss = classifier_loss + transfer_loss + entropy_loss_source * args.entropy_source + entropy_loss_target * args.entropy_target

        total_loss.backward()
        optimizer.step()

        train_fine_cross_loss += fine_classifier_loss.item()
        train_coarse_cross_loss += coarse_classifier_loss.item() / 4
        train_transfer_loss += transfer_loss.item()
        train_entropy_loss_source += entropy_loss_source.item()
        train_entropy_loss_target += entropy_loss_target.item()
        train_total_loss += total_loss.item()


        # test
        test_interval = 500
        if iter_num % test_interval == 0:
            my_fine_net.eval()
            test_acc = test_target(dataset_loaders, my_fine_net)
            print('test_acc:%.4f'%(test_acc))

            print("Iter {:05d}, Average Fine Cross Entropy Loss: {:.4f}; "
                  "Average Coarse Cross Entropy Loss: {:.4f}; "
                  "Average Transfer Loss: {:.4f}; "
                  "Average Entropy Loss Source: {:.4f}; "
                  "Average Entropy Loss Target: {:.4f}; "
                  "Average Training Loss: {:.4f}".format(
                iter_num,
                train_fine_cross_loss / float(test_interval),
                train_coarse_cross_loss / float(test_interval),
                train_transfer_loss / float(test_interval),
                train_entropy_loss_source / float(test_interval),
                train_entropy_loss_target / float(test_interval),
                train_total_loss / float(test_interval))
            )

            train_fine_cross_loss = 0.0
            train_coarse_cross_loss = 0.0
            train_transfer_loss = 0.0
            train_entropy_loss_source = 0.0
            train_entropy_loss_target = 0.0
            train_total_loss = 0.0

