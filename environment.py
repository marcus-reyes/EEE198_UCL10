#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import copy

import torch
import torchvision.transforms as transforms
import torchvision.datasets as ds
import torch.utils.data as data
from collections import OrderedDict

import models_to_prune

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import logging

logging.basicConfig(
    level=logging.INFO,
    format=("%(levelname)s:" + "[%(filename)s:%(lineno)d]" + " %(message)s"),
)


class PruningEnv:
    """Represents the neural network to be operated on.

    Attributes:
        dataset (str): dataset to evaluate the model against.
        model_type (str): type of model to operate on.
        train_dl (data.DataLoader): training data subset
        test_dl (data.DataLoader): test data subset
        valid_dl (data.DataLoader): data subset for evaluation in
            Simulated Annealing
        device (str): processing device to run on ie. cpu or gpu
        model: neural network to be pruned (initially by masking)
        loss_func: loss function for training model
        optimizer: optimizer for training model
        layers_to_prune ([str,]): layer names for iterating among layers to
            be masked
        layer (str): layer name of current layer that is being operated on
        layer_prune_amounts ({layer_name:num_channel}): for network parameter
            measurements
        layer_flops ({layer_name:flops}): for network flops estimations
        self.prev_out_feat = [0,0]  # List of [h,w] of prev layer's featmap
        prev_out_feat ([h,w]): previous layer's featmap size
    """

    def __init__(self, dataset="cifar10", model_type="basic"):
        """Inits model to be pruned and all environment vars"""

        # Assign dataset
        self.dataset = dataset
        self.train_dl, self.test_dl, self.valid_dl = self.get_dataloaders()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        logging.info("Device: {}".format(self.device))

        # Build chosen model to prune
        self.model_type = model_type
        self.model = self._build_model_to_prune().to(self.device)

        # logging.info("Starting Pre-Training")
        # set training parameters
        self.loss_func = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0008)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )

        # self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1
        # self._train_model(num_epochs=1)
        # self.init_full_weights = copy.deepcopy(self.model.state_dict())
        # initially, self.model has full-params
        # used in reset_to_k()

        # state
        # self.layers_to_prune = [name for name,_ in self.model.named_modules()
        #                        if 'conv' in name]
        self.layers_to_prune = ["conv1", "conv2", "conv3", "conv4"]
        logging.info("Layers to prune: {}".format(self.layers_to_prune))
        self.layer = None  # Layer to process,
        # str name, is usr-identified
        self.layer_prune_amounts = OrderedDict()
        self.layer_flops = OrderedDict()
        self.full_model_flops = 0  # will be calculated in reset_to_k()
        # self.amount_pruned_dict = {} # {layer_name : amount it was pruned}
        # self.amount_pruned = 0  # On layer_to_process, updated in prune_layer
        # self.prev_amount_pruned = 0 # previous layer's amt pruned
        self.prev_out_feat = [0, 0]  # List of [h,w] of prev layer's featmap

        # self.state_size = state_size
        self.max_layer_idx = 4  # TODO: can be derived from self.model

    def get_dataloaders(self):
        """Imports the chosen dataset

        Returns:
            train_dl, test_dl, valid_dl: dataloaders of datasets

        """

        if self.dataset.lower() == "cifar10":
            train_transform = transforms.Compose(
                [
                    # transforms.Pad(4),
                    # transforms.RandomCrop(32),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            train = ds.CIFAR10(
                root=os.getcwd(),
                train=True,
                download=True,
                transform=train_transform,
            )

            train_loader = data.DataLoader(
                train,
                batch_size=256,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            test = ds.CIFAR10(
                root=os.getcwd(),
                train=False,
                download=True,
                transform=test_transform,
            )

            test_loader = data.DataLoader(
                test,
                batch_size=256,  # testing use less
                # memory, can afford
                # larger batch_size
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            # val_loader for the SA algorithm
            val_loader = data.DataLoader(
                train,
                batch_size=1024,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            return train_loader, test_loader, val_loader

        elif self.dataset.lower() == "mnist":
            print("Using mnist")
            mnist_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            train = ds.MNIST(
                root=os.getcwd(),
                train=True,
                download=True,
                transform=mnist_transform,
            )

            train_loader = data.DataLoader(
                train,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            test = ds.MNIST(
                root=os.getcwd(),
                train=False,
                download=True,
                transform=mnist_transform,
            )

            test_loader = data.DataLoader(
                test,
                batch_size=256,  # testing use less
                # memory, can afford
                # larger batch_size
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            val_loader = data.DataLoader(
                train,
                batch_size=1024,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            return train_loader, test_loader, val_loader

        print("dataset not available")

        return -1

    def _build_model_to_prune(self):
        """Builds the model to compress/mask/prune"""

        if self.model_type.lower() == "basic":

            return models_to_prune.BasicCNN()

        else:
            print("model not available")  # TODO: use proper handling
            return -1

    def _estimate_layer_flops(self):
        """Estimates single conv layer flops

        Helper tool for calculate_network_flops() and _calculate_reward()

        Important: Assumes calculation is always done in order,
                        from first to last conv layer

        Returns:
            original_layer_flops (int): flops before pruning
            pruned_layer_flops (int): would-be flops after pruning

        """

        for name, module in self.model.named_modules():
            if self.layer in name:
                conv_layer = module
                # logging.info('conv name: {}'.format(name))
                break

        if "1" in self.layer:  # first conv layer
            for inputs, _ in self.train_dl:
                self.prev_out_feat = inputs.size()[2:]  # input is data
                prev_amount_pruned = 0  # no prev layer was pruned
                break
        else:  # get previous
            idx = list(self.layer_prune_amounts.keys()).index(self.layer) - 1
            prev_amount_pruned = list(self.layer_prune_amounts.values())[idx]

        amount_pruned = self.layer_prune_amounts[self.layer]

        input_h = self.prev_out_feat[0]
        input_w = self.prev_out_feat[1]
        kernel_h = conv_layer.kernel_size[0]
        kernel_w = conv_layer.kernel_size[1]
        pad_h = conv_layer.padding[0]
        pad_w = conv_layer.padding[1]
        stride_h = conv_layer.stride[0]
        stride_w = conv_layer.stride[1]
        C_in = conv_layer.in_channels - prev_amount_pruned
        C_out = conv_layer.out_channels
        groups = conv_layer.groups

        out_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
        out_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

        # ff assumes that flops estimation is always done in order
        self.prev_out_feat = [out_h, out_w]

        original_layer_flops = (
            C_out
            * (C_in / groups) 
            * kernel_h 
            * kernel_w 
            * out_h 
            * out_w
        )
        pruned_layer_flops = (
            (C_out - amount_pruned)
            * (C_in / groups)
            * kernel_h
            * kernel_w
            * out_h
            * out_w
        )

        return original_layer_flops, pruned_layer_flops

    def _calculate_network_flops(self):
        """Estimates flops of all CNN layers at current state

        Returns:
            reduced_layer_flops (int): flops of previous (already reduced)
                layers
            current_layers (int): flops of current layer being processed
            rest_layer_flops (int): flops of upcoming layers to be processed
        """

        # total_network_flops = sum(layer_flops.values())
        layer_idx = list(self.layer_flops.keys()).index(self.layer)
        reduced_layer_flops = sum(list(self.layer_flops.values())[:layer_idx])
        current_layer_flops = self.layer_flops[self.layer]
        rest_layer_flops = sum(
            list(self.layer_flops.values())[layer_idx + 1 :]
        )

        return reduced_layer_flops, current_layer_flops, rest_layer_flops

    def get_BNs(self):
        """MARCUS, concise def here

        Returns:
            bn_rep (type):
        """

        for key, var in self.model.named_parameters():
            if "bn" in key and "weight" in key:
                try:  # concatenate
                    bn_rep = torch.cat((bn_rep, var), 0)
                except:  # initialize if not initialized yet
                    bn_rep = var

        bn_rep -= bn_rep.min()
        bn_rep /= bn_rep.max()

        return bn_rep

    def updateBN(self):  # jeff: default model is the global model
        """ MARCUS, please add docstring here """

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # L1
                m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  

    def _train_model(self, num_epochs=10):
        """Trains the model being pruned
        Helper tool for _calculate_reward()

         Args:
             num_epochs (int): number of passes over the entire dataset
        """

        self.model.train()
        logging.info("Training CNN model")
        for epoch in range(num_epochs):
            train_acc = []
            start_time = time.time()
            for idx, train_data in enumerate(self.train_dl):
                inputs, labels = train_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # forward
                preds = self.model(inputs)
                loss = self.loss_func(preds, labels)
                # backward
                loss.backward()  # compute grads

                # self.updateBN()

                self.optimizer.step()  # update params w/ Adam update rule

                # print accuracy
                _, prediction = torch.max(preds, dim=1)  # idx w/ max val is
                # most confident class
                train_acc.append(
                    (prediction == labels).type(torch.double).mean()
                )
                if (idx + 1) % 2 == 0:
                    elapsed_time = time.time() - start_time
                    str_time = time.strftime(
                        "%H:%M:%S", time.gmtime(elapsed_time)
                    )
                    print(
                        (
                            "Epoch [{}/{}] Step [{}/{}] | "
                            + "Loss: {:.4f} Acc: {:.4f} Time: {}"
                        ).format(
                            epoch + 1,
                            num_epochs,
                            idx + 1,
                            len(self.train_dl),
                            loss.item(),
                            train_acc[-1],
                            str_time,
                        )
                    )

        logging.info("Training Done")

    def _evaluate_model(self):
        """Evaluates the model being pruned
        Helper tool for _calculate_reward()
        """

        self.model.eval()
        # logging.info('Evaluating CNN model''')
        total = 0  # total number of labels
        correct = 0  # total correct preds

        with torch.no_grad():
            for test_data in self.test_dl:
                inputs, labels = test_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(inputs)  # forward pass
                _, prediction = torch.max(preds, dim=1)
                total += labels.size(0)  # number of rows = num of samples
                correct += (prediction == labels).sum().item()

        val_acc = float(correct / total)
        val_acc = torch.tensor(val_acc, requires_grad=True)

        return val_acc

    def forward_pass(self, num_of_batches):
        """Forward pass on n batches"""

        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for _ in range(num_of_batches):
                data, target = next(iter(self.valid_dl))
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return (
            100.0 * correct / (num_of_batches * self.valid_dl.batch_size)
        )

    def maskbuildbias(self, indices, num_filters):
        """Builds a mask for the bias of the layer to be pruned.

        Sub function of prune_layer.

        Args:
            indices (list): indices to be pruned. i.e. [0,1,1,0,,1,0,1,...]
            num_filters (int): total filters on the layer

        Returns:
            bias_mask (tensor): mask vector of binary values

        """
        bias_mask = copy.copy(indices[0, :num_filters])
        bias_mask = bias_mask.type(torch.FloatTensor)

        return bias_mask.to(self.device)

    def maskbuildweight(self, indices, kernel1, kernel2, num_filters):
        """Builds a mask for the weights of the layer to be pruned.

        Sub function of prune_layer

        Args:
            indices (list): indices to be pruned i.e. [0,1,1,0,0,1,1,0,1,0...]
            MARCUS, please continue

        Returns:
            weight_mask (type):
        """

        weight_mask = copy.copy(indices[0, :num_filters]).view(-1, 1, 1)
        weight_mask = weight_mask.expand(-1, kernel1, kernel2)
        weight_mask = weight_mask.type(torch.FloatTensor)

        return weight_mask.to(self.device)

    def maskbuildweight2(
        self, prev_indices, kernel1, kernel2, num_filters_prev
    ):
        """Builds a mask for the weights of the next layer.
        Sub function of prune_layer
        Necessity from the previous layer's having less output feature maps
        Args:
            MARCUS, please continue
        Returns:
        """
        next_weight_mask = copy.copy(prev_indices[0, :num_filters_prev])
        next_weight_mask = next_weight_mask.view(-1, 1, 1)
        next_weight_mask = next_weight_mask.expand(-1, kernel1, kernel2)
        next_weight_mask = next_weight_mask.type(torch.FloatTensor)

        return next_weight_mask.to(self.device)

    def prune_layer(self, indices):
        """MARCUS

        Args:

        Returns:
        """

        iter_ = 0
        iterbn = 0
        amt_pruned = 0  # to be assigned in mask_per_channel condition
        total_filters = 0  # same as above,filter # before pruning the layer

        named_children = self.model.named_children()
        for idx, module in enumerate(named_children):
            if self.layer in module[0]:
                layer_number = idx
                # conv_layer = module
                _, next_conv_layer = next(named_children)
                break
        # iterate through all the parameters of the network
        for layer in self.model.children():
            # hardcode to find the last conv layer
            # this is not needed for now as long as you set the
            # last batchnorm layer to 0
            # proven empirically on the 3 layer 3*3 network

            # if convolutional layer
            if type(layer) == nn.Conv2d:
                # if not the layer to be pruned, skip the below
                if iter_ != layer_number and iter_ != layer_number + 1:
                    iter_ = iter_ + 1
                    continue

                # enumerate through all the contents of the layer.
                # use a different mask based on whether this is current or next
                # there should be no bias change if this is for the next
                # for a conv layer thats: 1. weights 2. biases
                for i, param in enumerate(layer.parameters()):
                    # use the param size to determine if weight or bias

                    size = param.size()
                    # if bias, then make the mask for current only
                    if len(size) == 1:

                        # stacked conditions so that it doesn't go to "else"
                        if iter_ == layer_number:
                            # print("a",a[0])

                            # multiply param.data with a mask of zeros up to
                            # desired index, all else are filled with ones
                            # logging.info('Build bias mask')
                            mask = self.maskbuildbias(indices, size[0])
                            param.data = torch.mul(param.data, mask)

                            # print("Built bias mask")
                            # iterate the cnn layer counter
                    # if weights
                    else:
                        # mask per channel
                        if iter_ == layer_number:
                            # size[2] == kernel size size[0] == num filters
                            # logging.info('Build filter mask')
                            mask = self.maskbuildweight(
                                indices, size[2], size[3], size[0]
                            )
                            masktuple = ((mask),) * size[1]
                            finalmask = torch.stack((masktuple), 1)
                            # get prune amount to return to caller
                            amt_pruned = (
                                size[0] - indices[0, : size[0]].sum()
                            ).item()

                            total_filters = size[0]

                            # update env pruning record
                            # self.prev_amount_pruned = self.amount_pruned
                            self.layer_prune_amounts[self.layer] = amt_pruned

                        elif iter_ == layer_number + 1:
                            # size[2]&[3] == kernel_size
                            # size[1] = prev_num_filters
                            # logging.info('Build next filter mask')
                            mask = self.maskbuildweight2(
                                indices, size[2], size[3], size[1]
                            )
                            masktuple = ((mask),) * size[0]
                            finalmask = torch.stack((masktuple), 0)

                        param.data = torch.mul(param.data, finalmask)
                        # print(param.data,"after")
                iter_ = iter_ + 1
            if type(layer) == nn.BatchNorm2d:
                for i, param in enumerate(layer.parameters()):
                    if iterbn == layer_number:
                        size = param.size()

                        # multiply param.data with a mask of zeros up to
                        # the desired index, all else are filled with ones
                        # logging.info('Build batchnorm mask')
                        mask = self.maskbuildbias(indices, size[0])

                        # print(param.data)
                        param.data = torch.mul(param.data, mask)
                iterbn = iterbn + 1

        return total_filters, amt_pruned

    def reset_to_k(self):
        """Resets CNN to partially trained net w/ full params"""

        self.model.load_state_dict(
            torch.load(os.getcwd() 
                + "/april_experiments_withBN_epoch_5.pth")["state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(os.getcwd() 
                + "/april_experiments_withBN_epoch_5.pth")["optim"]
        )

        # initialize starting layer to process
        self.layer = self.layers_to_prune[0]
        # initialize prune amounts to zero
        self.layer_prune_amounts = OrderedDict(
            zip(self.layers_to_prune, [0] * len(self.layers_to_prune))
        )

        # get layer_flops dict
        layer_to_process = self.layer  # preserve
        for name in self.layers_to_prune:
            self.layer = name
            orig_flops, flops_remain = self._estimate_layer_flops()
            # TODO: might be better to explicitly pass layer
            # name to estimate_flops()
            self.layer_flops[self.layer] = flops_remain
        self.layer = layer_to_process
        # save total network flops
        self.full_model_flops = sum(self.layer_flops.values())

    def reset_to_init_1(self):
        """Resets CNN to first initialization"""

        self.model.load_state_dict(
            torch.load(os.getcwd() + "/init_may_10_num_1.pth")["state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(os.getcwd() + "/init_may_10_num_1.pth")["optim"]
        )
        # initialize starting layer to process
        self.layer = self.layers_to_prune[0]
        # initialize prune amounts to zer
        self.layer_prune_amounts = OrderedDict(
            zip(self.layers_to_prune, [0] * len(self.layers_to_prune))
        )
        # get layer_flops dict
        layer_to_process = self.layer  # preserve
        for name in self.layers_to_prune:
            self.layer = name
            orig_flops, flops_remain = self._estimate_layer_flops()
            # name to estimate_flops()
            self.layer_flops[self.layer] = flops_remain
        self.layer = layer_to_process
        # save total network flops
        self.full_model_flops = sum(self.layer_flops.values())
