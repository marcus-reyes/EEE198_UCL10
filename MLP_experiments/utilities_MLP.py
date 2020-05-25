# Define Subroutines

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks_MLP import MaskedLinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Simulated Annealing Functions =====================


def step_from(mask_list, num_to_flip=1):
    """Function to transition from one state to another.

    Flips equal amounts of zeros and ones to maintain sparsity.
    num_to_flip is the Hamming distance bet prev and new mask"""

    new_masks = []
    for mask in mask_list:
        new_mask = mask.clone().detach()  # make copy

        one_indices = torch.nonzero(new_mask)  # get indices of ON weights
        zero_indices = torch.nonzero((new_mask == 0))  # OFF weights

        # shuffle sorting of indices and select first num_to_flip indices
        these_ones = torch.randperm(one_indices.size()[0])[:num_to_flip]
        if (
            num_to_flip > these_ones.numel()
        ):  # can't flip more ones than available
            print("num_to_flip is too big!")
            print("Number of Ones:", one_indices.numel())
            print("Layer:", mask.size())
            raise ValueError
        these_zeros = torch.randperm(zero_indices.size()[0])[:num_to_flip]

        ones_indices = one_indices[
            these_ones
        ]  # just for neatness, take indices
        zeros_indices = zero_indices[these_zeros]

        # flip ones to zeros and vice versa
        new_mask[ones_indices[:, 0], ones_indices[:, 1]] = 0
        new_mask[zeros_indices[:, 0], zeros_indices[:, 1]] = 1

        # append mask for this layer
        new_masks.append(new_mask)

    return new_masks


class MaskWrapper:
    """Wrapper object to allow use of set() data-structs w/ as mask memory"""

    def __init__(self, mask, ham_dist):
        self.mask = mask
        self.ham_dist = ham_dist

    def __hash__(self):
        return 1  # set() checks first if in same hash bucket b4 testing __eq__
        # here we put all items in same bucket to proceed to __eq__

    def __eq__(self, other):
        # invoked when an item's presence in the memory is checked

        # it seems that same objects id()'s are also checked first before
        # __eq__ thus duplicate obj using copy.deepcopy() when adding to
        # the memory

        diff = (
            torch.bitwise_xor(
                self.mask.type(torch.int).cpu(),
                other.mask.type(torch.int).cpu(),
            )
            .sum()
            .item()
        )
        # consider two masks "equal" when the diff is less than a 3rd the
        # supposed Hamming distance
        return diff < int(self.ham_dist)


def get_ham_dist(model1, model2, vs_string):
    """Gets hamming distance and percent diff of two models"""

    model_1_masks = []
    model_2_masks = []

    for layer in model1.children():
        if type(layer) == MaskedLinear:
            model_1_masks.append(layer.mask.clone())

    for layer in model2.children():
        if type(layer) == MaskedLinear:
            model_2_masks.append(layer.mask.clone())

    dists = []
    for idx in range(len(model_1_masks)):
        dists.append(
            torch.bitwise_xor(
                model_1_masks[idx].type(torch.int).cpu(),
                model_2_masks[idx].type(torch.int).cpu(),
            )
            .sum()
            .item()
        )

    p_diffs = [dists[i] / model_1_masks[i].numel() for i in range(len(dists))]

    print("--", vs_string, "--")
    print("Average ham:", sum(dists) / len(dists))
    print("Hamming distance:", dists)
    print("Percent diff:", p_diffs)


# ==================== Functions for Model-to-Prune ==========================


def train(args, model, device, train_loader, optimizer, epoch, writer=None):
    """One epoch training"""

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if writer is not None:
                writer.add_scalar(
                        "loss", 
                        loss.item(), 
                        # 60k test samples
                        epoch*int(60000/args.batch_size) + batch_idx 
                )



def test(args, model, device, test_loader, verbose=True):
    """Test using test dataloader"""

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n"
            .format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    return 100.0 * correct / len(test_loader.dataset)


def forward_pass(model, device, train_loader, num_of_batches):
    """Forward pass on n batches"""

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _ in range(num_of_batches):
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100.0 * correct / (num_of_batches * train_loader.batch_size)


def fine_tune_masked_copy(
    args,
    model_,
    init_state_dict,
    mask_type,
    sparsity,
    path,
    train_loader,
    test_loader,
    trained_model=None,
):
    """Creates a masked model, fine-tunes it, then saves it"""

    model = type(model_)().to(DEVICE)  # duplicate model
    model.load_state_dict(init_state_dict)
    if trained_model is not None:
        model = apply_mask(
                    model, 
                    mask_type, 
                    sparsity, 
                    train_loader,
                    trained_model,
                )
        # note: model has init weights w mask
        #       trained_w_model has trained weights w mask
    else:
        model = apply_mask(model, mask_type, sparsity, train_loader)

    untrained_acc = test(args, model, DEVICE, test_loader)

    print("\n===== Fine Tuning", mask_type, "Model =====")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    accs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, DEVICE, train_loader, optimizer, epoch)
        accs.append(test(args, model, DEVICE, test_loader))

    model_masks = [
        layer.mask for layer in model.children() if type(layer) == MaskedLinear
    ]

    torch.save(
        {
            "init_state_dict": init_state_dict,
            "trained_state_dict": model.state_dict(),
            mask_type + "_mask": model_masks,
            "trained_acc": max(accs),
            "untrained_acc": untrained_acc,
            "epochs": epoch,
        },
        path,
    )


# ======================== Masks/Pruning Functions ===========================


def get_mask_mag(weights, prune_percent):
    """Get magnitude based mask """
    # weights assumed cloned and detached
    weights = torch.abs(weights)
    flat_weights = torch.flatten(weights)
    k = int(len(flat_weights) * prune_percent)
    vals, _ = torch.topk(
        flat_weights, k, largest=False
    )  # return smallest first
    thresh = vals[-1]  # get k_th highest element
    mask = torch.gt(weights, thresh).type(torch.float)

    return mask


def get_mask_mag_increase(init_weights, final_weights, prune_percent):
    """Mask based on increase of weights' magnitude throughout training

    Note: Applies criterion per-layer  (instead of global)
    """

    weights = torch.abs(final_weights) - torch.abs(init_weights)
    flat_weights = torch.flatten(weights)
    k = int(len(flat_weights) * prune_percent)
    vals, _ = torch.topk(
        flat_weights, k, largest=False
    )  # return smallest first
    thresh = vals[-1]  # get k_th highest element
    mask = torch.gt(weights, thresh).type(torch.float)

    return mask


def get_mask_mag_sign(init_weights, final_weights, prune_percent):
    """Mask based on highest magnitude that didn't change signs"""

    sign_weights = final_weights * init_weights
    sign_mask = (sign_weights >= 0).type(torch.float)

    # weights assumed flat, only include weights with constant sign
    flat_weights = torch.abs(final_weights) * sign_mask 
    k = int(len(flat_weights) * prune_percent)
    vals, _ = torch.topk(
        flat_weights, k, largest=False
    )  # return smallest first, among those whose sign didn't change
    thresh = vals[-1]  # get k_th highest element

    mask = torch.gt(flat_weights, thresh).type(torch.float)

    return mask


def get_mask_rand(weights, prune_percent):
    """Random masking"""
    # weights assumed cloned and detached
    # replicate a rand version of the weights
    rand_weights = torch.rand_like(weights)  # range: [0,1)

    # get thresh
    flat_weights = torch.flatten(rand_weights)
    k = int(len(flat_weights) * prune_percent)
    vals, _ = torch.topk(
        flat_weights, k, largest=False
    )  # return smallest first
    thresh = vals[-1]  # get k_th highest element

    mask = torch.gt(rand_weights, thresh).type(torch.float)
    return mask


def get_mask_sensitivity(mask_grad, prune_percent):
    """ SNIP criterion """

    grad_mag = torch.abs(mask_grad)
    normed_grad = grad_mag / grad_mag.sum()

    # get thresh, make
    flat_grad = torch.flatten(normed_grad)
    k = int(len(flat_grad) * prune_percent)
    vals, _ = torch.topk(flat_grad, k, largest=False)  # return smallest first
    thresh = vals[-1]  # get k_th highest element

    mask = torch.gt(normed_grad, thresh).type(torch.float)
    return mask


def apply_mask_from_list(model, mask_list):
    """Apply mask in-place, from list"""

    for idx, layer in enumerate(model.children()):
        if type(layer) == MaskedLinear:
            layer.mask = mask_list[idx].clone().detach().to(DEVICE)


def apply_mask_from_vector(model, vector, device):
    """Apply global mask from vector"""
    last_numel = 0
    for idx, layer in enumerate(model.children()):
        if type(layer) == MaskedLinear:

            # print(last_numel + layer.mask.numel())
            layer.mask = (
                vector[0, last_numel : last_numel + layer.mask.numel()]
                .clone()
                .view(layer.mask.size())
                .to(device)
            )
            last_numel += layer.mask.numel()


def apply_mask(model, mask_type, sparsity, train_loader, trained_model=None):
    """Applies a mask and returns a copy of a new masked model"""

    if mask_type == "mag_sign":
        mag_s_model = type(trained_model)()  # create new instance of the model
        mag_s_model.load_state_dict(
            trained_model.state_dict()
        )  # copy state_dict
        mag_s_model.to(DEVICE)  # note device is global var

        init_model = type(model)()  # create new instance of the model
        init_model.load_state_dict(model.state_dict())  # copy state_dict
        init_model.to(DEVICE)  # note device is global var

        # collect per layer weights in one list
        init_weights = []
        final_weights = []
        for (final_layer, init_layer) in zip(
            mag_s_model.children(), init_model.children()
        ):
            if type(final_layer) == MaskedLinear:
                init_weights.append(init_layer.weight.clone().detach())
                final_weights.append(final_layer.weight.clone().detach())

        # gather scores in one vector
        all_init_scores = torch.cat([torch.flatten(x) for x in init_weights])
        all_final_scores = torch.cat([torch.flatten(x) for x in final_weights])
        mask = get_mask_mag_sign(
                    all_init_scores, 
                    all_final_scores, 
                    sparsity
               )

        apply_mask_from_vector(init_model, mask.unsqueeze(0), DEVICE)

        return init_model

    if mask_type == "mag_increase":
        mag_inc_model = type(
            trained_model
        )()  # create new instance of the model
        mag_inc_model.load_state_dict(
            trained_model.state_dict()
        )  # copy state_dict
        mag_inc_model.to(DEVICE)  # note device is global var

        init_model = type(model)()  # create new instance of the model
        init_model.load_state_dict(model.state_dict())  # copy state_dict
        init_model.to(DEVICE)  # note device is global var

        # get mask for each layer and apply
        for (final_layer, init_layer) in zip(
            mag_inc_model.children(), init_model.children()
        ):
            if type(final_layer) == MaskedLinear:
                init_weights = init_layer.weight.clone().detach()
                final_weights = final_layer.weight.clone().detach()
                init_layer.mask = get_mask_mag_increase(
                    init_weights, final_weights, sparsity
                )
                final_layer.mask = init_layer.mask.clone().detach()

        return init_model, mag_inc_model

    elif mask_type == "post_mag":
        post_mag_model = type(trained_model)()  # create new instance
        post_mag_model.load_state_dict(trained_model.state_dict()) # copy 
        post_mag_model.to(DEVICE)  # note device is global var

        init_model = type(model)()  # create new instance of the model
        init_model.load_state_dict(model.state_dict())  # copy state_dict
        init_model.to(DEVICE)  # note device is global var
        
        all_weights = []
        for final_layer in post_mag_model.children():
            if isinstance(final_layer, MaskedLinear):
                all_weights.append(final_layer.weight.clone().detach())
        
        # gather all in one long vector
        all_scores = torch.cat([torch.flatten(x) for x in all_weights])
        mask = get_mask_mag(all_scores, sparsity)

        apply_mask_from_vector(init_model, mask.unsqueeze(0), DEVICE)

        return init_model

    elif mask_type == "snip":
        temp_model = copy.deepcopy(model)

        # set require grads of MaskedLinear masks to True, and False to weights
        for child_layer in temp_model.children():
            if type(child_layer) == MaskedLinear:
                child_layer.mask.requires_grad = True
                child_layer.weight.requires_grad = False

        # get grads
        data, target = next(iter(train_loader))
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        temp_model.train()
        temp_model.zero_grad()
        data, target = data.to(DEVICE), target.to(DEVICE)
        loss = F.nll_loss(temp_model(data), target)
        loss.backward()  # compute grad wrt graph leaves (including mask)

        # get mask for each layer
        grads_abs = []
        for child_layer in temp_model.children():
            if isinstance(child_layer, MaskedLinear):
                grads_abs.append(torch.abs(child_layer.mask.grad))

        # gather all scores in one long vector
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        keep_ratio = 1 - sparsity
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = []
        for i, g in enumerate(grads_abs):
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
            # print(i, ":", keep_masks[i].sum() / keep_masks[i].numel())

        snip_model = type(model)()  # create new instance of the model
        snip_model.load_state_dict(model.state_dict())  # copy state_dict
        snip_model.to(DEVICE)  # note DEVICE is global var
        apply_mask_from_list(snip_model, keep_masks)

        return snip_model

    elif mask_type == "mag":
        mag_model = type(model)()  # create new instance of the model
        mag_model.load_state_dict(model.state_dict())  # copy state_dict
        mag_model.to(DEVICE)  # note DEVICE is global var
        for child_layer in mag_model.children():
            if type(child_layer) == MaskedLinear:
                layer_weights = child_layer.weight.clone().detach()
                child_layer.mask = get_mask_mag(layer_weights, sparsity)

        return mag_model

    elif mask_type == "rand":
        rand_model = type(model)()  # create new instance of the model
        rand_model.load_state_dict(model.state_dict())  # copy state_dict
        rand_model.to(DEVICE)  # note DEVICE is global var
        total_weights = 0
        for child_layer in rand_model.children():
            total_weights += child_layer.weight.numel()
        rand_mask = get_mask_mag(torch.rand((1, total_weights)), sparsity)  
        rand_mask.unsqueeze(0)
        apply_mask_from_vector(rand_model, rand_mask, DEVICE)
        # for child_layer in rand_model.children():
        #     if type(child_layer) == MaskedLinear:
        #         layer_weights = child_layer.weight.clone().detach()
        #         child_layer.mask = get_mask_rand(layer_weights, sparsity)

        return rand_model
