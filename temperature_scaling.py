"""
Authors: Nicholas Kashani Motlagh, Aswathnarayan Radhakrishnan, Dr. Jim Davis
Affiliation: Ohio State University
Email: kashanimotlagh.1@osu.edu
Date: 11/01/21
URL: https://github.com/osu-cvl/calibration/tree/main/temperature_scaling

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks,"
In ICML, pp. 2130-2143, 2017.
Available: https://arxiv.org/abs/1706.04599v2.
"""
# Import torch modules
import torch
from torch import nn, optim
from torch.nn import functional as F
# Import plot module
import matplotlib.pyplot as plt

class IdentityNet(nn.Module):
    """Simple network used when logits only are provided.
    """
    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, input):
        return input

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        Note: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model=None, n_bins=15, strategy='learn', per_class=False, device='cpu', verbose=False):
        """
        Args:
            model: A torch nn.Module neural network. Defaults to identity network.
            n_bins: The number of bins used in ECE. Default: 15.
            strategy: The strategy used to temperature scale, either 'learn' or 'grid'. Default: 'grid'
            per_class: Perform temperature scaling per-class. Default: False.
            device: The device to perform computations. Default: 'cpu'.
            verbose: Report updates on process. Default: False.
        """
        super(ModelWithTemperature, self).__init__()
        # Save parameters
        # Make new model
        if model is None:
            self.model = IdentityNet()
        else:
            self.model = model
        self.model.eval()
        self.model.to(device)
        self.strategy = strategy
        self.device = device
        self.per_class = per_class
        self.verbose = verbose
        self.n_bins = n_bins
        # Use ece loss criterion
        self.ece_criterion = ECE(n_bins=n_bins, device=device)


    def forward(self, input):
        """Forward function for nn.module

        Args:
            input: A tensor of inputs (rows are examples, and columns are features)

        Returns:
            A tensor of temperature scaled logits (rows are examples, and columns are features)
        """
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """Perform temperature scaling on logits.

        Args:
            logits: A tensor of logits (rows are examples, and columns are features).

        Returns:
            A tensor of temperature scaled logits.
        """

        # Expand temperature to match the size of logits (in global case)
        if not self.per_class:
            temperature = self.temperature.expand(logits.size(0), logits.size(1))
        else:
            # get argmax predictions to determine per-class temperatures
            preds = torch.argmax(logits, dim=1)

            # Get appropriate temperature values
            temperature = self.temperature[preds].unsqueeze(1).expand(logits.size(0), logits.size(1))

        # Divide logits by appropriate temperatures
        return logits / temperature

    def global_temperature_scale(self, logits, temp):
        """Static method which performs temperature scaling on logits with specified temperature.

        Args:
            logits: A tensor of logits (rows are examples, and columns are features).
            temp: A scalar temperature value.

        Returns:
            A tensor of logits scaled using a scalar temperature.

        """
        # Expand temperature to match the size of logits
        return logits / temp.expand(logits.size(0), logits.size(1))

    def set_temperature(self, valid_loader, t_vals=[1.0], lrs=[0.01], num_iters=[50]):
        """
        Resets and tunes the temperature of the model (using the validation set).
        We're going to set it to optimize NLL for the learning approach and ECE for grid search.
        Args:
            valid_loader (DataLoader): validation set loader.
            t_vals (List of floats): list of temperature values to search over (must use 'grid' strategy).
            lr (List of floats): Learning rate for learned temperature scaling (must use 'learn' strategy).
            num_iters (List of ints): Maximum number of iterations for learned temperature scaling
                                        (must use 'learn' strategy).

        Returns:
            Either a scalar float temperature, or a tensor of float temperatures.
        """
        # Initialize nll criterion
        self.nll_criterion = nn.CrossEntropyLoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                labels_list.append(label)
                logits = self.model(input)
                logits_list.append(logits)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Save parameters
        self.logits = logits
        self.targets = labels
        self.num_classes = logits.shape[1]

        # Perform the appropriate strategy
        if self.strategy == "grid":
            return self.set_temperature_grid(logits, labels, t_vals=t_vals)
        else:
            return self.set_temperature_learn(logits, labels, lrs=lrs, num_iters=num_iters)

    def set_temperature_learn(self, all_logits, all_labels, lrs=[0.01], num_iters=[50]):
        """Tune the temperature of the model (using the validation set) and learned temperature scaling.
            We're going to user torch LBFGS optimizer to optimize NLL.

        Args:
            all_logits: A tensor of all the logits that were in the validation loader.
            all_labels: A tensor of all the labels that were in the validation loader.
            lrs: A list of float learning rates to search through. Default: [0.01].
            num_iters: A list of int maximum number of iterations to search through. Default: [50].

        Return:
            The best temperature or tensor of temperatures.
        """
        # Per-class learned temperature
        if self.per_class:
            # List of optimal temperatures
            optim_temps = []
            # Network predictions
            preds = torch.argmax(all_logits, dim=1)
            # For each class, learn temperature
            for l in range(self.num_classes):
                if self.verbose:
                    print(f'Searching optimal temperature for class {l}')
                # Default temperature is 1
                optim_temps.append(torch.ones((1), device=self.device))
                # Get all logits which were predicted as class l (and their associated targets)
                logits = all_logits[preds==l]
                labels = all_labels[preds==l]
                # Check that there are any such logits (skip if not)
                if labels.shape[0] == 0:
                    continue
                # Get class ece and nll
                before_temperature_nll = self.nll_criterion(logits, labels).item()
                before_temperature_ece = self.ece_criterion(logits, labels).item()
                if self.verbose:
                    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
                # Initialize best_ece/best_nll for class as base ece/nll.
                best_ece = before_temperature_ece
                best_nll = before_temperature_nll
                # Search through hyper parameters
                for lr in lrs:
                    for num_iter in num_iters:
                        # Initialize temperature
                        temperature = nn.Parameter(torch.ones((1), device=self.device, requires_grad=True)* 1.)
                        def eval():
                            """Perform forward and backward pass of nll.
                            Returns:
                                A loss from nll.
                            """
                            loss = self.nll_criterion(self.global_temperature_scale(logits, temperature), labels)
                            loss.backward()
                            return loss
                        # Optimize the temperature w.r.t. NLL
                        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=num_iter)
                        optimizer.step(eval)

                        # Calculate NLL and ECE after temperature scaling
                        after_temperature_nll = self.nll_criterion(self.global_temperature_scale(logits, temperature), labels).item()
                        after_temperature_ece = self.ece_criterion(self.global_temperature_scale(logits, temperature), labels).item()
                        # Save temperature if ece was improved
                        if after_temperature_ece < best_ece:
                            if self.verbose: print(f'New Optimal temperature: {temperature.data.item():.3f}')
                            if self.verbose: print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
                            optim_temps[l] = temperature
                            best_ece = after_temperature_ece
                            best_nll = after_temperature_nll
                if self.verbose: print(f'Optimal temperature: {optim_temps[-1].data.item():.3f}')
                if self.verbose: print('After temperature - NLL: %.3f, ECE: %.3f' % (best_nll, best_ece))

            # Save all optimal temperatures
            self.temperature = torch.tensor(optim_temps).data.to(self.device)

        else:
            # No best temperature found yet
            optim_temp = nn.Parameter(torch.ones((1), device=self.device, requires_grad=True)* 1.)
            # Get global nll and ece before temperature scaling
            before_temperature_nll = self.nll_criterion(all_logits, all_labels).item()
            before_temperature_ece = self.ece_criterion(all_logits, all_labels).item()
            if self.verbose:
                print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
            # Save as best ece and nll
            best_ece = before_temperature_ece
            best_nll = before_temperature_nll
            # Search through hyper parameters
            for lr in lrs:
                for num_iter in num_iters:
                    # Initialize temperature
                    temperature = nn.Parameter(torch.ones((1), device=self.device, requires_grad=True)* 1.)
                    def eval():
                        """Perform forward and backward pass of nll.
                        Returns:
                            A loss from nll.
                        """
                        loss = self.nll_criterion(self.global_temperature_scale(all_logits, temperature), all_labels)
                        loss.backward()
                        return loss
                    # Optimize the temperature w.r.t. NLL
                    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=num_iter)
                    optimizer.step(eval)

                    # Calculate NLL and ECE after temperature scaling
                    after_temperature_nll = self.nll_criterion(self.global_temperature_scale(all_logits, temperature), all_labels).item()
                    after_temperature_ece = self.ece_criterion(self.global_temperature_scale(all_logits, temperature), all_labels).item()
                    # Save if improved ECE
                    if after_temperature_ece < best_ece:
                        if self.verbose: print(f'Optimal temperature: {temperature.data.item():.3f}')
                        optim_temp = temperature
                        best_ece = after_temperature_ece
                        best_nll = after_temperature_nll

            # Save the optimal temperature
            self.temperature = optim_temp.data
            if self.verbose: print(f'Optimal temperature: {optim_temp.data.item():.3f}')
            if self.verbose: print('After temperature - NLL: %.3f, ECE: %.3f' % (best_nll, best_ece))

        return self.temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature_grid(self, all_logits, all_labels, t_vals=[0.5, 1.0, 2.0]):
        """Tune the temperature of the model (using the validation set) and grid search temperature scaling.

        Args:
            all_logits: A tensor of all the logits that were in the validation loader.
            all_labels: A tensor of all the labels that were in the validation loader.
            t_vals: A list of float temperature values to search through. Default: [0.5, 1.0, 2.0].

        Return:
            The best temperature or tensor of temperatures.
        """

        if self.per_class:
            # Get network predictions
            preds = torch.argmax(all_logits, dim=1)

            # Initialize results
            optim_temps = []
            # Search over number of classes
            for l in range(self.num_classes):
                # Default temperature is 1
                optim_temps.append(torch.ones((1,1)).to(self.device))
                if self.verbose: print(f'Searching optimal temperature for label class: {l}')

                # Get logits and corresponding labels of predicted class examples
                c_idx = torch.where(preds == l)[0]
                logits = all_logits[c_idx]
                labels = all_labels[c_idx]

                # Check that there are predictions of the current class
                if labels.shape[0] == 0:
                    continue

                # Get ece and nll before temperature scaling
                before_temperature_nll = self.nll_criterion(logits, labels).item()
                before_temperature_ece = self.ece_criterion(logits, labels).item()
                if self.verbose: print('\tBefore temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
                # Initialize best ece to base ece
                best_ece = before_temperature_ece
                # Get most-recently added temperature (1.0)
                optim_temp = optim_temps[-1]

                # Search over possible temperature values
                for t in t_vals:
                    # Get current temperature value
                    temp = torch.ones((1,1)).to(self.device) * t
                    if self.verbose: print(f'\t\tTemperature values: {t}')
                    # Get ece score after temperature scaling with t
                    after_temperature_ece = self.ece_criterion(self.global_temperature_scale(logits, temp), labels).item()
                    if self.verbose: print(f'\t\tAfter temperature - ECE: {after_temperature_ece:.3f}')
                    # Save parameter if best ece found
                    if after_temperature_ece < best_ece:
                        best_ece = after_temperature_ece
                        optim_temp = temp
                        if self.verbose: print(f'\t\tCurrent best ECE: {best_ece}')
                        if self.verbose: print(f'\t\tCurrent optimum T: {optim_temp.item()}')

                if self.verbose: print(f'\tFinal best ECE for label class {l}: {best_ece}')
                if self.verbose: print(f'\tFinal optimum T for label class {l}: {optim_temp.item()}')
                # Save optimal temperature
                optim_temps[-1] = optim_temp

            # Save as temperature
            self.temperature = torch.tensor(optim_temps).data.to(self.device)
        else:
            if self.verbose: print(f'Searching optimal global temperature')
            logits = all_logits
            labels = all_labels
            # Get base ece score
            before_temperature_ece = self.ece_criterion(logits, labels).item()
            if self.verbose: print(f'Before temperature - ECE: {before_temperature_ece:.3f}')
            # Save base ece as best ece
            best_ece = before_temperature_ece
            # Initialize optimal temperature to 1
            optim_temp = torch.ones((1,1)).to(self.device)
            # Iterate through all temperature values
            for t in t_vals:
                if self.verbose: print(f'\tTemperature values: {t}')
                # Set temperature to t
                temp = torch.ones((1,1)).to(self.device) * t
                # Compute ece using new temperature
                after_temperature_ece = self.ece_criterion(self.global_temperature_scale(logits, temp), labels).item()
                if self.verbose: print(f'\tAfter temperature - ECE: {after_temperature_ece:.3f}')
                # If temperature reduced ece, save temperature and best ece
                if after_temperature_ece<best_ece:
                    best_ece = after_temperature_ece
                    optim_temp = temp
                    if self.verbose: print(f'\tCurrent best ECE: {best_ece}')
                    if self.verbose: print(f'\tCurrent optimum T: {optim_temp.item()}')
            # Save optimal temperature
            if self.verbose: print(f'Final best ECE: {best_ece}')
            if self.verbose: print(f'Final optimum T: {optim_temp.item()}')
            self.temperature = optim_temp.data

        return self.temperature

    def reliability_diagram_and_bin_count(self):
        """Plots reliability and bin count diagrams
        """
        if self.per_class:
            preds = torch.argmax(self.logits, dim=1)
            for c_idx in range(self.num_classes):
                class_logits = self.logits[preds == c_idx]
                class_targets = self.targets[preds == c_idx]
                self.ece_criterion.reliability_diagram_and_bin_count(logits=class_logits, targets=class_targets,
                                                                     title=f"Class-{c_idx}")
        else:
            self.ece_criterion.reliability_diagram_and_bin_count(logits=self.logits, targets=self.targets)


class ECE(nn.Module):
    """
    ADAPTED FROM: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin.

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15, device='cpu'):
        """
            n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.device = device

    def compute_ece(self, model, val_loader):
        """Will compute the ECE of the given model on the data loader.

        Args:
            model: A model to compute ECE on.
            val_loader: A pytorch data loader.

        Returns:
            ECE on data loader.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(self.device)
                labels_list.append(label)
                logits = model(input)
                logits_list.append(logits)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return self.forward(logits, labels)

    def forward(self, logits, labels, sm=False):
        if sm:
            self.sms = logits
        else:
            self.sms = torch.softmax(logits, dim=1)
        # Save for later plotting
        self.targets = labels

        # Get softmax scores and predictions
        confidences, predictions = torch.max(self.sms, 1)
        # Get accuracy
        accuracies = predictions.eq(labels.int())
        # Initialize ece
        ece = torch.zeros(1, device=logits.device)
        # Iterate through bins
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            # Save bin if there are elements
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def reliability_diagram_and_bin_count(self, logits=None, targets=None, sm=False, title=""):
        """Creates reliability diagram and bin count plots for saved logits. Logits are saved in the forward pass.
        Can also optionally pass logits and targets to plot specific reliability diagrams and bin counts.

        Args:
            logits (optional): A tensor of logits. Defaults to None.
            targets (optional): A tensor of targets. Defaults to None.
            title (optional): A title to prepend to the default title. Defaults to "".
        """
        if logits is not None:
            if sm:
                self.sms = logits
            else:
                self.sms = torch.softmax(logits, dim=1)
        if targets is not None:
            self.targets = targets

        # Get bin precision and counts
        bin_precision, count_in_bin = self.get_full_range_bin_precision()
        # Get the number of bins
        n_bin = len(bin_precision)
        # Get the width and center of each bin
        bin_width = 1./n_bin
        bin_center = torch.linspace(0.0+0.5*bin_width,1.0+0.5*bin_width,n_bin+1)[:-1]
        # Create plots
        fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,5))
        if title != "":
            title = title + " "
        fig.suptitle(title + 'Reliability Diagram and Bin Counts')
        # Create relaiability diagram
        ax0.bar(bin_center,bin_precision,align='center',width=bin_width*0.7,label=f'Bin precision',color='orange')
        ax0.set_xlim(0,1)
        ax0.set_ylim(0,1)
        ax0.plot(bin_center,bin_center,label='ideal case',color='blue',linestyle='-.')
        ax0.set_xlabel('Estimated label posterior')
        ax0.set_ylabel('Actual precision')
        ax0.legend()
        # Create bin counts diagram
        ax1.bar(bin_center,count_in_bin,align='center',width=bin_width*0.7,label=f'Bin counts',color='blue')
        for k,c in enumerate(count_in_bin):
            ax1.text(bin_center[k]-.005,count_in_bin[k]+.1,str(int(c)),color='black',fontsize='small',fontweight='bold')
        ax1.set_xlim(0,1)
        ax1.set_xlabel('Estimated label posterior')
        ax1.set_ylabel('Example counts in bin')
        ax1.legend()
        plt.show()

    def get_full_range_bin_precision(self):
        conf, preds = torch.max(self.sms, dim=1)
        acc = (preds == self.targets)

        bin_precision = torch.zeros(self.n_bins)
        prop_in_bin = torch.zeros(self.n_bins)
        count_in_bin = torch.zeros(self.n_bins)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (conf>= bin_lower) * (conf<=bin_upper)
            # proportion of examples in bin over all examples
            prop_in_bin[i] = in_bin.float().mean()
            count_in_bin[i] = in_bin.sum()
            if prop_in_bin[i]>0:
                bin_precision[i] = (1.*acc[in_bin]).mean()
        return bin_precision, count_in_bin

class PerClassECE(nn.Module):
    """
    Calculates the MEAN Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    """
    def __init__(self, n_bins=15, device='cpu'):
        """
        n_bins (int): number of confidence interval bins
        """
        super(PerClassECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.device = device

    def compute_ece(self, model, val_loader):
        """Will compute the per-class ECE of the given model on the data loader.

        Args:
            model: A model to compute per-class ECE on.
            val_loader: A pytorch data loader.

        Returns:
            Tensor of ECE scores per-class on data loader.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(self.device)
                labels_list.append(label)
                logits = model(input)
                logits_list.append(logits)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return self.forward(logits, labels)

    def forward(self, logits, labels, sm=False):
        self.num_classes = logits.shape[1]
        if sm:
            self.sms = logits
        else:
            self.sms = torch.softmax(logits, dim=1)
        self.targets = labels
        confidences, predictions = torch.max(self.sms, dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(self.num_classes).to(self.device)
        self.bin_accuracy = torch.zeros((self.num_classes,self.n_bins))
        self.prop_in_bin = torch.zeros((self.num_classes,self.n_bins))
        self.count_in_bin = torch.zeros((self.num_classes,self.n_bins))

        for c in range(self.num_classes):
            class_idx = torch.where(predictions == c)[0]
            class_confidences = confidences[class_idx]
            class_accuracies = accuracies[class_idx]

            for i,(bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
                # Calculated |confidence - accuracy| in each bin
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                self.count_in_bin[c,i] =  in_bin.sum()
                self.prop_in_bin[c,i] = in_bin.float().mean()
                if self.count_in_bin[c,i].item() > 0:
                    self.bin_accuracy[c,i] = class_accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].float().mean()
                    ece[c] += torch.abs(avg_confidence_in_bin - self.bin_accuracy[c,i]) * self.prop_in_bin[c,i]

        return ece

    def reliability_diagram_and_bin_count(self, logits=None, targets=None, sm=False):
        """Creates reliability diagram and bin count plots for saved logits. Logits are saved in the forward pass.
        Can also optionally pass logits and targets to plot specific reliability diagrams and bin counts.

        Args:
            logits (optional): A tensor of logits. Defaults to None.
            targets (optional): A tensor of targets. Defaults to None.
            title (optional): A title to prepend to the default title. Defaults to "".
        """
        if logits is not None:
            if sm:
                self.sms = logits
            else:
                self.sms = torch.softmax(logits, dim=1)
        if targets is not None:
            self.targets = targets
        self.num_classes = self.sms.shape[1]

        for c_idx in range(self.num_classes):
            # Get bin precision and counts
            bin_precision, count_in_bin = self.get_full_range_bin_precision(c_idx)
            # Get the number of bins
            n_bin = len(bin_precision)
            # Get the width and center of each bin
            bin_width = 1./n_bin
            bin_center = torch.linspace(0.0+0.5*bin_width,1.0+0.5*bin_width,n_bin+1)[:-1]
            # Create plots
            fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,5))
            title = f"Class-{c_idx} "
            fig.suptitle(title + 'Reliability Diagram and Bin Counts')
            # Create relaiability diagram
            ax0.bar(bin_center,bin_precision,align='center',width=bin_width*0.7,label=f'Bin precision',color='orange')
            ax0.set_xlim(0,1)
            ax0.set_ylim(0,1)
            ax0.plot(bin_center,bin_center,label='ideal case',color='blue',linestyle='-.')
            ax0.set_xlabel('Estimated label posterior')
            ax0.set_ylabel('Actual precision')
            ax0.legend()
            # Create bin counts diagram
            ax1.bar(bin_center,count_in_bin,align='center',width=bin_width*0.7,label=f'Bin counts',color='blue')
            for k,c in enumerate(count_in_bin):
                ax1.text(bin_center[k]-.005,count_in_bin[k]+.1,str(int(c)),color='black',fontsize='small',fontweight='bold')
            ax1.set_xlim(0,1)
            ax1.set_xlabel('Estimated label posterior')
            ax1.set_ylabel('Example counts in bin')
            ax1.legend()
            plt.show()

    def get_full_range_bin_precision(self, c_idx):
        conf, preds = torch.max(self.sms, dim=1)
        class_idx = torch.where(preds == c_idx)[0]
        acc = (preds[class_idx] == self.targets[class_idx])

        bin_precision = torch.zeros(self.n_bins)
        prop_in_bin = torch.zeros(self.n_bins)
        count_in_bin = torch.zeros(self.n_bins)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (conf[class_idx]>= bin_lower) * (conf[class_idx]<=bin_upper)
            # proportion of examples in bin over all examples
            prop_in_bin[i] = in_bin.float().mean()
            count_in_bin[i] = in_bin.sum()
            if prop_in_bin[i]>0:
                bin_precision[i] = acc[in_bin].float().mean()
        return bin_precision, count_in_bin


def load_model(path):
    print("Load your torch model")
    pass

if __name__=="__main__":
    """An example using CIFAR10. Note that this code will not run correctly until load_model() is implemented.
    For this code to work, load_model() should return a pytorch model suitable to classify CIFAR10. See
    below.
    """
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # apply histogram binning approach --------------------------------------- #

    # load base model
    #####################################
    # CHANGE THIS LINE
    base_model = load_model("model_path")
    #####################################
    device = 'cuda'

    # init class instance
    n_bins = 15
    ece_criterion = ECE(n_bins=n_bins, device=device)
    pece_criterion = PerClassECE(n_bins=n_bins, device=device)
    # run histogram binning on validation set
    ece = ece_criterion.compute_ece(base_model, testloader)
    print(f"ECE: {ece:.2f}")
    pece = pece_criterion.compute_ece(base_model, testloader)
    print(f"Per-class ECE: {pece.mean():.2f} +- {pece.std():.2f}")

    model_temp_scaled = ModelWithTemperature(model=base_model, n_bins=n_bins, strategy="grid",
                                             per_class=True, device=device)
    # Setup values to iterate over during learning or grid search
    # For the grid search approach
    temps = torch.linspace(0.25, 4.0, 100)
    temperature = model_temp_scaled.set_temperature(testloader, t_vals=list(temps))
    print(f"Temperature: {temperature:.2f}")

    ece = ece_criterion.compute_ece(model_temp_scaled, testloader)
    print(f"ECE: {ece:.2f}")
    pece = pece_criterion.compute_ece(model_temp_scaled, testloader)
    print(f"Per-class ECE: {pece.mean():.2f} +- {pece.std():.2f}")

    # viz of bin reliability and counts
    ece_criterion.reliability_diagram_and_bin_count()
    pece_criterion.reliability_diagram_and_bin_count()