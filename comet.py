import hashlib
import os

import comet_ml
import torch

import const
import ddp


class Experiment:
    def __init__(self, run_id):
        experiment_key = hashlib.sha1(run_id.encode('utf-8')).hexdigest()
        os.environ['COMET_EXPERIMENT_KEY'] = experiment_key
        self.experiment = None

        if ddp.is_main_process():
            self.experiment = comet_ml.Experiment(
                project_name=const.COMET_PROJECT_NAME,
                workspace=const.COMET_WORKSPACE,)

        ddp.sync()

        if not self.experiment:
            self.experiment = comet_ml.ExistingExperiment(
                project_name=const.COMET_PROJECT_NAME,
                workspace=const.COMET_WORKSPACE,)

    def end(self):
        self.experiment.end()

    def log_initial_params(self, world_size, arch_mode, train_data_size, valid_data_size,
                           input_lang_n_words, output_lang_n_words):
        if ddp.is_main_process():
            self.experiment.log_parameters(const.get_hyperparams({
                'setup   world_size': world_size,
                'setup   architecture_mode': arch_mode,
                'data   train_data_size': train_data_size,
                'data   valid_data_size': valid_data_size,
                'data   input_lang_n_words': input_lang_n_words,
                'data   output_lang_n_words': output_lang_n_words,
            }))

    def log_learning_rate(self, lr, epoch):
        if ddp.is_main_process():
            self.experiment.log_metric(f'learning_rate', lr, epoch=epoch)

    def log_metrics(self, mode, loss, accuracies=None, grad_norms=None, text=None, step=None, epoch=None):
        if ddp.is_main_process():
            self.experiment.log_metric(f'{mode}_loss', loss, step=step, epoch=epoch)
            if accuracies is not None:
                for i, accuracy in enumerate(accuracies):
                    self.experiment.log_metric(f'{mode}_{i}_accuracy', accuracy, step=step, epoch=epoch)
            if grad_norms is not None:
                for i, grad_norm in enumerate(grad_norms):
                    self.experiment.log_metric(f'{i}_grad_norm', grad_norm, step=step, epoch=epoch)
            if text:
                self.experiment.log_text(text, step=epoch)

    def log_acc_std_mean(self, mode, batch_accuracies: list[torch.tensor], epoch):
        if ddp.is_main_process():
            std, mean = torch.std_mean(torch.stack(batch_accuracies), dim=0)
            for i in range(len(std)):
                self.experiment.log_metric(f'{mode}_{i}_acc_std', float(std[i]), epoch=epoch)
            for i in range(len(mean)):
                self.experiment.log_metric(f'{mode}_{i}_acc_mean', float(mean[i]), epoch=epoch)


def generate_text_seq(input_lang, output_langs, inputs, results, step):
    inputs = [' '.join(input_lang.seq_from_tensor(el.flatten())) for el in inputs]
    results = [[' '.join(output_langs[i].seq_from_tensor(el.flatten())) for el in res] for i, res in enumerate(results)]
    return str(step) + '\n' + '\n\n'.join(
        str(input) + '\n  ====>  \n' + str(result) for input, result in zip(inputs, zip(*results)))
        