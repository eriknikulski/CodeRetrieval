import hashlib
import os

import comet_ml

import const
import ddp
import keys


class Experiment:
    def __init__(self, run_id):
        experiment_key = hashlib.sha1(run_id.encode('utf-8')).hexdigest()
        os.environ['COMET_EXPERIMENT_KEY'] = experiment_key
        self.experiment = None

        if ddp.is_main_process():
            self.experiment = comet_ml.Experiment(
                api_key=keys.COMET_API_KEY,
                project_name=const.COMET_PROJECT_NAME,
                workspace=const.COMET_WORKSPACE,)

        ddp.sync()

        if not self.experiment:
            self.experiment = comet_ml.ExistingExperiment(
                api_key=keys.COMET_API_KEY,
                project_name=const.COMET_PROJECT_NAME,
                workspace=const.COMET_WORKSPACE,)

    def end(self):
        self.experiment.end()

    def log_initial_metrics(self, world_size, train_data_size, valid_data_size, input_lang_n_words, output_lang_n_words):
        if ddp.is_main_process():
            self.experiment.log_parameters(const.get_hyperparams({
                'setup   world_size': world_size,
                'data   train_data_size': train_data_size,
                'data   valid_data_size': valid_data_size,
                'data   input_lang_n_words': input_lang_n_words,
                'data   output_lang_n_words': output_lang_n_words,
            }))

    def log_learning_rate(self, encoder_lr, decoder_lr, epoch):
        if ddp.is_main_process():
            self.experiment.log_metric(f'learning_rate_encoder', encoder_lr, epoch=epoch)
            self.experiment.log_metric(f'learning_rate_decoder', decoder_lr, epoch=epoch)
    
    def log_batch_metrics(self, mode, loss, accuracy, encoder_grad_norm, decoder_grad_norm, step):
        if ddp.is_main_process():
            self.experiment.log_metric(f'{mode}_loss', loss, step=step)
            self.experiment.log_metric(f'{mode}_accuracy', accuracy, step=step)
            self.experiment.log_metric(f'encoder_grad_norm', encoder_grad_norm, step=step)
            self.experiment.log_metric(f'decoder_grad_norm', decoder_grad_norm, step=step)
    
    def log_epoch_metrics(self, mode, loss, accuracy, translations, epoch):
        if ddp.is_main_process():
            self.experiment.log_metric(f'{mode}_loss', loss, epoch=epoch)
            self.experiment.log_metric(f'{mode}_accuracy', accuracy, epoch=epoch)
            self.experiment.log_text(translations)


def generate_text_seq(input_lang, output_lang, inputs, results, step):
    inputs = [' '.join(input_lang.seq_from_tensor(el.flatten())) for el in inputs]
    results = [' '.join(output_lang.seq_from_tensor(el.flatten())) for el in results]
    return str(step) + '\n' + '\n\n'.join(
        str(input) + '\n  ====>  \n' + str(result) for input, result in zip(inputs, results))
        