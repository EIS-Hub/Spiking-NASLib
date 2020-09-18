import codecs
import time
import json
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from naslib.utils import utils
from naslib.utils.logging import log_every_n_seconds, log_first_n


logger = logging.getLogger(__name__)

class Trainer(object):
    """
    Class which handles all the training.

    - Data loading and preparing batches
    - train loop
    - gather statistics
    - do the final evaluation
    """

    def __init__(self, optimizer, dataset, config):
        self.optimizer = optimizer
        self.dataset = dataset
        self.config = config
        self.epochs = self.config.search.epochs

        # preparations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._prepare_dataloaders(config.search)

        # measuring stuff
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_loss = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()

        n_parameters = optimizer.get_model_size()
        logger.info("param size = %fMB", n_parameters)
        self.errors_dict = utils.AttrDict(
            {'train_acc': [],
             'train_loss': [],
             'valid_acc': [],
             'valid_loss': [],
             'test_acc': [],
             'test_loss': [],
             'runtime': [],
             'params': n_parameters}
        )


    def _prepare_dataloaders(self, config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(config)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue
    

    def _setup_checkpointers(self, resume_from="", search=True, **add_checkpointables):

        checkpointables = self.optimizer.get_checkpointables()
        checkpointables.update(add_checkpointables)

        self.checkpointer = checkpointer = Checkpointer(
            model=checkpointables.pop('model'),
            save_dir=self.config.save + "/search" if search else self.config.save + "/eval",
            **checkpointables
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            self.checkpointer,
            period=1,
            max_iter=self.config.search.epochs if search else self.config.evaluation.epochs
        )

        if resume_from:
            logger.info("loading model from file {}".format(resume_from))
            checkpoint = self.checkpointer.resume_or_load(resume_from, resume=True)
            if self.checkpointer.has_checkpoint():
                return checkpoint.get("iteration", -1) + 1
        return 0


    def search(self, resume_from=""):
        logger.info("Start training")
        self.optimizer.before_training()
        if self.optimizer.using_step_function:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer.op_optimizer, float(self.epochs), eta_min=self.config.search.learning_rate_min)
        
        start_epoch = self._setup_checkpointers(resume_from, scheduler=self.scheduler)
        
        for e in range(start_epoch, self.epochs):
            self.optimizer.new_epoch(e)
            
            start_time = time.time()
            if self.optimizer.using_step_function:
                for step, (data_train, data_val) in enumerate(zip(self.train_queue, self.valid_queue)):
                    data_train = (data_train[0].to(self.device), data_train[1].to(self.device, non_blocking=True))
                    data_val = (data_val[0].to(self.device), data_val[1].to(self.device, non_blocking=True))

                    stats = self.optimizer.step(data_train, data_val)
                    logits_train, logits_val, train_loss, val_loss = stats

                    self._store_accuracies(logits_train, data_train[1], 'train')
                    self._store_accuracies(logits_val, data_val[1], 'val')

                    log_every_n_seconds(logging.INFO, "Epoch {}-{}, Train loss: {:.5}, validation loss: {:.5}, learning rate: {}".format(
                        e, step, train_loss, val_loss, self.scheduler.get_last_lr()), n=5)
                    
                    if torch.cuda.is_available():
                        log_first_n(logging.INFO, "cuda consumption\n {}".format(torch.cuda.memory_summary()), n=3)

                    self.train_loss.update(float(train_loss.detach().cpu()))
                    self.val_loss.update(float(val_loss.detach().cpu()))
                    
                self.scheduler.step()
                self.periodic_checkpointer.step(e)

                end_time = time.time()

                self.errors_dict.train_acc.append(self.train_top1.avg)
                self.errors_dict.train_loss.append(self.train_loss.avg)
                self.errors_dict.valid_acc.append(self.val_top1.avg)
                self.errors_dict.valid_loss.append(self.val_loss.avg)
                self.errors_dict.runtime.append(end_time - start_time)
            else:
                end_time = time.time()
                train_acc, train_loss, valid_acc, valid_loss = self.optimizer.train_statistics()
                self.errors_dict.train_acc.append(train_acc)
                self.errors_dict.train_loss.append(train_loss)
                self.errors_dict.valid_acc.append(valid_acc)
                self.errors_dict.valid_loss.append(valid_loss)
                self.errors_dict.runtime.append(end_time - start_time)
                self.train_top1.avg = train_acc
                self.val_top1.avg = valid_acc

            anytime_results = self.optimizer.test_statistics()
            if anytime_results:
                # record anytime performance
                self.errors_dict.test_acc.append(anytime_results[0])
                self.errors_dict.test_loss.append(anytime_results[1])
                
            self.log_to_json()
            self._log_and_reset_accuracies(e)

        self.optimizer.after_training()
        logger.info("Training finished")
    

    def _log_and_reset_accuracies(self, epoch):
        logger.info("Epoch {} done. Train accuracy (top1, top5): {:.5f}, {:.5f}, Validation accuracy: {:.5f}, {:.5f}".format(
                epoch,
                self.train_top1.avg, self.train_top5.avg,
                self.val_top1.avg, self.val_top5.avg
            ))
        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()


    def _store_accuracies(self, logits, target, split):

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == 'train':
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == 'val':
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")


    def log_to_json(self):
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save,
                                      'errors_{}.json'.format(self.config.seed)),
                         'w', encoding='utf-8') as file:
            json.dump(self.errors_dict, file, separators=(',', ':'))


    def evaluate(
            self, 
            retrain=True, 
            search_model="", 
            resume_from=""
        ):
        logger.info("Start evaluation")
        self._prepare_dataloaders(self.config.evaluation)

        if not search_model:
            search_model = os.path.join(self.config.save, "search", "model_final.pth"),
        self._setup_checkpointers(search_model)      # required to load the architecture

        best_arch = self.optimizer.get_final_architecture()
        logger.info("Final architecture:\n" + best_arch.modules_str())

        if best_arch.QUERYABLE:
            metric = 'eval_acc1es'
            result = best_arch.query(
                metric=metric, dataset=self.config.dataset, path=self.config.data
            )
            logger.info("Queried results ({}): {}".format(metric, result))
        else:
            best_arch.to(self.device)
            if retrain:
                logger.info("Starting retraining from scratch")
                best_arch.reset_weights(inplace=True)

                epochs = self.config.evaluation.epochs
                optim = self.optimizer.get_op_optimizer()
                optim = optim(
                    best_arch.parameters(), 
                    self.config.evaluation.learning_rate,
                    momentum=self.config.evaluation.momentum,
                    weight_decay=self.config.evaluation.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, float(epochs), eta_min=self.config.evaluation.learning_rate_min)

                start_epoch = self._setup_checkpointers(resume_from, search=False, 
                    model=best_arch,
                    optim=optim,
                    scheduler=scheduler
                )

                grad_clip = self.config.evaluation.grad_clip
                loss = torch.nn.CrossEntropyLoss()

                best_arch.train()
                self.train_top1.reset()
                self.train_top5.reset()

                # train from scratch
                for e in range(start_epoch, epochs):
                    for i, (input_train, target_train) in enumerate(self.train_queue):
                        input_train = input_train.to(self.device)
                        target_train = target_train.to(self.device, non_blocking=True)

                        optim.zero_grad()
                        logits_train = best_arch(input_train)
                        train_loss = loss(logits_train, target_train)
                        train_loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(best_arch.parameters(), grad_clip)
                        optim.step()

                        self._store_accuracies(logits_train, target_train, 'train')
                        log_every_n_seconds(logging.INFO, "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                            e, i, train_loss, scheduler.get_last_lr()), n=5)
                        
                        if torch.cuda.is_available():
                            log_first_n(logging.INFO, "cuda consumption\n {}".format(torch.cuda.memory_summary()), n=3)
                        
                    scheduler.step()
                    self.periodic_checkpointer.step(e)

                    logger.info("Epoch {} done. Train accuracy (top1, top5): {:.5}, {:.5}".format(e,
                        self.train_top1.avg, self.train_top5.avg))
                    self.train_top1.reset()
                    self.train_top5.reset()

            # measure final test accuracy
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            best_arch.eval()

            for i, data_test in enumerate(self.test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(self.device)
                target_test = target_test.to(self.device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                log_every_n_seconds(logging.INFO, "Inference batch {} of {}.".format(i, len(self.test_queue)), n=5)

            logger.info("Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(top1.avg, top5.avg))


