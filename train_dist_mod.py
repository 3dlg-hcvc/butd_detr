# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Main script for language modulation."""

import os
import torch.optim as optim
import numpy as np
import torch
import argparse
# import torch.distributed as dist
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator, GroundingGTEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths
from utils import get_scheduler, setup_logger
from models import SetCriterion, compute_hungarian_loss
import json
import random
import wandb

# import ipdb
# st = ipdb.set_trace


class TrainTester():
    """Train/test a language grounder."""

    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=0,
            name=name
        )
        # Save config file and initialize tb writer
        # if dist.get_rank() == 0:
        path = os.path.join(args.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        self.logger.info("Full config saved to {}".format(path))
        self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.joint_det:
            dataset_dict['scannet'] = 10
        print('Loading datasets:', sorted(list(dataset_dict.keys())))
        train_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='train' if not args.debug else 'val',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            augment_det=args.augment_det
        )
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls
        )
        return train_dataset, test_dataset

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            self_attend=args.self_attend
        )
        return model

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances'],
            "det_boxes": batch_data['all_detected_boxes'],
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],
            "det_class_ids": batch_data['all_detected_class_ids']
        }


    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        # train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            # sampler=train_sampler,
            drop_last=True,
            generator=g,
        )
        # test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            # sampler=test_sampler,
            drop_last=False,
            generator=g,
        )
        return train_loader, test_loader

    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        n_data = len(train_loader.dataset)
        self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        self.logger.info("Loading models ...")
        model = self.get_model(args)

        # wandb.watch(model)

        # Get criterion
        self.logger.info("Loading criterions ...")

        losses = ['boxes', 'labels']
        if args.use_contrastive_align:
            losses.append('contrastive_align')

        set_criterion = SetCriterion(
            losses=losses, eos_coef=0.1, temperature=0.07
        )

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        scheduler = get_scheduler(optimizer, len(train_loader), args)

        # Move model to devices

        model = model.cuda()
        # model = DistributedDataParallel(
        #     model, device_ids=[args.local_rank],
        #     broadcast_buffers=False  # , find_unused_parameters=True
        # )

        # Check for a checkpoint
        if args.checkpoint_path:
            self.logger.info("Loading checkpoints ...")
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)

        # Just eval and end execution
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, set_criterion, args
            )
            return

        # Training loop
        self.logger.info("Start training ...")
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            # train_loader.sampler.set_epoch(epoch)
            self.logger.info(f"==> Epoch {epoch}")
            tic = time.time()
            self.train_one_epoch(
                epoch, train_loader, model,
                set_criterion,
                optimizer, scheduler, args
            )
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )
            if epoch % args.val_freq == 0:
                # if dist.get_rank() == 0:  # save model
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Test evaluation.......")
                self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, set_criterion, args
                )

        # Training is over, evaluate
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, set_criterion, args
        )
        return saved_path

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" not in n and "text_encoder" not in n
                       and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "backbone_net" in n and p.requires_grad
                ],
                "lr": args.lr_backbone
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            }
        ]
        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def train_one_epoch(self, epoch, train_loader, model,
                        set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        # Loop over batches
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            # Forward pass
            end_points = model(inputs)

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert (key not in end_points)
                end_points[key] = batch_data[key]

            # compute_loss
            loss, end_points = compute_hungarian_loss(
                end_points, args.num_decoder_layers,
                set_criterion,
                query_points_obj_topk=args.query_points_obj_topk
            )
            # loss, end_points = self._compute_loss(
            #     end_points, criterion, set_criterion, args
            # )
            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm
            optimizer.step()
            scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / args.print_freq:.4f} \t'
                    for key in sorted(stat_dict.keys()) if 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key
                ]))

                wandb.log({"train/loss_ce": stat_dict["loss_ce"] / args.print_freq})
                wandb.log({"train/loss_bbox": stat_dict["loss_bbox"] / args.print_freq})
                wandb.log({"train/loss_giou": stat_dict["loss_giou"] / args.print_freq})
                wandb.log({"train/loss_constrastive_align": stat_dict["loss_constrastive_align"] / args.print_freq})
                wandb.log({"train/total_loss": stat_dict["loss"] / args.print_freq})
                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model, stat_dict,
                          set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        end_points = model(inputs)

        # Compute loss
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]

        # compute loss
        _, end_points = compute_hungarian_loss(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk
        )

        # _, end_points = self._compute_loss(
        #     end_points, criterion, set_criterion, args
        # )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                   and 'last_' not in key and 'head_' not in key
            ]))
        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """

        # if args.test_dataset == 'scannet':
        #     # NOT HERE
        #     return self.evaluate_one_epoch_det(
        #         epoch, test_loader, model,
        #         criterion, set_criterion, args
        #     )

        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
        else:
            # NOT HERE
            prefixes = ['proposal_']  # only proposal
        prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]

        if args.butd_cls or args.butd_gt:
            # NOT HERE
            evaluator = GroundingGTEvaluator(prefixes=prefixes)
        else:
            evaluator = GroundingEvaluator(
                only_root=True, thresholds=[0.25, 0.5], topks=[1, 5, 10], prefixes=prefixes
            )

        # Main eval branch
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                set_criterion, args
            )
            if evaluator is not None:
                for prefix in prefixes:
                    evaluator.evaluate(end_points, prefix)
        # evaluator.synchronize_between_processes()
        # if dist.get_rank() == 0:
        if evaluator is not None:
            evaluator.print_stats()
        return None

    # @torch.no_grad()
    # def evaluate_one_epoch_det(self, epoch, test_loader,
    #                            model, criterion, set_criterion, args):
    #     """
    #     Eval grounding after a single epoch.
    #
    #     Some of the args:
    #         model: a nn.Module that returns end_points (dict)
    #         criterion: a function that returns (loss, end_points)
    #     """
    #     dataset_config = ScannetDatasetConfig(18)
    #     # Used for AP calculation
    #     CONFIG_DICT = {
    #         'remove_empty_box': False, 'use_3d_nms': True,
    #         'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
    #         'per_class_proposal': True, 'conf_thresh': 0.0,
    #         'dataset_config': dataset_config,
    #         'hungarian_loss': True
    #     }
    #     stat_dict = {}
    #     model.eval()  # set model to eval mode (for bn and dp)
    #     if set_criterion is not None:
    #         set_criterion.eval()
    #
    #     if args.num_decoder_layers > 0:
    #         prefixes = ['last_', 'proposal_']
    #         prefixes += [
    #             f'{i}head_' for i in range(args.num_decoder_layers - 1)
    #         ]
    #     else:
    #         prefixes = ['proposal_']  # only proposal
    #     prefixes = ['last_']
    #     ap_calculator_list = [
    #         APCalculator(iou_thresh, dataset_config.class2type)
    #         for iou_thresh in args.ap_iou_thresholds
    #     ]
    #     mAPs = [
    #         [iou_thresh, {k: 0 for k in prefixes}]
    #         for iou_thresh in args.ap_iou_thresholds
    #     ]
    #
    #     batch_pred_map_cls_dict = {k: [] for k in prefixes}
    #     batch_gt_map_cls_dict = {k: [] for k in prefixes}
    #
    #     # Main eval branch
    #     wordidx = np.array([
    #         0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
    #         12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
    #     ])
    #     tokenidx = np.array([
    #         1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
    #         25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
    #     ])
    #     for batch_idx, batch_data in enumerate(tqdm(test_loader)):
    #         stat_dict, end_points = self._main_eval_branch(
    #             batch_idx, batch_data, test_loader, model, stat_dict,
    #             criterion, set_criterion, args
    #         )
    #         # contrast
    #         proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
    #         proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
    #         sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
    #         sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
    #         sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
    #         sem_scores = sem_scores.to(sem_scores_.device)
    #         sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
    #         end_points['last_sem_cls_scores'] = sem_scores
    #         # end contrast
    #         sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19]
    #         for w, t in zip(wordidx, tokenidx):
    #             sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
    #         end_points['last_sem_cls_scores'] = sem_cls
    #
    #         # Parse predictions
    #         # for prefix in prefixes:
    #         prefix = 'last_'
    #         batch_pred_map_cls = parse_predictions(
    #             end_points, CONFIG_DICT, prefix,
    #             size_cls_agnostic=True)
    #         batch_gt_map_cls = parse_groundtruths(
    #             end_points, CONFIG_DICT,
    #             size_cls_agnostic=True)
    #         batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
    #         batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)
    #
    #     mAP = 0.0
    #     # for prefix in prefixes:
    #     prefix = 'last_'
    #     for (batch_pred_map_cls, batch_gt_map_cls) in zip(
    #             batch_pred_map_cls_dict[prefix],
    #             batch_gt_map_cls_dict[prefix]):
    #         for ap_calculator in ap_calculator_list:
    #             ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    #     # Evaluate average precision
    #     for i, ap_calculator in enumerate(ap_calculator_list):
    #         metrics_dict = ap_calculator.compute_metrics()
    #         self.logger.info(
    #             '=====================>'
    #             f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
    #             '<====================='
    #         )
    #         for key in metrics_dict:
    #             self.logger.info(f'{key} {metrics_dict[key]}')
    #         if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
    #             mAP = metrics_dict['mAP']
    #         mAPs[i][1][prefix] = metrics_dict['mAP']
    #         ap_calculator.reset()
    #
    #     for mAP in mAPs:
    #         self.logger.info(
    #             f'IoU[{mAP[0]}]:\t'
    #             + ''.join([
    #                 f'{key}: {mAP[1][key]:.4f} \t'
    #                 for key in sorted(mAP[1].keys())
    #             ])
    #         )
    #
    #     return None

def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict, strict=True)

    # model.load_state_dict(checkpoint['model'], strict=True)
    if not args.eval and not args.reduce_lr:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['sr3d'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='sr3d')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    # parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)
    parser.add_argument('--reduce_lr', action='store_true')

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    opt = args

    wandb.init(project="BUTD-DETR", name=f"{args.dataset[0]}_run1")

    # torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
