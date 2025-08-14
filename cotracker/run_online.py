# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional

from cotracker.utils.visualizer import Visualizer


class QueryManager:
    def __init__(
        self,
        num_queries,
        device,
        model_window_length=16,
    ):  
        self.device = device
        self.num_queries = num_queries
        self.model_window_length = model_window_length
        self.replace_query_after_num_invisible_timesteps = model_window_length
        
        self.queries = None             # (batch_size, num_queries, txy) with t: timestep of query
        self.pred_tracks = None         # (num_timesteps, num_queries, xy)
        self.pred_visibility = None     # (num_timesteps, num_queries)
        self.resample_indices = None
        self.window_seg_masks = deque(maxlen=model_window_length)

    def __call__(self):
        return self.queries

    def update_tracks_and_visibility_masks(
        self,
        pred_tracks: Optional[torch.Tensor],
        pred_visibility: Optional[torch.Tensor] # (batch_size, timesteps, num_points)
    ):
        if pred_tracks is not None and pred_visibility is not None:
            pred_tracks, pred_visibility = pred_tracks.squeeze(0), pred_visibility.squeeze(0)
            self.pred_tracks, self.pred_visibility = pred_tracks[-self.model_window_length:], pred_visibility[-self.model_window_length:]
        else:
            self.pred_tracks = None
            self.pred_visibility = None

    def check_query_visibility(self) -> Optional[torch.Tensor]:
        """
            Based on the visibility mask of the last model_window_length timesteps, check which points were not visible at all.
        """
        if self.pred_visibility is not None:
            # self.pred_visibility (num_timesteps, num_points)
            visible_once = self.pred_visibility.any(dim=0)
            never_visible = ~visible_once
            never_visible_indices = never_visible.nonzero(as_tuple=True)[0]
            if never_visible_indices.numel() == 0:
                self.resample_indices = None
            else:
                self.resample_indices = never_visible_indices 
        else:
            self.resample_indices = None
             
    def sample_queries_from_mask(
        self,
        timestep_in_window=0,                 # at which timestep in the window to sample new queries
    ):
        """
            Based on the segmentation mask, sample random queries.
            If resample_indices are given, only these queries are getting replaced by newly sampled points.
            # TODO: currently assuming batch size of one

            resample_indices: Optional[torch.Tensor]
        """
        if self.resample_indices is not None:
            num_queries_to_sample = self.resample_indices.shape[0]
        else:
            num_queries_to_sample = self.num_queries
        t = timestep_in_window

        # require 3x3 foreground neighborhood which is free of other queries for a newly sampled query
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.device)
        fg_mask = (self.window_seg_masks[t] > 0).to(dtype=torch.float32)
        if self.resample_indices is not None:
            # change fg to bg for current query coordinates
            current_queries_xy = self.queries[..., 1:].squeeze(0).long()
            fg_query_mask = fg_mask
            fg_query_mask[current_queries_xy[:, 1], current_queries_xy[:, 0]] = 0.0
            mask4conv = fg_query_mask.unsqueeze(0).unsqueeze(0)
        else:
            mask4conv = fg_mask.unsqueeze(0).unsqueeze(0)
        neighbor_count = F.conv2d(mask4conv, kernel, padding=1).squeeze()
        # neighbor_count == 9 means the center + all 8 neighbors are FG
        # TODO: Have hyperparameter not in code!
        candidate_mask = neighbor_count >= 9.0

        if candidate_mask.sum() == 0:
            # Option: candidate_mask = neighbor_count >= 3.0
            raise ValueError("No query candidates to sample from!")
        elif candidate_mask.sum() < num_queries_to_sample:
            # Option: candidate_mask = neighbor_count >= 6.0
            raise ValueError("Number of queries to sample is higher than number of candidates!")
        
        candidates_yx = torch.nonzero(candidate_mask, as_tuple=False)   
        candidates_xy = candidates_yx[:, [1,0]]

        num_candidates = candidates_xy.shape[0]
        if num_candidates >= num_queries_to_sample:
            sampled_indices = torch.randperm(num_candidates)[:num_queries_to_sample]
        else:
            # TODO: not optimal approach
            # it is better to determine num_of_queries based on object size
            raise ValueError("Number of queries to sample is higher than number of candidates!")
        
        subsampled_xy_coords = candidates_xy[sampled_indices, :].float()

        # image = np.zeros(mask.shape)
        # subsampled_xy_coords_np = subsampled_xy_coords.cpu().numpy()
        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.scatter(subsampled_xy_coords_np[:,0], subsampled_xy_coords_np[:,1], c='red', s=20)  # x then y
        # plt.title("Segmentation Mask with Queries")
        # plt.savefig("init_queries_on_mask.png", bbox_inches='tight', pad_inches=0)

        if self.resample_indices is not None:
            # queries: (batch_size, num_points, txy)
            self.queries = torch.ones(size=(1, self.num_queries, 3), device=self.device) * t
            # tracks at t as new query points (still visible)
            self.queries[..., 1:] = self.pred_tracks[t]             # (num_points, xy)
            self.queries[0, self.resample_indices, 1:] = subsampled_xy_coords     
        else:
            self.queries = torch.ones(size=(1, self.num_queries, 3), device=self.device) * t
            self.queries[0, :, 1:] = subsampled_xy_coords
            

class OnlineCoTracking:
    def __init__(
        self,
        num_queries,
        grid_query_frame,
        record_video,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        self.grid_query_frame = grid_query_frame
        self.is_first_step = True

        self.window_frames = deque(maxlen=self.model.model.window_len)
        self.window_seg_masks = deque(maxlen=self.model.model.window_len)
        self.pred_tracks = None
        self.pred_visibility = None

        self.query_manager = QueryManager(
            num_queries, 
            model_window_length=self.model.model.window_len,
            device=self.device)

        self.record_video = record_video
        if self.record_video:
            self.video_frames = []
            self.vis = Visualizer(
                save_dir="./saved_videos", 
                pad_value=120, 
                linewidth=2, 
                mode='rainbow', 
                tracks_leave_trace=50)
        self.processed_frames = 0

    def process_window(
        self, 
        new_rgb_frame,
        new_segmentation_mask,
    ):  
        self.window_frames.append(new_rgb_frame)
        self.query_manager.window_seg_masks.append(new_segmentation_mask)

        if self.processed_frames == 0:
            # first forward pass of model at t=15 such that it gets the timestep for initialization
            # starting tracking at t=16 with initial seg_mask at t=0
            self.query_manager.sample_queries_from_mask(timestep_in_window=0)
            self.is_first_step = True
        else:
            self.query_manager.check_query_visibility()
            if self.query_manager.resample_indices is not None:
                # model needs one timestep to track again after it has received first_step_flag
                # in this time, the window of frames already shifts by one 
                # seg_mask at previous timestep 1 is now timestep 0
                self.query_manager.sample_queries_from_mask(timestep_in_window=1)
                self.is_first_step = True
        
        pred_tracks, pred_visibility = None, None
        # whenever the model experiences first step, it processes the queries and outputs None
        # maxlen-1 needed to process the initial frame 
        if len(self.window_frames) >= (self.window_frames.maxlen-1):
            video_chunk = torch.stack(
                tuple(self.window_frames)).float().permute(0, 3, 1, 2).unsqueeze(0) # (1, T, 3, H, W)
            start = time.time()
            pred_tracks, pred_visibility = self.model(
                video_chunk,
                queries=self.query_manager(),
                is_first_step=self.is_first_step,
            )
            print(f"Point Tracking took: {time.time() - start} seconds")
            
            self.query_manager.update_tracks_and_visibility_masks(pred_tracks, pred_visibility)
            
            if self.is_first_step == True:
                self.is_first_step = False
        
        self.processed_frames += 1
        print(f"Processed Frames: {self.processed_frames}")
        
        if self.record_video:
            self.video_frames.append(new_rgb_frame)
            if self.query_manager.resample_indices is not None:
                # save previous tracking for recording  
                if self.pred_tracks is not None and self.pred_visibility is not None:
                    # window is shifted t+1, queries get resetted, model outputs None as new first step
                    # window is shifted t+2, model tracks again 
                    num_old_frames_updated_with_new_queries = self.model.model.window_len - 2
                    self.pred_tracks = torch.concatenate(
                        [
                            self.pred_tracks[:, :-num_old_frames_updated_with_new_queries, ...], 
                            pred_tracks
                        ], dim=1)
                    self.pred_visibility = torch.concatenate(
                        [
                            self.pred_visibility[:, :-num_old_frames_updated_with_new_queries, ...], 
                            pred_visibility
                        ], dim=1)
                else:
                    self.pred_tracks = pred_tracks
                    self.pred_visibility = pred_visibility

            if len(self.video_frames) == 100:
                self.save_video_with_point_tracks()
            
        return pred_tracks, pred_visibility, self.query_manager.resample_indices

    def save_video_with_point_tracks(
        self,
    ):  
        print("Saving video of point tracking")
        video = torch.stack(self.video_frames).permute(0, 3, 1, 2).unsqueeze(0)
        file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.vis.visualize(
            video[:, :-1, ...], 
            self.pred_tracks, 
            self.pred_visibility, 
            query_frame=self.grid_query_frame,
            filename=file_name,
        )
