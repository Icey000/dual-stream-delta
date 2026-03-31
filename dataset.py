from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json
from collections import Counter
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import getMetaDataTask
from torch.utils.data.dataloader import default_collate
import numpy as np
from transformers import AutoTokenizer

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


def _init_spotting_targets(num_clips, num_classes):
    targets = np.zeros((num_clips, num_classes + 1), dtype=np.float32)
    targets[:, 0] = 1.0
    return targets


def _init_center_regression_targets(num_clips):
    return np.zeros((num_clips,), dtype=np.float32), np.zeros((num_clips,), dtype=np.float32)


def _compute_normalized_center_offset(frame, clip_index, window_size_frame):
    clip_center = (clip_index + 0.5) * float(window_size_frame)
    offset = (float(frame) - clip_center) / float(window_size_frame)
    return float(np.clip(offset, -1.0, 1.0))


def _set_hard_spotting_target(targets, clip_index, class_index):
    if clip_index < 0 or clip_index >= targets.shape[0]:
        return
    targets[clip_index, :] = 0.0
    targets[clip_index, class_index + 1] = 1.0


def _set_soft_spotting_targets(targets, center_clip, class_index, radius=2, sigma=1.0):
    if center_clip < 0 or center_clip >= targets.shape[0]:
        return

    radius = max(0, int(radius))
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1.0

    for delta in range(-radius, radius + 1):
        clip_index = center_clip + delta
        if clip_index < 0 or clip_index >= targets.shape[0]:
            continue

        event_prob = float(np.exp(-((delta ** 2) / (2.0 * sigma * sigma))))
        event_prob = min(max(event_prob, 0.0), 1.0)
        existing_event_prob = float(targets[clip_index, 1:].max()) if targets.shape[1] > 1 else 0.0
        if event_prob < existing_event_prob:
            continue

        targets[clip_index, :] = 0.0
        targets[clip_index, 0] = 1.0 - event_prob
        targets[clip_index, class_index + 1] = event_prob


def _build_spotting_targets(
    num_clips,
    num_classes,
    annotations,
    dict_event,
    framerate,
    window_size_frame,
    target_mode="hard_multiclass",
    soft_window_radius=2,
    soft_window_sigma=1.0,
    build_center_targets=False,
    center_positive_threshold=0.5,
):
    targets = _init_spotting_targets(num_clips, num_classes)
    offset_targets, offset_masks = _init_center_regression_targets(num_clips)

    for annotation in annotations:
        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])
        minutes, seconds = time.split(' ')[-1].split(':')
        minutes, seconds = int(minutes), int(seconds)
        frame = framerate * (seconds + 60 * minutes)

        if event not in dict_event or half > 2:
            continue

        class_index = dict_event[event]
        clip_index = frame // window_size_frame
        center_offset = _compute_normalized_center_offset(frame, clip_index, window_size_frame)

        if target_mode == "soft_window_multiclass":
            _set_soft_spotting_targets(
                targets,
                clip_index,
                class_index,
                radius=soft_window_radius,
                sigma=soft_window_sigma,
            )
            for delta in range(-max(0, int(soft_window_radius)), max(0, int(soft_window_radius)) + 1):
                target_clip_index = clip_index + delta
                if target_clip_index < 0 or target_clip_index >= num_clips:
                    continue

                event_prob = float(np.exp(-((delta ** 2) / (2.0 * float(max(soft_window_sigma, 1e-6)) ** 2))))
                existing_event_prob = float(targets[target_clip_index, 1:].max()) if targets.shape[1] > 1 else 0.0
                if event_prob + 1e-8 < existing_event_prob:
                    continue

                if build_center_targets:
                    offset_targets[target_clip_index] = _compute_normalized_center_offset(
                        frame,
                        target_clip_index,
                        window_size_frame,
                    )
                    offset_masks[target_clip_index] = 1.0 if event_prob >= float(center_positive_threshold) else 0.0
        else:
            _set_hard_spotting_target(targets, clip_index, class_index)
            if build_center_targets and 0 <= clip_index < num_clips:
                offset_targets[clip_index] = center_offset
                offset_masks[clip_index] = 1.0

    if build_center_targets:
        return targets, offset_targets, offset_masks
    return targets, None, None

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    captions = [t[-1] for t in batch]
    idx = [t[-3:-1] for t in batch]
    ## padd
    tokens = [([SOS_TOKEN] + t[-4] + [EOS_TOKEN]) if t[-4] else [PAD_TOKEN, PAD_TOKEN] for t in batch]
    tokens = [torch.Tensor(t).long() for t in tokens ]
    ## get sequence lengths
    lengths = torch.tensor([ len(t) for t in tokens ])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    ## compute mask
    mask = (tokens != PAD_TOKEN)
    return default_collate([t[:-4] for t in batch ]) + [tokens], lengths, mask, captions, idx

class CollateGPT:
    def __init__(self, llm_model_path="Qwen/Qwen2.5-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        captions = [t[-1] for t in batch]
        idx = [t[-3:-1] for t in batch]
        
        # token化: 添加 BOS (":") 和 EOS
        bos_ids = self.tokenizer.encode(":", add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        
        tokens = [
            (bos_ids + t[-4] + [eos_id]) if t[-4] else [self.tokenizer.pad_token_id, self.tokenizer.pad_token_id]
            for t in batch
        ]
        tokens = [torch.Tensor(t).long() for t in tokens]
        
        ## get sequence lengths
        lengths = torch.tensor([len(t) for t in tokens])
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        ## compute mask
        mask = (tokens != self.tokenizer.pad_token_id)
        
        return default_collate([t[:-4] for t in batch]) + [tokens], lengths, mask, captions, idx



def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]

class SoccerNetClips(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting training phase.
    """
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=2, 
                framerate=2, window_size=15, target_mode="hard_multiclass",
                soft_window_radius=2, soft_window_sigma=1.0,
                build_center_targets=False, center_positive_threshold=0.5):
        self.path = path
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.target_mode = target_mode
        self.soft_window_radius = int(soft_window_radius)
        self.soft_window_sigma = float(soft_window_sigma)
        self.build_center_targets = bool(build_center_targets)
        self.center_positive_threshold = float(center_positive_threshold)
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)

        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()
        self.game_offset_targets = list()
        self.game_offset_masks = list()

        # NARY_CAPTION_V2 = {"corner" : 0,"substitution" : 0,"y-card" : 0,"whistle" : 0,"soccer-ball" : 0,"injury" : 0,"penalty" :
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features), mmap_mode="r")
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features), mmap_mode="r")
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1, offset_half1, offset_mask_half1 = _build_spotting_targets(
                feat_half1.shape[0],
                self.num_classes,
                [ann for ann in labels["annotations"] if int(ann["gameTime"][0]) == 1],
                self.dict_event,
                framerate=framerate,
                window_size_frame=self.window_size_frame,
                target_mode=self.target_mode,
                soft_window_radius=self.soft_window_radius,
                soft_window_sigma=self.soft_window_sigma,
                build_center_targets=self.build_center_targets,
                center_positive_threshold=self.center_positive_threshold,
            )
            label_half2, offset_half2, offset_mask_half2 = _build_spotting_targets(
                feat_half2.shape[0],
                self.num_classes,
                [ann for ann in labels["annotations"] if int(ann["gameTime"][0]) == 2],
                self.dict_event,
                framerate=framerate,
                window_size_frame=self.window_size_frame,
                target_mode=self.target_mode,
                soft_window_radius=self.soft_window_radius,
                soft_window_sigma=self.soft_window_sigma,
                build_center_targets=self.build_center_targets,
                center_positive_threshold=self.center_positive_threshold,
            )
            
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)
            if self.build_center_targets:
                self.game_offset_targets.append(offset_half1)
                self.game_offset_targets.append(offset_half2)
                self.game_offset_masks.append(offset_mask_half1)
                self.game_offset_masks.append(offset_mask_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)
        if self.build_center_targets:
            self.game_offset_targets = np.concatenate(self.game_offset_targets)
            self.game_offset_masks = np.concatenate(self.game_offset_masks)
        else:
            self.game_offset_targets = None
            self.game_offset_masks = None



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        if self.build_center_targets:
            return (
                self.game_feats[index, :, :],
                self.game_labels[index, :],
                self.game_offset_targets[index],
                self.game_offset_masks[index],
            )
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)

class SoccerNetClipsTesting(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting inference phase.
    """
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=2, 
                framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], task="caption", verbose=False,randomized=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))


        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = self.framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

class SoccerNetCaptions(Dataset):
    """
    This class is used to download and pre-compute clips and captions from the SoccerNet dataset for captining training phase.
    """
    def __init__(self, path, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15, aug_ratio=0.0, llm_model_path="Qwen/Qwen2.5-7B"):
        self.path = path
        self.split = split
        split = [s for s in split if s!= "challenge"]
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], task="caption",split=split, verbose=False,randomized=True)

        self.data = list()
        self.game_feats = list()

        l_pad = self.window_size_frame//2 + self.window_size_frame%2
        r_pad = self.window_size_frame//2 
        looper = self.listGames

        for game_id, game in enumerate(tqdm(looper)):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features), mmap_mode="r")
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features), mmap_mode="r")
            
            self.game_feats.append((feat_half1, feat_half2)) 

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))
            
            for caption_id, annotation in enumerate(labels["annotations"]):

                time = annotation["gameTime"]
                event = annotation["label"]
                half = int(time[0])
                if event not in self.dict_event or half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( seconds + 60 * minutes)

                self.data.append(((game_id, half-1, frame) , (caption_id, annotation['anonymized'])))
        
        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        #launch a TextProcessor that will tokenize a caption
        self.text_processor = HFTextProcessor(llm_model_path)
        self.vocab_size = len(self.text_processor.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            caption_tokens (np.array): tokens of captions.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
            caption (List[strings]): list of original captions.
        """
        clip_id, (caption_id, caption) = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)
        caption_tokens = self.text_processor(caption)    
        
        return vfeats, caption_tokens, clip_id[0], caption_id, caption
    
    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus
    
    def detokenize(self, tokens, remove_EOS=False):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string
        #return string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}") if remove_EOS else string
    
class SoccerNetClassification(Dataset):
    """
    This class is used to download and pre-compute clips and captions from the SoccerNet dataset for captining training phase.
    """
    def __init__(self, path, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15):
        self.path = path
        split = [s for s in split if s!= "challenge"]
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.class_labels = [k for k in self.dict_event.keys()]

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], task="caption",split=split, verbose=False,randomized=True)

        self.data = list()
        self.game_feats = list()

        l_pad = self.window_size_frame//2 + self.window_size_frame%2
        r_pad = self.window_size_frame//2 
        looper = self.listGames
        # if split == ["train"]:
        #     looper = self.listGames[103:120]
        # else:
        #     looper = self.listGames[10:20]
        for game_id, game in enumerate(tqdm(looper)):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features), mmap_mode="r")
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features), mmap_mode="r")
            
            self.game_feats.append((feat_half1, feat_half2)) 

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))
            
            for caption_id, annotation in enumerate(labels["annotations"]):

                time = annotation["gameTime"]
                event = annotation["label"]
                half = int(time[0])
                if event not in self.dict_event or half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( seconds + 60 * minutes) 
                
                self.data.append(((game_id, half-1, frame), self.class_labels.index(event)))
        
        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        #launch a TextProcessor that will tokenize a caption

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            caption_tokens (np.array): tokens of captions.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
            caption (List[strings]): list of original captions.
        """
        clip_id, class_label = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)
        
        return vfeats, class_label
    

class SoccerNetVideoProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length
        self.l_pad = clip_length // 2 + clip_length % 2
        self.r_pad = clip_length // 2

    def __call__(self, video_fn, feats):
        video_id, half, frame = video_fn
        video_feature = feats[video_id][half]
        
        original_length = video_feature.shape[0]
        # Simulate padded length
        padded_length = original_length + self.l_pad + self.r_pad
        # Keep original logic which assumes the feature length is padded_length
        start = min(max(frame, 0), max(0, padded_length - self.clip_length))
        
        real_start = start - self.l_pad
        real_end = real_start + self.clip_length
        
        # Use np.clip to handle out-of-bounds indices by edge padding
        indices = np.clip(np.arange(real_start, real_end), 0, max(0, original_length - 1))
        
        return video_feature[indices]

class HFTextProcessor(object):
    def __init__(self, llm_model_path="Qwen/Qwen2.5-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Compatibility with legacy dataset.py
        self.vocab = [True] * len(self.tokenizer)

    def __call__(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, tokens):
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], torch.Tensor):
            tokens = tokens[0].tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
       


class PredictionCaptions(Dataset):
    def __init__(self, SoccerNetPath, PredictionPath, features="ResNET_TF2_PCA512.npy", split=["train"], version=2, framerate=2, window_size=15, llm_model_path="Qwen/Qwen2.5-7B"):
        self.path = SoccerNetPath
        self.PredictionPath = PredictionPath
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        self.labels, _, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.split = split

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(self.path)
        downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], task="caption", split=split, verbose=False,randomized=True)

        self.data = list()
        self.game_feats = list()

        l_pad = self.window_size_frame//2 + self.window_size_frame%2
        r_pad = self.window_size_frame//2 

        for game_id, game in enumerate(tqdm(self.listGames)):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features), mmap_mode="r")
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features), mmap_mode="r")
            
            self.game_feats.append((feat_half1, feat_half2)) 

            # Load labels
            preds = json.load(open(os.path.join(self.PredictionPath, game, "results_spotting.json")))
            
            for caption_id, annotation in enumerate(preds["predictions"]):

                if annotation["label"] not in self.dict_event:
                    continue

                time = annotation["gameTime"]
                half = int(time[0])
                if half > 2:
                    continue

                minutes, seconds = time.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * ( int(seconds) + 60 * int(minutes)) 
                
                self.data.append(((game_id, half-1, frame), caption_id))
        
        #launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        #launch a TextProcessor that will tokenize a caption
        self.text_processor = HFTextProcessor(llm_model_path)
        self.vocab_size = len(self.text_processor.vocab)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
        """
        clip_id, caption_id = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)   
        return vfeats, clip_id[0], caption_id


    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return string
    
    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in json.load(open(os.path.join(self.path, game, self.labels)))["annotations"]]
        return corpus
