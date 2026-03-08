"""
dataset_dual.py - 双模态数据集 (Video + Audio)

为双流晚期融合架构提供数据加载：
- SoccerNetCaptionsDual: Captioning 训练/验证数据集
- SoccerNetClipsDual: Spotting 训练数据集 (clip 级别)
- SoccerNetClipsTestingDual: Spotting 推理数据集 (整场比赛)
- collate_fn_gpt_dual: 双模态 GPT collate 函数

核心特性:
    1. 支持双根目录 (vision_root / audio_root)
    2. 在 __init__ 中执行严格的三重交集校验
    3. __getitem__ 同时加载视觉和音频 .npy 特征
"""

from torch.utils.data import Dataset
import numpy as np
import os
import logging
import json
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import default_collate

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import getMetaDataTask
import tiktoken

# 从原有 dataset.py 复用常量和处理器
from dataset import (
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
    feats2clip,
    SoccerNetVideoProcessor,
    TikTokenTextProcessor,
)


# ============================================================================
#  辅助函数: 校验比赛半场的完整性
# ============================================================================

def _check_half_valid(vision_root, audio_root, game, half_idx, vision_features):
    """
    严格的三重校验: 只有视觉特征、标签、音频特征三者同时存在时才返回 True。

    Args:
        vision_root (str):   视觉特征与标签的根目录
        audio_root (str):    音频特征的根目录
        game (str):          比赛相对路径 (如 "england_epl/2014-2015/xxx")
        half_idx (int):      半场编号 (1 或 2)
        vision_features (str): 视觉特征文件名 (如 "baidu_soccer_embeddings.npy")

    Returns:
        bool: 三重校验是否通过
    """
    # 1. 视觉特征 .npy
    vision_feat_path = os.path.join(vision_root, game, "{}_{}" .format(half_idx, vision_features))
    if not os.path.isfile(vision_feat_path):
        return False

    # 2. 标签 Labels-caption.json
    label_path = os.path.join(vision_root, game, "Labels-caption.json")
    if not os.path.isfile(label_path):
        return False

    # 3. 音频特征 _audio_clap.npy
    audio_feat_path = os.path.join(audio_root, game, "{}_audio_clap.npy".format(half_idx))
    if not os.path.isfile(audio_feat_path):
        return False

    return True


def _validate_game(vision_root, audio_root, game, vision_features):
    """
    校验一场比赛的两个半场，返回有效的半场列表。

    Returns:
        valid_halves: list of int, 例如 [1, 2] 或 [1] 或 []
    """
    valid = []
    for h in [1, 2]:
        if _check_half_valid(vision_root, audio_root, game, h, vision_features):
            valid.append(h)
    return valid


# ============================================================================
#  1. Captioning 双模态数据集
# ============================================================================

class SoccerNetCaptionsDual(Dataset):
    """
    双模态 Captioning 数据集。

    同时加载视觉特征 (百度) 和音频特征 (LAION-CLAP)，
    在 __init__ 中执行严格的三重交集校验，
    确保只有同时具备视觉、音频、标签的比赛才被使用。

    Args:
        vision_root (str):      视觉特征与标签的根目录
        audio_root (str):       音频特征的根目录
        features (str):         视觉特征文件名
        split (list):           数据集分割 ["train"], ["valid"], ["test"]
        version (int):          SoccerNet 版本
        framerate (int):        帧率
        window_size (int):      窗口大小 (秒)
    """

    def __init__(
        self,
        vision_root,
        audio_root,
        features="baidu_soccer_embeddings.npy",
        split=None,
        version=2,
        framerate=2,
        window_size=15,
    ):
        if split is None:
            split = ["train"]

        self.vision_root = vision_root
        self.audio_root = audio_root
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version
        self.framerate = framerate

        # 获取比赛列表 (排除 challenge 集)
        clean_split = [s for s in split if s != "challenge"]
        self.listGames = getListGames(clean_split, task="caption")

        # 获取元数据
        self.labels_filename, self.num_classes, self.dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )

        # ---------- 数据预加载 + 三重校验 ----------
        self.data = []           # 存储 ((game_id, half_idx, frame), (caption_id, caption_str))
        self.game_feats = []     # 存储 (视觉特征_half1, 视觉特征_half2) 元组
        self.game_audio = []     # 存储 (音频特征_half1, 音频特征_half2) 元组

        l_pad = self.window_size_frame // 2 + self.window_size_frame % 2
        r_pad = self.window_size_frame // 2

        # 重新排列 game_id: 只给通过校验的比赛分配 ID
        valid_game_id = 0
        skipped = 0

        for game in tqdm(self.listGames, desc="Loading dual-modal data"):
            # ---------- 三重校验 ----------
            valid_halves = _validate_game(vision_root, audio_root, game, features)
            if not valid_halves:
                # 两个半场都不满足条件，跳过整场比赛
                logging.warning("Skipping game (no valid halves): {}".format(game))
                skipped += 1
                continue

            # 检查是否两个半场都有效 (标签文件是整场共享的)
            has_h1 = 1 in valid_halves
            has_h2 = 2 in valid_halves

            if not (has_h1 and has_h2):
                logging.warning(
                    "Game '{}' only has valid half(s): {}. Skipping entirely for data consistency.".format(
                        game, valid_halves
                    )
                )
                skipped += 1
                continue

            # ---------- 加载视觉特征 ----------
            feat_half1 = np.load(os.path.join(vision_root, game, "1_" + features))
            feat_half1 = np.pad(
                feat_half1.reshape(-1, feat_half1.shape[-1]),
                ((l_pad, r_pad), (0, 0)), "edge"
            )
            feat_half2 = np.load(os.path.join(vision_root, game, "2_" + features))
            feat_half2 = np.pad(
                feat_half2.reshape(-1, feat_half2.shape[-1]),
                ((l_pad, r_pad), (0, 0)), "edge"
            )
            self.game_feats.append((feat_half1, feat_half2))

            # ---------- 加载音频特征 ----------
            audio_half1 = np.load(os.path.join(audio_root, game, "1_audio_clap.npy"))
            audio_half1 = audio_half1.reshape(-1, audio_half1.shape[-1])
            # 对音频特征做与视觉相同的 padding，使窗口对齐
            audio_half1 = np.pad(audio_half1, ((l_pad, r_pad), (0, 0)), "edge")

            audio_half2 = np.load(os.path.join(audio_root, game, "2_audio_clap.npy"))
            audio_half2 = audio_half2.reshape(-1, audio_half2.shape[-1])
            audio_half2 = np.pad(audio_half2, ((l_pad, r_pad), (0, 0)), "edge")
            self.game_audio.append((audio_half1, audio_half2))

            # ---------- 加载标签 ----------
            label_path = os.path.join(vision_root, game, self.labels_filename)
            labels = json.load(open(label_path))

            for caption_id, annotation in enumerate(labels["annotations"]):
                time_str = annotation["gameTime"]
                event = annotation["label"]
                half = int(time_str[0])

                if event not in self.dict_event or half > 2:
                    continue

                minutes, seconds = time_str.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)

                self.data.append(
                    ((valid_game_id, half - 1, frame), (caption_id, annotation["anonymized"]))
                )

            valid_game_id += 1

        logging.info(
            "Dual dataset loaded: {} valid games, {} skipped, {} samples".format(
                valid_game_id, skipped, len(self.data)
            )
        )

        # ---------- 视频/音频裁剪器 ----------
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)

        # ---------- 文本处理器 ----------
        self.text_processor = TikTokenTextProcessor(self._get_corpus(clean_split))
        self.vocab_size = len(self.text_processor.vocab)

    def _get_corpus(self, split):
        """构建词表所需的语料库 (只从有效比赛中提取)"""
        corpus = []
        for game in getListGames(["train"], task="caption"):
            label_path = os.path.join(self.vision_root, game, self.labels_filename)
            if os.path.isfile(label_path):
                annotations = json.load(open(label_path))["annotations"]
                corpus.extend([ann["anonymized"] for ann in annotations])
        return corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            vfeats (np.array):      视觉特征 clip [window_size_frame, D_vid]
            afeats (np.array):      音频特征 clip [window_size_frame, D_aud]
            caption_tokens (list):  caption 的 token ID 列表
            game_id (int):          比赛 ID
            caption_id (int):       caption ID
            caption (str):          原始 caption 字符串
        """
        clip_id, (caption_id, caption) = self.data[idx]

        # 视觉特征裁剪
        vfeats = self.video_processor(clip_id, self.game_feats)

        # 音频特征裁剪 (使用相同的裁剪逻辑)
        afeats = self.video_processor(clip_id, self.game_audio)

        # 文本 tokenize
        caption_tokens = self.text_processor(caption)

        return vfeats, afeats, caption_tokens, clip_id[0], caption_id, caption

    def getCorpus(self, split=None):
        if split is None:
            split = ["train"]
        return self._get_corpus(split)

    def detokenize(self, tokens, remove_EOS=False):
        return self.text_processor.detokenize(tokens)


# ============================================================================
#  2. Spotting 训练双模态数据集
# ============================================================================

class SoccerNetClipsDual(Dataset):
    """
    双模态 Spotting 训练数据集 (clip 级别)。

    与原有 SoccerNetClips 对齐，但同时加载视觉和音频特征。
    在 __init__ 中执行三重交集校验。
    """

    def __init__(
        self,
        vision_root,
        audio_root,
        features="baidu_soccer_embeddings.npy",
        split=None,
        version=2,
        framerate=2,
        window_size=15,
    ):
        if split is None:
            split = ["train"]

        self.vision_root = vision_root
        self.audio_root = audio_root
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version

        self.listGames = getListGames(split, task="caption")
        labels_filename, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels_filename = labels_filename
        self.num_classes = num_classes
        self.dict_event = dict_event

        logging.info("Pre-compute dual-modal clips for spotting training")

        self.game_vfeats = []
        self.game_afeats = []
        self.game_labels = []
        skipped = 0

        for game in tqdm(self.listGames, desc="Loading dual clips"):
            # 三重校验
            valid_halves = _validate_game(vision_root, audio_root, game, features)
            if len(valid_halves) < 2:
                skipped += 1
                continue

            # 加载视觉特征并切 clip
            feat_half1 = np.load(os.path.join(vision_root, game, "1_" + features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(vision_root, game, "2_" + features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            vclip_h1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            vclip_h2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # 加载音频特征并切 clip
            audio_half1 = np.load(os.path.join(audio_root, game, "1_audio_clap.npy"))
            audio_half1 = audio_half1.reshape(-1, audio_half1.shape[-1])
            audio_half2 = np.load(os.path.join(audio_root, game, "2_audio_clap.npy"))
            audio_half2 = audio_half2.reshape(-1, audio_half2.shape[-1])

            aclip_h1 = feats2clip(torch.from_numpy(audio_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            aclip_h2 = feats2clip(torch.from_numpy(audio_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # 确保视觉和音频 clip 数量一致 (取较小者)
            n_clips_h1 = min(vclip_h1.shape[0], aclip_h1.shape[0])
            n_clips_h2 = min(vclip_h2.shape[0], aclip_h2.shape[0])
            vclip_h1 = vclip_h1[:n_clips_h1]
            aclip_h1 = aclip_h1[:n_clips_h1]
            vclip_h2 = vclip_h2[:n_clips_h2]
            aclip_h2 = aclip_h2[:n_clips_h2]

            # 加载标签
            labels = json.load(open(os.path.join(vision_root, game, self.labels_filename)))

            label_half1 = np.zeros((n_clips_h1, self.num_classes + 1))
            label_half1[:, 0] = 1  # 默认为背景类
            label_half2 = np.zeros((n_clips_h2, self.num_classes + 1))
            label_half2[:, 0] = 1

            for annotation in labels["annotations"]:
                time_str = annotation["gameTime"]
                event = annotation["label"]
                half = int(time_str[0])

                minutes, seconds = time_str.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)

                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                if half == 1 and frame // self.window_size_frame < n_clips_h1:
                    label_half1[frame // self.window_size_frame][0] = 0
                    label_half1[frame // self.window_size_frame][label + 1] = list(self.dict_event.keys()).index(event) + 1
                if half == 2 and frame // self.window_size_frame < n_clips_h2:
                    label_half2[frame // self.window_size_frame][0] = 0
                    label_half2[frame // self.window_size_frame][label + 1] = list(self.dict_event.keys()).index(event) + 1

            self.game_vfeats.append(vclip_h1)
            self.game_vfeats.append(vclip_h2)
            self.game_afeats.append(aclip_h1)
            self.game_afeats.append(aclip_h2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        if self.game_vfeats:
            self.game_vfeats = np.concatenate(self.game_vfeats)
            self.game_afeats = np.concatenate(self.game_afeats)
            self.game_labels = np.concatenate(self.game_labels)
        else:
            self.game_vfeats = np.array([])
            self.game_afeats = np.array([])
            self.game_labels = np.array([])

        logging.info("Dual clips loaded: {} clips, {} games skipped".format(len(self.game_vfeats), skipped))

    def __getitem__(self, index):
        return self.game_vfeats[index, :, :], self.game_afeats[index, :, :], self.game_labels[index, :]

    def __len__(self):
        return len(self.game_vfeats)


# ============================================================================
#  3. Spotting 推理双模态数据集
# ============================================================================

class SoccerNetClipsTestingDual(Dataset):
    """
    双模态 Spotting 推理数据集 (整场比赛级别)。

    __getitem__ 返回整个半场的特征和标签。
    """

    def __init__(
        self,
        vision_root,
        audio_root,
        features="baidu_soccer_embeddings.npy",
        split=None,
        version=2,
        framerate=2,
        window_size=15,
    ):
        if split is None:
            split = ["test"]

        self.vision_root = vision_root
        self.audio_root = audio_root
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.version = version
        self.split = split

        all_games = getListGames(split, task="caption")
        labels_filename, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)
        self.labels_filename = labels_filename
        self.num_classes = num_classes
        self.dict_event = dict_event

        # 过滤有效比赛
        self.listGames = []
        for game in all_games:
            valid_halves = _validate_game(vision_root, audio_root, game, features)
            if len(valid_halves) == 2:
                self.listGames.append(game)
            else:
                logging.warning("Skipping test game (incomplete): {}".format(game))

        self.path = vision_root  # 兼容原有 test_spotting 中使用 dataloader.dataset.path

    def __getitem__(self, index):
        game = self.listGames[index]

        # 视觉特征
        feat_half1 = np.load(os.path.join(self.vision_root, game, "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.vision_root, game, "2_" + self.features))

        # 音频特征
        audio_half1 = np.load(os.path.join(self.audio_root, game, "1_audio_clap.npy"))
        audio_half1 = audio_half1.reshape(-1, audio_half1.shape[-1])
        audio_half2 = np.load(os.path.join(self.audio_root, game, "2_audio_clap.npy"))
        audio_half2 = audio_half2.reshape(-1, audio_half2.shape[-1])

        # 标签
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        label_path = os.path.join(self.vision_root, game, self.labels_filename)
        if os.path.exists(label_path):
            labels = json.load(open(label_path))
            for annotation in labels["annotations"]:
                time_str = annotation["gameTime"]
                event = annotation["label"]
                half = int(time_str[0])
                minutes, seconds = time_str.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = self.framerate * (seconds + 60 * minutes)

                if event not in self.dict_event or half > 2:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0] - 1)
                    label_half1[frame][label] = value
                if half == 2:
                    frame = min(frame, feat_half2.shape[0] - 1)
                    label_half2[frame][label] = value

        # 切 clip (视觉)
        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=1, off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )
        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=1, off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        # 切 clip (音频)
        audio_half1 = feats2clip(
            torch.from_numpy(audio_half1),
            stride=1, off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )
        audio_half2 = feats2clip(
            torch.from_numpy(audio_half2),
            stride=1, off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        # === 🌟 就是加了下面这几行强行对齐！ ===
        # 上半场对齐
        n_h1 = min(feat_half1.shape[0], audio_half1.shape[0])
        feat_half1 = feat_half1[:n_h1]     # 把长的那个拦腰砍断
        audio_half1 = audio_half1[:n_h1]   # 强行逼迫大家都一样长
        label_half1 = label_half1[:n_h1]   # 标签也跟着砍

        # 下半场对齐
        n_h2 = min(feat_half2.shape[0], audio_half2.shape[0])
        feat_half2 = feat_half2[:n_h2]
        audio_half2 = audio_half2[:n_h2]
        label_half2 = label_half2[:n_h2]

        return (game,
                feat_half1, feat_half2,
                audio_half1, audio_half2,
                label_half1, label_half2)

    def __len__(self):
        return len(self.listGames)


# ============================================================================
#  4. Classification 双模态数据集
# ============================================================================

class SoccerNetClassificationDual(Dataset):
    """
    双模态事件分类数据集 (用于编码器预训练)。
    """

    def __init__(
        self,
        vision_root,
        audio_root,
        features="baidu_soccer_embeddings.npy",
        split=None,
        version=2,
        framerate=2,
        window_size=15,
    ):
        if split is None:
            split = ["train"]

        self.vision_root = vision_root
        self.audio_root = audio_root
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version

        clean_split = [s for s in split if s != "challenge"]
        self.listGames = getListGames(clean_split, task="caption")
        self.labels_filename, self.num_classes, self.dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )
        self.class_labels = [k for k in self.dict_event.keys()]

        self.data = []
        self.game_feats = []
        self.game_audio = []

        l_pad = self.window_size_frame // 2 + self.window_size_frame % 2
        r_pad = self.window_size_frame // 2

        valid_game_id = 0
        for game in tqdm(self.listGames, desc="Loading dual classification data"):
            valid_halves = _validate_game(vision_root, audio_root, game, features)
            if len(valid_halves) < 2:
                continue

            # 视觉
            feat_h1 = np.load(os.path.join(vision_root, game, "1_" + features))
            feat_h1 = np.pad(feat_h1.reshape(-1, feat_h1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            feat_h2 = np.load(os.path.join(vision_root, game, "2_" + features))
            feat_h2 = np.pad(feat_h2.reshape(-1, feat_h2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            self.game_feats.append((feat_h1, feat_h2))

            # 音频
            audio_h1 = np.load(os.path.join(audio_root, game, "1_audio_clap.npy"))
            audio_h1 = np.pad(audio_h1.reshape(-1, audio_h1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            audio_h2 = np.load(os.path.join(audio_root, game, "2_audio_clap.npy"))
            audio_h2 = np.pad(audio_h2.reshape(-1, audio_h2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            self.game_audio.append((audio_h1, audio_h2))

            # 标签
            labels = json.load(open(os.path.join(vision_root, game, self.labels_filename)))
            for caption_id, annotation in enumerate(labels["annotations"]):
                time_str = annotation["gameTime"]
                event = annotation["label"]
                half = int(time_str[0])
                if event not in self.dict_event or half > 2:
                    continue
                minutes, seconds = time_str.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)
                self.data.append(((valid_game_id, half - 1, frame), self.class_labels.index(event)))

            valid_game_id += 1

        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id, class_label = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)
        afeats = self.video_processor(clip_id, self.game_audio)
        return vfeats, afeats, class_label


# ============================================================================
#  5. 双模态 Collate 函数
# ============================================================================

def collate_fn_gpt_dual(batch):
    """
    双模态 GPT 版 collate 函数。

    batch 中每个元素为:
        (vfeats, afeats, caption_tokens, game_id, caption_id, caption_str)

    Returns:
        (vfeats_batch, afeats_batch, tokens_batch), lengths, mask, captions, idx
    """
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    encode = lambda s: enc.encode(s, allowed_special=set())

    vfeats_list = [torch.tensor(t[0], dtype=torch.float32) for t in batch]
    afeats_list = [torch.tensor(t[1], dtype=torch.float32) for t in batch]
    captions = [t[-1] for t in batch]         # 原始 caption 字符串
    idx = [(t[-3], t[-2]) for t in batch]     # (game_id, caption_id)

    # token 化: 添加 BOS (":") 和 EOT
    bos_ids = encode(":")
    tokens = [
        (bos_ids + t[2] + [eot]) if t[2] else [14841, 14841]
        for t in batch
    ]
    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    # padding
    lengths = torch.tensor([len(t) for t in tokens])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=14841)
    mask = (tokens != 14841)

    # stack 视觉和音频特征
    vfeats_batch = torch.stack(vfeats_list)
    afeats_batch = torch.stack(afeats_list)

    return (vfeats_batch, afeats_batch, tokens), lengths, mask, captions, idx


def collate_fn_padd_dual(batch):
    """
    双模态 LSTM 版 collate 函数 (用于 validate_captioning)。

    batch: (vfeats, afeats, caption_tokens, game_id, caption_id, caption_str)
    """
    vfeats_list = [torch.tensor(t[0], dtype=torch.float32) for t in batch]
    afeats_list = [torch.tensor(t[1], dtype=torch.float32) for t in batch]
    captions = [t[-1] for t in batch]
    idx = [(t[-3], t[-2]) for t in batch]

    tokens = [
        ([SOS_TOKEN] + t[2] + [EOS_TOKEN]) if t[2] else [PAD_TOKEN, PAD_TOKEN]
        for t in batch
    ]
    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    lengths = torch.tensor([len(t) for t in tokens])
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    mask = (tokens != PAD_TOKEN)

    vfeats_batch = torch.stack(vfeats_list)
    afeats_batch = torch.stack(afeats_list)

    return (vfeats_batch, afeats_batch, tokens), lengths, mask, captions, idx


# ============================================================================
#  6. PredictionCaptions 双模态版 (DVC 推理用)
# ============================================================================

class PredictionCaptionsDual(Dataset):
    """
    双模态 DVC 推理数据集。
    从 spotting 阶段的预测结果中读取时间戳，生成对应的视觉+音频 clip。
    """

    def __init__(
        self,
        vision_root,
        audio_root,
        PredictionPath,
        features="baidu_soccer_embeddings.npy",
        split=None,
        version=2,
        framerate=2,
        window_size=15,
    ):
        if split is None:
            split = ["test"]

        self.vision_root = vision_root
        self.audio_root = audio_root
        self.PredictionPath = PredictionPath
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version
        self.split = split

        self.listGames = getListGames(split, task="caption")
        self.labels_filename, _, self.dict_event, _ = getMetaDataTask("caption", "SoccerNet", version)

        self.data = []
        self.game_feats = []
        self.game_audio = []

        l_pad = self.window_size_frame // 2 + self.window_size_frame % 2
        r_pad = self.window_size_frame // 2

        valid_game_id = 0
        valid_games = []

        for game in tqdm(self.listGames, desc="Loading prediction dual data"):
            valid_halves = _validate_game(vision_root, audio_root, game, features)
            if len(valid_halves) < 2:
                continue

            # 检查预测文件是否存在
            pred_path = os.path.join(PredictionPath, game, "results_spotting.json")
            if not os.path.isfile(pred_path):
                continue

            # 视觉
            feat_h1 = np.load(os.path.join(vision_root, game, "1_" + features))
            feat_h1 = np.pad(feat_h1.reshape(-1, feat_h1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            feat_h2 = np.load(os.path.join(vision_root, game, "2_" + features))
            feat_h2 = np.pad(feat_h2.reshape(-1, feat_h2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            self.game_feats.append((feat_h1, feat_h2))

            # 音频
            audio_h1 = np.load(os.path.join(audio_root, game, "1_audio_clap.npy"))
            audio_h1 = np.pad(audio_h1.reshape(-1, audio_h1.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            audio_h2 = np.load(os.path.join(audio_root, game, "2_audio_clap.npy"))
            audio_h2 = np.pad(audio_h2.reshape(-1, audio_h2.shape[-1]), ((l_pad, r_pad), (0, 0)), "edge")
            self.game_audio.append((audio_h1, audio_h2))

            preds = json.load(open(pred_path))
            for caption_id, annotation in enumerate(preds["predictions"]):
                if annotation["label"] not in self.dict_event:
                    continue
                time_str = annotation["gameTime"]
                half = int(time_str[0])
                if half > 2:
                    continue
                minutes, seconds = time_str.split(' ')[-1].split(':')
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)
                self.data.append(((valid_game_id, half - 1, frame), caption_id))

            valid_games.append(game)
            valid_game_id += 1

        self.listGames = valid_games
        self.path = vision_root

        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        self.text_processor = TikTokenTextProcessor(self._get_corpus())
        self.vocab_size = len(self.text_processor.vocab)

    def _get_corpus(self):
        corpus = []
        for game in getListGames(["train"], task="caption"):
            label_path = os.path.join(self.vision_root, game, self.labels_filename)
            if os.path.isfile(label_path):
                annotations = json.load(open(label_path))["annotations"]
                corpus.extend([ann["anonymized"] for ann in annotations])
        return corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id, caption_id = self.data[idx]
        vfeats = self.video_processor(clip_id, self.game_feats)
        afeats = self.video_processor(clip_id, self.game_audio)
        return vfeats, afeats, clip_id[0], caption_id

    def detokenize(self, tokens, remove_EOS=True):
        return self.text_processor.detokenize(tokens)

    def getCorpus(self, split=None):
        if split is None:
            split = ["train"]
        return self._get_corpus()
