from .operators import *
import torch, json, pandas


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class DPOVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        video_operator=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.video_operator = video_operator
        self.max_data_items = max_data_items
        self.load_from_cache = False
        self.data = []
        if self.video_operator is None:
            raise ValueError("DPOVideoDataset requires `video_operator`.")
        if metadata_path is None:
            raise ValueError("DPOVideoDataset requires `metadata_path`.")
        self._load_metadata(metadata_path)
        if len(self.data) == 0:
            raise ValueError(f"DPOVideoDataset metadata is empty: {metadata_path}")

    def _load_metadata(self, metadata_path):
        if metadata_path is None:
            return
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                self.data = json.load(f)
        elif metadata_path.endswith(".jsonl"):
            with open(metadata_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if len(self.data) == 0:
            raise RuntimeError("DPOVideoDataset has no data.")
        item_id = data_id % len(self.data)
        item = self.data[item_id].copy()

        # 检查数据
        required_keys = ("prompt", "video_chosen", "video_rejected")
        missing_keys = [key for key in required_keys if key not in item]
        if len(missing_keys) > 0:
            raise KeyError(f"DPO sample missing keys {missing_keys}. data_id={item_id}")

        prompt = item["prompt"]
        chosen_path = item["video_chosen"]
        rejected_path = item["video_rejected"]
        # 处理视频
        video_chosen = self.video_operator(chosen_path)
        video_rejected = self.video_operator(rejected_path)
        
        # 检查数据是否有问题
        if not isinstance(video_chosen, list) or not isinstance(video_rejected, list):
            raise TypeError(
                f"DPO video_operator must return list frames. "
                f"chosen_type={type(video_chosen)}, rejected_type={type(video_rejected)}, "
                f"data_id={item_id}, chosen={chosen_path}, rejected={rejected_path}"
            )
        if len(video_chosen) == 0 or len(video_rejected) == 0:
            raise ValueError(
                f"DPO sample has empty frames. "
                f"chosen_len={len(video_chosen)}, rejected_len={len(video_rejected)}, "
                f"data_id={item_id}, chosen={chosen_path}, rejected={rejected_path}"
            )
        if len(video_chosen) != len(video_rejected):
            raise ValueError(
                f"DPO pair frame count mismatch. "
                f"chosen_len={len(video_chosen)}, rejected_len={len(video_rejected)}, "
                f"data_id={item_id}, chosen={chosen_path}, rejected={rejected_path}"
            )
        for frame_id, (frame_chosen, frame_rejected) in enumerate(zip(video_chosen, video_rejected)):
            if not hasattr(frame_chosen, "size") or not hasattr(frame_rejected, "size"):
                raise TypeError(
                    f"DPO frame type must provide `.size`. "
                    f"frame_id={frame_id}, chosen_type={type(frame_chosen)}, rejected_type={type(frame_rejected)}, "
                    f"data_id={item_id}, chosen={chosen_path}, rejected={rejected_path}"
                )
            if frame_chosen.size != frame_rejected.size:
                raise ValueError(
                    f"DPO pair frame size mismatch. "
                    f"frame_id={frame_id}, chosen_size={frame_chosen.size}, rejected_size={frame_rejected.size}, "
                    f"data_id={item_id}, chosen={chosen_path}, rejected={rejected_path}"
                )

        return {
            "prompt": prompt,
            "video_chosen": video_chosen,
            "video_rejected": video_rejected,
        }

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        return len(self.data) * self.repeat


class MaskDPOVideoDataset(torch.utils.data.Dataset):
    """
    MaskDPO数据集，每个样本包含:
    - prompt: mask DPO使用的文本提示
    - video_chosen, video_rejected: mask DPO偏好对
    - mask: 二值掩码
    - prompt_sft: SFT使用的文本提示
    - video_sft: 用于SFT loss
    - prompt_vdpo: 全局DPO使用的文本提示
    - video_vdpo_chosen, video_vdpo_rejected: 用于普通DPO loss
    - strength: 
    """
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        video_operator=None,
        mask_operator=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.video_operator = video_operator
        self.mask_operator = mask_operator
        self.max_data_items = max_data_items
        self.load_from_cache = False
        self.data = []
        if self.video_operator is None:
            raise ValueError("MaskDPOVideoDataset requires `video_operator`.")
        if metadata_path is None:
            raise ValueError("MaskDPOVideoDataset requires `metadata_path`.")
        self._load_metadata(metadata_path)
        if len(self.data) == 0:
            raise ValueError(f"MaskDPOVideoDataset metadata is empty: {metadata_path}")

    def _load_metadata(self, metadata_path):
        if metadata_path is None:
            return
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                self.data = json.load(f)
        elif metadata_path.endswith(".jsonl"):
            with open(metadata_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def _validate_video_pair(self, video_a, video_b, name_a, name_b, item_id):
        for videos, name in [(video_a, name_a), (video_b, name_b)]:
            if not isinstance(videos, list):
                raise TypeError(f"MaskDPO {name} must be list of frames, got {type(videos)}. data_id={item_id}")
            if len(videos) == 0:
                raise ValueError(f"MaskDPO {name} has empty frames. data_id={item_id}")
        if len(video_a) != len(video_b):
            raise ValueError(
                f"MaskDPO {name_a} and {name_b} frame count mismatch: "
                f"{len(video_a)} vs {len(video_b)}. data_id={item_id}"
            )
        for frame_id, (fa, fb) in enumerate(zip(video_a, video_b)):
            if not hasattr(fa, "size") or not hasattr(fb, "size"):
                raise TypeError(f"MaskDPO frame must provide `.size`. frame_id={frame_id}, data_id={item_id}")
            if fa.size != fb.size:
                raise ValueError(
                    f"MaskDPO {name_a}/{name_b} frame size mismatch at frame {frame_id}: "
                    f"{fa.size} vs {fb.size}. data_id={item_id}"
                )

    def __getitem__(self, data_id):
        if len(self.data) == 0:
            raise RuntimeError("MaskDPOVideoDataset has no data.")
        item_id = data_id % len(self.data)
        item = self.data[item_id].copy()

        required_keys = ("prompt", "video_chosen", "video_rejected", "mask", "strength",
                         "prompt_sft", "video_sft",
                         "prompt_vdpo", "video_vdpo_chosen", "video_vdpo_rejected")
        missing_keys = [key for key in required_keys if key not in item]
        if len(missing_keys) > 0:
            raise KeyError(f"MaskDPO sample missing keys {missing_keys}. data_id={item_id}")

        prompt = item["prompt"]
        prompt_sft = item["prompt_sft"]
        prompt_vdpo = item["prompt_vdpo"]
        strength = item["strength"]

        video_chosen = self.video_operator(item["video_chosen"])
        video_rejected = self.video_operator(item["video_rejected"])
        
        if self.mask_operator is not None:
            mask = self.mask_operator(item["mask"])
        else:
            mask = self.video_operator(item["mask"])
        video_sft = self.video_operator(item["video_sft"])
        video_vdpo_chosen = self.video_operator(item["video_vdpo_chosen"])
        video_vdpo_rejected = self.video_operator(item["video_vdpo_rejected"])

        # 校验mask dpo偏好对
        self._validate_video_pair(video_chosen, video_rejected, "video_chosen", "video_rejected", item_id)
        
        # 校验sft视频
        if not isinstance(video_sft, list) or len(video_sft) == 0:
            raise ValueError(f"MaskDPO video_sft must be non-empty list. data_id={item_id}")
        # 校验vdpo偏好对
        self._validate_video_pair(video_vdpo_chosen, video_vdpo_rejected, "video_vdpo_chosen", "video_vdpo_rejected", item_id)

        return {
            "prompt": prompt,
            "prompt_sft": prompt_sft,
            "prompt_vdpo": prompt_vdpo,
            "strength": strength,
            "video_chosen": video_chosen,
            "video_rejected": video_rejected,
            "mask": mask,
            "video_sft": video_sft,
            "video_vdpo_chosen": video_vdpo_chosen,
            "video_vdpo_rejected": video_vdpo_rejected,
        }

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        return len(self.data) * self.repeat
