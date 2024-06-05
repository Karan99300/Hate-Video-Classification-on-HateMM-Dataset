import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any
from torch.utils.data import Dataset

class VideoRecord(object):
    def __init__(self, annotation_row, root_folder):
        self._path = os.path.join(root_folder, annotation_row[0])
        self._data = annotation_row

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])

class VideoFrameDataset(Dataset):
    def __init__(self, root_folder, annotationfile_path,
                 num_segments: int = 8, frames_per_segment: int = 2,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None, test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()
        self.root_folder = root_folder
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self.parse_annotationfile()
        self.sanity_check_samples()

    def load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')

    def parse_annotationfile(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_folder) for x in open(self.annotationfile_path)]

    def sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: video {record.path} seems to have zero frames!\n")

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Padding will be applied.\n")

    def get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)
            start_indices = np.array([
                int((distance_between_indices / 2.0) + distance_between_indices * x)
                for x in range(self.num_segments)
            ])
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]'):
        frame_start_indices = frame_start_indices + record.start_frame
        images = []

        for start_index in frame_start_indices:
            frame_index = int(start_index)

            for _ in range(self.frames_per_segment):
                if frame_index <= record.end_frame:
                    image = self.load_image(record.path, frame_index)
                else:
                    # Create a white image (padding)
                    image = Image.new('RGB', (224, 224), (255, 255, 255))
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return {
            "video_frames":images,
            "label":record.label
        }

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx: int):
        record: VideoRecord = self.video_list[idx]
        frame_start_indices: 'np.ndarray[int]' = self.get_start_indices(record)
        return self._get(record, frame_start_indices)

class ImgListToTensor(torch.nn.Module):
    @staticmethod
    def forward(img_list: List[Image.Image]):
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
