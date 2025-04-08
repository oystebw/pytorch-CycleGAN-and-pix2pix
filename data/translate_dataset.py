from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from pathlib import Path
from PIL import Image
import numpy as np
import os

class TranslateDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # self.samples = []

        # for i in range(min(self.A_size, self.B_size)):
        #     # imgA is SAR, usually grayscale (convert to RGB-like format)
        #     imgA_raw = Image.open(self.A_paths[i])
        #     if imgA_raw.mode != "RGB":
        #         imgA = Image.merge("RGB", (imgA_raw, imgA_raw, imgA_raw))
        #     else:
        #         imgA = imgA_raw  # just in case it's already RGB

        #     # imgB is optical, should already be RGB
        #     imgB = Image.open(self.B_paths[i]).convert("RGB")

        #     self.samples.append({'opt': imgA, 'sar': imgB})

        btoA = self.opt.direction == 'BtoA'

        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    # def __getitem__(self, index):
    #     sample = self.samples[index]
    #     A = self.transform_A(sample['opt'])  # SAR (A)
    #     B = self.transform_B(sample['sar'])  # Optical (B)
    #     return {'A': A, 'B': B, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # Load SAR image (domain A)
        imgA_raw = Image.open(A_path)
        if imgA_raw.mode != "RGB":
            imgA = Image.merge("RGB", (imgA_raw, imgA_raw, imgA_raw))
        else:
            imgA = imgA_raw

        # Load Optical image (domain B)
        imgB = Image.open(B_path).convert("RGB")

        A = self.transform_A(imgA)
        B = self.transform_B(imgB)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        # return len(self.samples)
        return min(self.A_size, self.B_size)