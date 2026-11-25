from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from torchvision.io import read_video
from Resnet101 import ResNet101



class VisExtNet(nn.Module):

    def __init__(self, resnet_weight_path, feature_dim, device):
        super().__init__()

        self.mtcnn = MTCNN(image_size = 224, keep_all = True, device = device)
        self.resnet = ResNet101().to(device)
        self.resnet.load_state_dict(torch.load(resnet_weight_path))
        self.feature_dim = feature_dim
        self.device = device


    def forward(self, video_path):
        video_frames, _, _ = read_video(filename = video_path)
        video_faces = self.mtcnn(video_frames)
        video_faces_delete_none_scene = [video_face for video_face in video_faces if video_face is not None]
        num_frames = len(video_faces_delete_none_scene)
        if num_frames == 0:
            return torch.zeros(self.feature_dim).to(self.device)
        elif num_frames <= 20:
            all_video_faces = torch.cat(video_faces_delete_none_scene, dim = 0).to(self.device)
        else:
            steps = int(num_frames / 20)
            all_video_faces = torch.cat(video_faces_delete_none_scene[::steps][:20], dim = 0).to(self.device)
        visual_embedding = torch.mean(self.resnet(all_video_faces), dim = 0)

        return visual_embedding
#
#
#
#

# from timm import create_model  # 使用 timm 库加载 Swin Transformer
#
#
# class VisExtNet(nn.Module):
#
#     def __init__(self, swin_weight_path, feature_dim, device):
#         super().__init__()
#
#         self.mtcnn = MTCNN(image_size=224, keep_all=True, device=device)
#         self.swin_transformer = create_model("swin_base_patch4_window7_224", pretrained=False).to(device)
#         self.swin_transformer.load_state_dict(torch.load(swin_weight_path))
#         self.feature_dim = feature_dim
#         self.device = device
#
#     def forward(self, video_path):
#         video_frames, _, _ = read_video(filename=video_path)
#         video_faces = self.mtcnn(video_frames)
#         video_faces_delete_none_scene = [video_face for video_face in video_faces if video_face is not None]
#         num_frames = len(video_faces_delete_none_scene)
#         if num_frames == 0:
#             return torch.zeros(self.feature_dim).to(self.device)
#         elif num_frames <= 20:
#             all_video_faces = torch.cat(video_faces_delete_none_scene, dim=0).to(self.device)
#         else:
#             steps = int(num_frames / 20)
#             all_video_faces = torch.cat(video_faces_delete_none_scene[::steps][:20], dim=0).to(self.device)
#         visual_embedding = torch.mean(self.swin_transformer(all_video_faces), dim=0)
#
#         return visual_embedding

