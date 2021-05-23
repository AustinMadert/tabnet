from typing import List
import torch
import torch.nn as nn

from transformers import FeatureBlock


class TabNetDecoderStep(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.feat = self.build_feature_transformer(shared_feat)
        self.fc = nn.Linear(input_size, output_size)


    def build_feature_transformer(self, shared_feat: FeatureBlock) -> nn.Module:
        return nn.Sequential(
            shared_feat,
            FeatureBlock(self.input_size)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.feat(X)
        features = self.fc(X)
        return features


class TabNetDecoder(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_size: int, output_size: int, 
                batch_size: int, n_steps: int)  -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.input_size = input_size
        self.output_size = output_size
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.output = torch.zeros(batch_size, input_size)
        self.steps = self.build_decoder_steps()


    def build_decoder_steps(self) -> List[nn.Module]:
        return [TabNetDecoderStep(self.shared_feat, self.input_size, self.output_size)
                for _ in range(self.n_steps)]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            X = step(X)
            self.output += X
        return self.output
