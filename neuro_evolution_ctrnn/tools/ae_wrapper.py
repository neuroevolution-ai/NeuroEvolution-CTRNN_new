import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gym

from gym.spaces import Box
import logging


class AEWrapper(gym.Wrapper):

    def __init__(self, env):
        super(AEWrapper, self).__init__(env)
        logging.debug("Setting number of torch-threads to 1")
        # see https://github.com/pytorch/pytorch/issues/13757
        torch.set_num_threads(1)
        self.fe = FeatureExtractor(use_diff=True)
        self.observation_space = Box(low=0, high=10,
                                     shape=(30, 1),
                                     dtype=np.float16)
        assert env.spec.id == "QbertNoFrameskip-v4", "this wrapper only works for QbertNoFrameskip-v4"

    def step(self, action):
        ob, rew, done, info = super(AEWrapper, self).step(action)
        return self.fe.extract(ob), rew, done, info

    def reset(self, ):
        ob = super(AEWrapper, self).reset()
        return self.fe.extract(ob)


class AutoEncoderVAE(nn.Module):

    def __init__(self):
        super(AutoEncoderVAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=(2, 2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View

        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)
        self.fc_mu = nn.Linear(30, 30)
        self.fc_logsigma = nn.Linear(30, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)

        # Decoder
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1)

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x, _ = self.maxpool1(x)
            x = F.relu(self.conv2(x))
            x, _ = self.maxpool2(x)
            x = x.flatten()
            x = torch.sigmoid(self.lv1(x))
            x = torch.sigmoid(self.lv2(x))

            return x


class FeatureExtractor:
    def __init__(self, use_diff):
        self.ae = AutoEncoderVAE()
        self.ae.load_state_dict(torch.load("neuro_evolution_ctrnn/tools/ae.pt", map_location=torch.device("cpu")))
        # self.ae.load_state_dict(torch.load("neuro_evolution_ctrnn/tools/ae.pt"))

        # Sets the Model into Evaluation Mode
        self.ae.eval()

        self.last = np.zeros(30)
        self.use_diff = use_diff

    def extract(self, obs):
        frame = self.frame2tensor(obs)
        features = self.ae(frame)
        features = features.numpy().flatten()

        if self.use_diff:
            diff = features - self.last
            self.last = features
            diff *= 10e1
            return diff

        return features

    def reencode(self, obs):
        # TODO Don't use does not work atm -> pred is (1, 30) dimensional so permute does not work
        frame = self.frame2tensor(obs)
        pred = self.ae(frame)
        pred = pred.permute(0, 2, 3, 1)

        with torch.no_grad():
            yay = torch.tensor([255], dtype=torch.int)
            imgpred = pred * yay
            return imgpred.numpy().flatten().astype(np.uint8)

    @staticmethod
    def normalize(v):
        return v / 255

    @staticmethod
    def frame2tensor(frame):
        frame = FeatureExtractor.normalize(np.array([frame]))
        return torch.from_numpy(frame).type(torch.float32).permute(0, 3, 1, 2)


def feature2img(feature):
    feat_raw = feature.reshape((5, 6))
    feat = (feat_raw * 255).astype(np.uint8)
    feat_data = cv2.cvtColor(feat, cv2.COLOR_GRAY2BGR)
    return cv2.resize(feat_data, (160, 210), interpolation=cv2.INTER_NEAREST)


if __name__ == "__main__":
    fe = FeatureExtractor(True)
    env = gym.make("QbertNoFrameskip-v4")
    video_obs = cv2.VideoWriter("obs.avi", cv2.VideoWriter_fourcc(*"PIM1"), 25, (160, 210), True)
    video_pred = cv2.VideoWriter("pred.avi", cv2.VideoWriter_fourcc(*"PIM1"), 25, (160, 210), True)
    video_feat = cv2.VideoWriter("feat.avi", cv2.VideoWriter_fourcc(*"PIM1"), 25, (160, 210), True)
    for _ in range(1):
        ob = env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
            video_obs.write(observation.astype(np.uint8))
            video_pred.write(fe.reencode(observation))
            video_feat.write(feature2img(fe.extract(observation)))
            print("Step " + str(step))
    env.close()
    cv2.destroyAllWindows()
    video_obs.release()
    video_pred.release()
    video_feat.release()
