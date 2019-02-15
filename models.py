import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from dataset import YoutubeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect
from gpu_mem_track import MemTracker
from torch.utils.checkpoint import checkpoint
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self):
        super(ConvLSTMCell, self).__init__()
        self.input_size = (8, 14)
        self.hidden_size = (8, 14)
        self.Gates = nn.Conv2d(1024, 2048, 3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.relu(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.relu(cell)

        return hidden, cell

class VOS(nn.Module):
    def __init__(self, seq):
        super(VOS, self).__init__()
        self.initializer = models.vgg16(pretrained=True).features
        self.initializer[0] = nn.Conv2d(4, 64, 3, 1, 1)
        self.init_a = nn.Conv2d(512, 512, 1)
        self.init_b = nn.Conv2d(512, 512, 1)
        nn.init.xavier_uniform_(self.init_a.weight)
        nn.init.xavier_uniform_(self.init_b.weight)
        self.state = [ConvLSTMCell().half().cuda() for i in range(seq)]
        self.seq = seq
        # Encoder 

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x, mask, t):
        frame0 = x[:, :3, :, :]
        mask0 = mask[:, :1, :, :]
        t.track()
        output = []
        init_input = torch.cat((frame0, mask0), 1)
        tmp = self.initializer(init_input)
        c = self.init_a(tmp)
        h = self.init_b(tmp)
        t.track()
        x = x.view(-1, 3, 256, 448)
        f = x[3:, :, :, :]
        f = self.enc1(f)
        f, id1 = F.max_pool2d(f, kernel_size=2, stride=2, return_indices=True)

        f = self.enc2(f)
        f, id2 = F.max_pool2d(f, kernel_size=2, stride=2, return_indices=True)

        f = self.enc3(f)
        f, id3 = F.max_pool2d(f, kernel_size=2, stride=2, return_indices=True)

        f = self.enc4(f)
        f, id4 = F.max_pool2d(f, kernel_size=2, stride=2, return_indices=True)
        size = f.size()
        f = self.enc5(f)
        f, id5 = F.max_pool2d(f, kernel_size=2, stride=2, return_indices=True)
        f = f.view(-1, self.seq - 1, 512, 8, 14)
        for i in range(self.seq):
            c, h = self.state[i](f[:, i, :, :, :], (c, h))
            output.append(h)
        output = torch.cat(output, 0)
        output = output.view(-1, self.seq - 1, 512, 8, 14)

        y = F.max_unpool2d(output, id5, 2, 2, output_size = size)
        y = self.dec1(y)

        y = F.max_unpool2d(y, id4, 2, 2)
        y = self.dec2(y)

        y = F.max_unpool2d(y, id3, 2, 2)
        y = self.dec3(y)

        y = F.max_unpool2d(y, id2, 2, 2)
        y = self.dec4(y)

        y = F.max_unpool2d(y, id1, 2, 2)
        del id1, id2, id3, id4, id5
        y = self.dec5(y)

        y = self.out(y)
        t.track()
        print(y.shape)
        raise KeyboardInterrupt
        return output

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset = YoutubeDataset()
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    model = VOS(5)

    frame = inspect.currentframe()
    gpu_tracker = MemTracker(frame)

    gpu_tracker.track()
    model.cuda().float()
    gpu_tracker.track()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    j = 0
    logger = tqdm(train_loader)
    for (a,b) in logger:
        a = a.float().cuda()
        gpu_tracker.track()
        b = b.float().cuda()
        gpu_tracker.track()
        output = checkpoint_sequential(model, 5, a, b)
        output.sum().backward()
        gpu_tracker.track()
        target = b[:, 1:, :, :]
        target = target.long().cuda()
        target = target.view(-1, 4, 256*448)
        train_loss = 0
        gpu_tracker.track()
        i = 0
        loss = criterion(output[:, i, :, :], target[:, i, :])
        for i in range(1, 4):
            loss += criterion(output[:, i, :, :], target[:, i, :])
            gpu_tracker.track()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gpu_tracker.track()
        print(loss.data)
        logger.set_description("{}:Loss:{}".format(j, loss.data[0]))
        j += 1


