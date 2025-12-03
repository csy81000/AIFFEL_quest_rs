import os
import json
import math
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 설정 및 경로 (환경에 맞게 수정하세요)
# ==========================================
PROJECT_PATH = './posedata'
IMAGE_PATH = os.path.join(PROJECT_PATH, 'images')
MODEL_PATH = os.path.join(PROJECT_PATH, 'models')
TRAIN_JSON = os.path.join(PROJECT_PATH, 'mpii_human_pose_v1_u12_2', 'train.json')
VALID_JSON = os.path.join(PROJECT_PATH, 'mpii_human_pose_v1_u12_2', 'validation.json')

IMAGE_SHAPE = (256, 256, 3)
HEATMAP_SIZE = (64, 64)

# 결과 그래프 저장 경로
GRAPH_PATH = os.path.join(PROJECT_PATH, 'graphs')
if not os.path.exists(GRAPH_PATH):
    os.makedirs(GRAPH_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# ==========================================
# 2. 데이터 전처리 및 데이터셋 클래스 (공통)
# ==========================================
def parse_one_annotation(anno, image_dir):
    filename = anno['image']
    joints = anno['joints']
    joints_visibility = anno['joints_vis']
    annotation = {
        'filename': filename,
        'filepath': os.path.join(image_dir, filename),
        'joints_visibility': joints_visibility,
        'joints': joints,
        'center': anno['center'],
        'scale' : anno['scale']
    }
    return annotation

class Preprocessor(object):
    def __init__(self, image_shape=(256, 256, 3), heatmap_shape=(64, 64, 16), is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)
        image = Image.open(io.BytesIO(features['image/encoded']))

        if self.is_train:
            random_margin = float(torch.empty(1).uniform_(0.1, 0.3).item())
            image, keypoint_x, keypoint_y = self.crop_roi(image, features, margin=random_margin)
            image = image.resize((self.image_shape[1], self.image_shape[0]))
        else:
            image, keypoint_x, keypoint_y = self.crop_roi(image, features)
            image = image.resize((self.image_shape[1], self.image_shape[0]))

        image_np = np.array(image).astype(np.float32)
        image_np = image_np / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        heatmaps = self.make_heatmaps(features, keypoint_x, keypoint_y, self.heatmap_shape)
        return image_tensor, heatmaps

    def parse_tfexample(self, example):
        annotation = example['annotation']
        joints = annotation['joints']
        keypoint_x = [joint[0] for joint in joints]
        keypoint_y = [joint[1] for joint in joints]
        joints_vis = annotation.get('joints_vis', [1] * len(joints))

        features = {
            'image/encoded': self.image_to_bytes(example['image']),
            'image/object/parts/x': keypoint_x,
            'image/object/parts/y': keypoint_y,
            'image/object/parts/v': joints_vis,
            'image/object/center/x': annotation['center'][0],
            'image/object/center/y': annotation['center'][1],
            'image/object/scale': annotation['scale'],
        }
        return features

    def image_to_bytes(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def crop_roi(self, image, features, margin=0.2):
        img_width, img_height = image.size
        keypoint_x = torch.tensor(features['image/object/parts/x'], dtype=torch.int32)
        keypoint_y = torch.tensor(features['image/object/parts/y'], dtype=torch.int32)
        body_height = features['image/object/scale'] * 200.0
        
        masked_keypoint_x = keypoint_x[keypoint_x > 0]
        masked_keypoint_y = keypoint_y[keypoint_y > 0]
        
        if len(masked_keypoint_x) == 0:
             keypoint_xmin = 0
             keypoint_xmax = img_width
             keypoint_ymin = 0
             keypoint_ymax = img_height
        else:
            keypoint_xmin = int(masked_keypoint_x.min().item())
            keypoint_xmax = int(masked_keypoint_x.max().item())
            keypoint_ymin = int(masked_keypoint_y.min().item())
            keypoint_ymax = int(masked_keypoint_y.max().item())

        extra = int(body_height * margin)
        xmin = keypoint_xmin - extra
        xmax = keypoint_xmax + extra
        ymin = keypoint_ymin - extra
        ymax = keypoint_ymax + extra

        effective_xmin = max(xmin, 0)
        effective_ymin = max(ymin, 0)
        effective_xmax = min(xmax, img_width)
        effective_ymax = min(ymax, img_height)

        cropped_image = image.crop((effective_xmin, effective_ymin, effective_xmax, effective_ymax))
        new_width = effective_xmax - effective_xmin
        new_height = effective_ymax - effective_ymin

        effective_keypoint_x = (keypoint_x.float() - effective_xmin) / new_width
        effective_keypoint_y = (keypoint_y.float() - effective_ymin) / new_height
        return cropped_image, effective_keypoint_x, effective_keypoint_y

    def generate_2d_gaussian(self, height, width, y0, x0, visibility=2, sigma=1, scale=12):
        heatmap = torch.zeros((height, width), dtype=torch.float32)
        xmin = x0 - 3 * sigma
        ymin = y0 - 3 * sigma
        xmax = x0 + 3 * sigma
        ymax = y0 + 3 * sigma

        if xmin >= width or ymin >= height or xmax < 0 or ymax < 0 or visibility == 0:
            return heatmap

        size = int(6 * sigma + 1)
        grid_range = torch.arange(0, size, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(grid_range, grid_range, indexing='xy')
        center_x = size // 2
        center_y = size // 2

        gaussian_patch = torch.exp(-(((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))) * scale

        patch_xmin = max(0, -xmin)
        patch_ymin = max(0, -ymin)
        patch_xmax = min(xmax, width) - xmin
        patch_ymax = min(ymax, height) - ymin

        heatmap_xmin = max(0, xmin)
        heatmap_ymin = max(0, ymin)
        heatmap_xmax = min(xmax, width)
        heatmap_ymax = min(ymax, height)

        heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = \
            gaussian_patch[int(patch_ymin):int(patch_ymax), int(patch_xmin):int(patch_xmax)]
        return heatmap

    def make_heatmaps(self, features, keypoint_x, keypoint_y, heatmap_shape):
        v = torch.tensor(features['image/object/parts/v'], dtype=torch.float32)
        x = torch.round(keypoint_x * heatmap_shape[1]).to(torch.int32)
        y = torch.round(keypoint_y * heatmap_shape[0]).to(torch.int32)
        num_heatmap = heatmap_shape[2]
        heatmaps_list = []
        for i in range(num_heatmap):
            gaussian = self.generate_2d_gaussian(
                height=heatmap_shape[0],
                width=heatmap_shape[1],
                y0=int(y[i].item()),
                x0=int(x[i].item()),
                visibility=int(v[i].item())
            )
            heatmaps_list.append(gaussian)
        heatmaps = torch.stack(heatmaps_list, dim=0)
        return heatmaps

class MPIIDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        self.annotations = [parse_one_annotation(anno, image_dir) for anno in annotations]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image = Image.open(anno['filepath']).convert('RGB')
        if self.transform:
            image, heatmaps = self.transform({'image': image, 'annotation': anno})
            return image, heatmaps
        else:
            return image, anno

def create_dataloader(annotation_file, image_dir, batch_size, num_heatmap, is_train=True, num_workers=0):
    preprocess = Preprocessor(
        image_shape=IMAGE_SHAPE,
        heatmap_shape=(HEATMAP_SIZE[0], HEATMAP_SIZE[1], num_heatmap),
        is_train=is_train
    )
    dataset = MPIIDataset(annotation_file=annotation_file, image_dir=image_dir, transform=preprocess)
    
    prefetch_factor = 2 if num_workers > 0 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch_factor
    )
    return dataloader

# ==========================================
# 3-1. 모델 정의: Stacked Hourglass
# ==========================================
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, filters // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample_conv(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += identity
        return out

class HourglassModule(nn.Module):
    def __init__(self, order, filters, num_residual):
        super(HourglassModule, self).__init__()
        self.order = order
        self.up1_0 = BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
        self.up1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])
        if order > 1:
            self.low2 = HourglassModule(order - 1, filters, num_residual)
        else:
            self.low2_blocks = nn.Sequential(*[
                BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
                for _ in range(num_residual)
            ])
        self.low3_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1_0(x)
        up1 = self.up1_blocks(up1)
        low1 = self.pool(x)
        low1 = self.low1_blocks(low1)
        if self.order > 1:
            low2 = self.low2(low1)
        else:
            low2 = self.low2_blocks(low1)
        low3 = self.low3_blocks(low2)
        up2 = self.upsample(low3)
        return up2 + up1

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StackedHourglassNetwork(nn.Module):
    def __init__(self, input_shape=(256, 256, 3), num_stack=4, num_residual=1, num_heatmap=16):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stack = num_stack
        in_channels = input_shape[2]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck1 = BottleneckBlock(in_channels=64, filters=128, stride=1, downsample=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck2 = BottleneckBlock(in_channels=128, filters=128, stride=1, downsample=False)
        self.bottleneck3 = BottleneckBlock(in_channels=128, filters=256, stride=1, downsample=True)
        
        self.hourglass_modules = nn.ModuleList()
        self.residual_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.heatmap_convs = nn.ModuleList()
        self.intermediate_convs = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        for i in range(num_stack):
            self.hourglass_modules.append(HourglassModule(order=4, filters=256, num_residual=num_residual))
            self.residual_modules.append(nn.Sequential(*[
                BottleneckBlock(in_channels=256, filters=256, stride=1, downsample=False)
                for _ in range(num_residual)
            ]))
            self.linear_layers.append(LinearLayer(in_channels=256, out_channels=256))
            self.heatmap_convs.append(nn.Conv2d(256, num_heatmap, kernel_size=1, stride=1, padding=0))
            if i < num_stack - 1:
                self.intermediate_convs.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.intermediate_outs.append(nn.Conv2d(num_heatmap, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bottleneck1(x)
        x = self.pool(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        outputs = []
        for i in range(self.num_stack):
            hg = self.hourglass_modules[i](x)
            res = self.residual_modules[i](hg)
            lin = self.linear_layers[i](res)
            heatmap = self.heatmap_convs[i](lin)
            outputs.append(heatmap)
            if i < self.num_stack - 1:
                inter1 = self.intermediate_convs[i](lin)
                inter2 = self.intermediate_outs[i](heatmap)
                x = inter1 + inter2
        return outputs

# ==========================================
# 3-2. 모델 정의: SimpleBaseline (ResNet50)
# ==========================================
class SimpleBaseline(nn.Module):
    def __init__(self, num_heatmap=16):
        super(SimpleBaseline, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.in_planes = 2048 
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]
        )
        self.final_layer = nn.Conv2d(
            in_channels=256, out_channels=num_heatmap, kernel_size=1, stride=1, padding=0
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_planes,
                    out_channels=num_filters[i],
                    kernel_size=num_kernels[i],
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            self.in_planes = num_filters[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        # Trainer와의 호환성을 위해 리스트로 감싸서 반환
        return [x]

# ==========================================
# 4. 학습 엔진 (Trainer) - 수정됨
# ==========================================
class Trainer(object):
    def __init__(self, model, epochs, global_batch_size, initial_learning_rate, run_name):
        self.model = model
        self.epochs = epochs
        self.global_batch_size = global_batch_size
        self.loss_object = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_learning_rate)
        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 10
        self.best_model = None
        self.run_name = run_name # 모델 구분용 이름 (예: "Hourglass", "SimpleBaseline")
        
        self.train_loss_history = []
        self.val_loss_history = []

    def lr_decay(self):
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_learning_rate

    def compute_loss(self, labels, outputs):
        loss = 0
        for output in outputs:
            weights = (labels > 0).float() * 81 + 1
            squared_error = (labels - output) ** 2
            weighted_error = squared_error * weights
            loss += weighted_error.mean() / self.global_batch_size
        return loss

    def train_step(self, images, labels, device):
        self.model.train()
        images = images.to(device)
        labels = labels.to(device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.compute_loss(labels, outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, images, labels, device):
        self.model.eval()
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = self.model(images)
            loss = self.compute_loss(labels, outputs)
        return loss.item()

    def save_model(self, epoch, loss):
        filename = f'{self.run_name}_model-epoch-{epoch}-loss-{loss:.4f}.pt'
        model_name = os.path.join(MODEL_PATH, filename)
        torch.save(self.model.state_dict(), model_name)
        self.best_model = model_name
        print(f"\n[{self.run_name}] Model saved: {model_name}")

    def plot_losses(self):
        """개별 모델의 학습 그래프 저장"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Train Loss', marker='o')
        plt.plot(self.val_loss_history, label='Validation Loss', marker='s')
        plt.title(f'{self.run_name} Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(GRAPH_PATH, f'{self.run_name}_loss_graph.png')
        plt.savefig(save_path)
        print(f"[{self.run_name}] Loss graph saved to {save_path}")
        plt.close() # 메모리 해제

    def run(self, train_loader, val_loader, device):
        print(f"\n>>> Starting Training for: {self.run_name}")
        for epoch in range(1, self.epochs + 1):
            self.lr_decay()
            print(f"\n[Epoch {epoch}/{self.epochs}] Learning Rate: {self.current_learning_rate:.6f}")
            
            # --- Training Loop ---
            total_train_loss = 0.0
            num_train_batches = 0
            train_pbar = tqdm(train_loader, desc=f"Training {self.run_name}", unit="batch")
            
            for images, labels in train_pbar:
                batch_loss = self.train_step(images, labels, device)
                total_train_loss += batch_loss
                num_train_batches += 1
                train_pbar.set_postfix({'Batch Loss': f'{batch_loss:.4f}', 'Avg Loss': f'{total_train_loss/num_train_batches:.4f}'})
            
            train_loss = total_train_loss / num_train_batches
            self.train_loss_history.append(train_loss)
            print(f"  - Train Loss: {train_loss:.4f}")

            # --- Validation Loop ---
            total_val_loss = 0.0
            num_val_batches = 0
            val_pbar = tqdm(val_loader, desc=f"Validating {self.run_name}", unit="batch")
            
            for images, labels in val_pbar:
                batch_loss = self.val_step(images, labels, device)
                num_val_batches += 1
                if not math.isnan(batch_loss):
                    total_val_loss += batch_loss
                else:
                    num_val_batches -= 1
                val_pbar.set_postfix({'Val Loss': f'{batch_loss:.4f}'})
            
            if num_val_batches > 0:
                val_loss = total_val_loss / num_val_batches
            else:
                val_loss = float('nan')
            
            self.val_loss_history.append(val_loss)
            print(f"  - Val Loss: {val_loss:.4f}")

            if val_loss < self.lowest_val_loss:
                self.save_model(epoch, val_loss)
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss
        
        self.plot_losses()
        return self.best_model, self.train_loss_history, self.val_loss_history

# ==========================================
# 5. 통합 실행 및 그래프 비교
# ==========================================
def train_model(model_class, model_kwargs, run_name, epochs, lr, batch_size, num_heatmap, num_workers):
    """
    단일 모델 학습을 위한 래퍼 함수
    """
    # Dataloader 생성
    train_loader = create_dataloader(TRAIN_JSON, IMAGE_PATH, batch_size, num_heatmap, is_train=True, num_workers=num_workers)
    val_loader = create_dataloader(VALID_JSON, IMAGE_PATH, batch_size, num_heatmap, is_train=False, num_workers=num_workers)

    # 모델 생성
    print(f"\nBuilding Model: {run_name}...")
    model = model_class(**model_kwargs)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # 학습 시작
    trainer = Trainer(model, epochs, batch_size, initial_learning_rate=lr, run_name=run_name)
    best_model_path, train_hist, val_hist = trainer.run(train_loader, val_loader, device)
    
    # 메모리 정리
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return train_hist, val_hist

def plot_comparison_graphs(results):
    """
    여러 모델의 결과를 비교하는 그래프 생성 (총 3번째 그래프)
    results: {'Name': (train_hist, val_hist), ...}
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['b', 'r', 'g', 'c'] # 모델별 색상 지정
    
    for i, (name, (train_h, val_h)) in enumerate(results.items()):
        c = colors[i % len(colors)]
        plt.plot(train_h, label=f'{name} Train', linestyle='--', color=c, alpha=0.7)
        plt.plot(val_h, label=f'{name} Val', linestyle='-', color=c)

    plt.title('Model Comparison: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(GRAPH_PATH, 'combined_comparison_graph.png')
    plt.savefig(save_path)
    print(f"\nCombined comparison graph saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # 공통 설정
    epochs = 20
    batch_size = 2
    num_heatmap = 16
    num_workers = 4
    
    results = {}

    # 1. Train Stacked Hourglass
    print("\n" + "="*50)
    print(" STEP 1: Training Stacked Hourglass Network")
    print("="*50)
    hg_train, hg_val = train_model(
        model_class=StackedHourglassNetwork,
        model_kwargs={'input_shape': IMAGE_SHAPE, 'num_stack': 4, 'num_residual': 1, 'num_heatmap': num_heatmap},
        run_name="StackedHourglass",
        epochs=epochs,
        lr=0.007, # Hourglass LR
        batch_size=batch_size,
        num_heatmap=num_heatmap,
        num_workers=num_workers
    )
    results['StackedHourglass'] = (hg_train, hg_val)

    # 2. Train SimpleBaseline (ResNet50)
    print("\n" + "="*50)
    print(" STEP 2: Training SimpleBaseline (ResNet50)")
    print("="*50)
    res_train, res_val = train_model(
        model_class=SimpleBaseline,
        model_kwargs={'num_heatmap': num_heatmap},
        run_name="SimpleBaseline",
        epochs=epochs,
        lr=0.007, # SimpleBaseline LR (조금 더 높게 잡음)
        batch_size=batch_size,
        num_heatmap=num_heatmap,
        num_workers=num_workers
    )
    results['SimpleBaseline'] = (res_train, res_val)

    # 3. 종합 비교 그래프 그리기
    print("\n" + "="*50)
    print(" STEP 3: Generating Comparison Graph")
    print("="*50)
    plot_comparison_graphs(results)
    
    print("\nAll tasks finished.")