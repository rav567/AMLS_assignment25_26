import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Code.utils.augmentation import augment_image

# Fix random seeds to ensure fully reproducible training behaviour
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Defines a compact CNN architecture for controlled medical image classification experiments
class BreastCNN(nn.Module):
    # Builds a shallow multi-block CNN to balance representational power and overfitting risk
    def __init__(self, use_dropout=False):
        super(BreastCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Dropout, set as optional
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    # Defines the forward pass to ensure consistent feature extraction and classification flow
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

# Trains the model for one epoch while optionally injecting augmentation to improve robustness
def train_epoch(model, loader, criterion, optimizer, device, use_augmentation=False):
    model.train()
    total_loss = 0
    
    for images, labels in loader:
        # Apply augmentation if enabled
        if use_augmentation:
            augmented = []
            for img in images:
                img_np = img.squeeze().numpy()
                img_aug = augment_image(img_np)
                augmented.append(torch.FloatTensor(img_aug).unsqueeze(0))
            images = torch.stack(augmented)
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Evaluates the model using clinically relevant metrics without gradient updates

def evaluate(model, loader, criterion, device):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader),
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
        'Recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        'F1': f1_score(all_labels, all_preds, average='binary', zero_division=0)
    }
    
    return metrics

# Runs the full CNN experimental pipeline to assess regularisation and augmentation effects

def run_model_b(train_data, val_data, test_data):
    """
    Run all Model B experiments.
    Tests 4 configurations: dropout (on/off) × augmentation (on/off)
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Unpack data
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    #Setup data loaders
    # Reshape: (N, 28, 28) -> (N, 1, 28, 28) for PyTorch conv layers
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train.flatten())
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_t = torch.LongTensor(y_val.flatten())
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test.flatten())
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=32, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=32, shuffle=False
    )
    
    #Define experiments
    experiments = [
        {"name": "No Dropout, No Aug", "dropout": False, "augmentation": False},
        {"name": "Dropout, No Aug", "dropout": True, "augmentation": False},
        {"name": "No Dropout, Aug", "dropout": False, "augmentation": True},
        {"name": "Dropout, Aug", "dropout": True, "augmentation": True},
    ]
    
    # Training settings
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    
    # Store results
    results = {}
    best_val_f1 = 0
    best_config = None
    best_history = None
    
    #Run each experiment
    for exp in experiments:
        # Create fresh model
        model = BreastCNN(use_dropout=exp['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # Track history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        # Train for all epochs
        for epoch in range(num_epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                device, use_augmentation=exp['augmentation']
            )
            scheduler.step()
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['F1'])
        
        # Final validation metrics
        final_val = evaluate(model, val_loader, criterion, device)
        
        # Store results
        results[exp['name']] = {
            'val_f1': final_val['F1'],
            'val_accuracy': final_val['Accuracy'],
            'val_precision': final_val['Precision'],
            'val_recall': final_val['Recall'],
            'history': history
        }
        
        #Track best model
        if final_val['F1'] > best_val_f1:
            best_val_f1 = final_val['F1']
            best_config = exp
            best_history = history
    
    #Retrain best on train+val combined
    # Combine train and val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    X_combined_t = torch.FloatTensor(X_combined).unsqueeze(1)
    y_combined_t = torch.LongTensor(y_combined.flatten())
    
    combined_loader = DataLoader(
        TensorDataset(X_combined_t, y_combined_t),
        batch_size=32, shuffle=True
    )
    
    # Train final model
    final_model = BreastCNN(use_dropout=best_config['dropout']).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        train_epoch(
            final_model, combined_loader, criterion, optimizer,
            device, use_augmentation=best_config['augmentation']
        )
        scheduler.step()
    
    #Test once
    test_metrics = evaluate(final_model, test_loader, criterion, device)    
    return test_metrics, best_config['name'], results, best_history
