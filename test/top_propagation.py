#Comparison to Naitzat results
import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import persim

def optimize_nn(model, dat_loader, optim, crit, num_epochs = 100):
    loss_hist = []
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dat_loader:
            # Forward pass
            outputs = model(inputs)[-1]
            loss = crit(outputs, targets)
    
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_hist.append(loss.item())
        if (epoch+1) % 10 == 0 and epoch<300: #No need to compute performance at every step early on
            class_predictions = torch.argmax(model(dat_loader.dataset.tensors[0])[-1],1)==torch.argmax(dat_loader.dataset.tensors[1],1)
            class_performance = class_predictions.sum()/len(dat_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Classification performance: {class_performance:.4f}')
            if class_performance>0.999:
                return outputs, loss_hist
        elif epoch>300:
            class_predictions = torch.argmax(model(dat_loader.dataset.tensors[0])[-1],1)==torch.argmax(dat_loader.dataset.tensors[1],1)
            class_performance = class_predictions.sum()/len(dat_loader)
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Classification performance: {class_performance:.4f}')
            if class_performance>0.999:
                return outputs, loss_hist
    return outputs, loss_hist


#%% Network parameters from Naitzat

Hidden_layers = {"D-I": [15]*9,
                 "D-II": [15]*9,
                 "D-II-2": [25]*9,
                 "D-III": [15]*9,
                 "D-III-2": [50]*9}

n_models = 30

#%%Optimization on disk dataset
X_train = torch.Tensor(np.load("../data//D-I-train.npy"))
y_train = torch.Tensor(np.load("../data/D-I-train-labels.npy"))

X_test = torch.Tensor(np.load("../data//D-I-test.npy"))
y_test = torch.Tensor(np.load("../data/D-I-test-labels.npy"))

data_train = TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss()
trained_models = []
for n in range(n_models):
    model = FeedforwardNetwork(2, Hidden_layers['D-I'], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    optimizer = optim.Adam(model.parameters(), lr=0.00002)
    outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=5000)
    trained_models.append(model)    

#%%save trained models
# for i, m in enumerate(trained_models):
#     torch.save(m.state_dict(), f"../models/D-I_model_{i}.pth")


#%%Optimiize on ring dataset
X_train = torch.Tensor(np.load("../data//D-II-train.npy"))
labels = torch.Tensor(np.load("../data/D-II-train-labels.npy"))
y_train = torch.zeros([len(labels),2])
y_train[labels==0,0] = 1
y_train[labels==1,1] = 1

X_test = torch.Tensor(np.load("../data//D-II-test.npy"))
y_test = torch.Tensor(np.load("../data/D-II-test-labels.npy"))

data_train = TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

n_models=21
criterion = nn.CrossEntropyLoss()
trained_models_2 = []
for n in range(n_models):
    model = FeedforwardNetwork(3, Hidden_layers['D-II-2'], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=1000)
    trained_models_2.append(model)    

#%%save trained models
# for i, m in enumerate(trained_models_2):
#     torch.save(m.state_dict(), f"../models/D-II-2_model_{i+9}.pth")

#%%Optimiize on sphere dataset
X_train = torch.Tensor(np.load("../data//D-III-train.npy"))
labels = torch.Tensor(np.load("../data/D-III-train-labels.npy"))
y_train = torch.zeros([len(labels),2])
y_train[labels==0,0] = 1
y_train[labels==1,1] = 1

X_test = torch.Tensor(np.load("../data//D-III-test.npy"))
y_test = torch.Tensor(np.load("../data/D-III-test-labels.npy"))

data_train = TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

n_models=30
criterion = nn.CrossEntropyLoss()
trained_models_3 = []
for n in range(n_models):
    model = FeedforwardNetwork(3, Hidden_layers['D-III-2'], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=1000)
    trained_models_3.append(model)    
    
#%%save trained models
# for i, m in enumerate(trained_models_3):
#     torch.save(m.state_dict(), f"../models/D-III-2_model_{i}.pth")
