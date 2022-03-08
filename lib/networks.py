import torch
import torch.nn as nn




class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        layers.append(nn.BatchNorm1d(sizes[0]))
        for j in range(1,len(sizes)):
            layers.append(nn.Linear(sizes[j-1], sizes[j]))
            if j<(len(sizes)-1):
                layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(activation())
            else:
                #layers.append(nn.BatchNorm1d(sizes[j]))
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        x = torch.cat(args, -1)

        return self.net(x)

    def fit(self, X, Y, n_epochs=200):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for i in range(n_epochs):
            optimizer.zero_grad()
            pred = self.net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            print("iter: {}, loss: {:.4f}".format(i, loss.item()))




class LinearRegression(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
    
    def forward(self, X):
        return self.linear(X)
    
    def fit(self, X, Y, n_epochs=200):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for i in range(n_epochs):
            optimizer.zero_grad()
            pred = self.linear(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            print("iter: {}, loss: {:.4f}".format(i, loss.item()))

        
