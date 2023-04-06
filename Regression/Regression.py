import math
import torch
from BBB_base import *


# Define training step for regression

# The code defines a function named "train" that takes in the neural network, optimizer, data and target values as input 
# and performs one forward and backward pass of the network on the given data 
# and updates the network parameters.

def train(net, optimizer, data, target, NUM_BATCHES, epoch):
    #net.train()
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))
        loss, ll = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()
        if (epoch)%1000 == 0:
           print("loss", loss)
           print("ll", ll)
        
        
        

#Hyperparameter setting
TRAIN_EPOCHS = 10000
SAMPLES = 5
TEST_SAMPLES = 10
BATCH_SIZE = 200
NUM_BATCHES = 10
TEST_BATCH_SIZE = 50
CLASSES = 1
PI = 0.25
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)

print('Generating Data set.')




Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor


# Data Generation 



# continuous data range
x = np.random.uniform(-0.5, 0.5, size=(NUM_BATCHES,BATCH_SIZE))
x_test = np.linspace(-1, 1,TEST_BATCH_SIZE)

# split cluster
# un-comment this to investigate split cluster data range
'''
x = np.random.uniform(-0.5, 0.5, size=(NUM_BATCHES,BATCH_SIZE))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i][j]>0:
            x[i][j] += 0.5

x_test = np.linspace(-1, 1.5, 75)

'''

noise = np.random.normal(0, 0.02, size=(NUM_BATCHES,BATCH_SIZE)) #metric as mentioned in the paper
def noise_model(x):
    return 0.45*(x+0.5)**2



# HOMOSKEDASTIC REGRESSION from BLUNDELL15
y = x + 0.3*np.sin(2*np.pi*(x+noise)) + 0.3*np.sin(4*np.pi*(x+noise)) + noise
y_test = x_test + 0.3*np.sin(2*np.pi*x_test) + 0.3*np.sin(4*np.pi*x_test)

# HETEROSKEDASTIC REGRESSION
#y_test = -(x_test+0.5)*np.sin(3 * np.pi *x_test)
#y = -(x+0.5)*np.sin(3 * np.pi *x) + np.random.normal(0, noise_model(x))


def BBB_Regression(x,y):

    print('BBB Training')

    X = Var(x)
    Y = Var(y)

    #Declare Network
    # can use BayesianNetworkLRT for Local Reparametrisation trick
    net = BayesianNetwork(inputSize = 1,\
                        CLASSES = CLASSES, \
                        layers=np.array([16,16,16]), \
                        activations = np.array(['relu','relu','relu','none']), \
                        SAMPLES = SAMPLES, \
                        BATCH_SIZE = BATCH_SIZE,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = True,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2,\
                        GOOGLE_INIT= False).to(DEVICE)
    
    #Declare the optimizer
    #optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.95)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for epoch in range(TRAIN_EPOCHS):
        train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES, epoch=epoch)
        if (epoch)%500 == 0:
            print('Epoch: ', epoch)
            torch.save(net.state_dict(), 'Models/Regression_BBB_cont_homo_' +str(epoch)+ '.pth')


    #Save the trained model
    torch.save(net.state_dict(), 'Models/Regression_BBB_cont_homo_10000.pth')




#Comparing to standard neural network
def NN_Regression(x,y,x_test):

    print('SGD Training')

    x = x.flatten()
    X = Var(x)
    X = torch.unsqueeze(X,1)
    
    y = y.flatten()
    Y = Var(y)
    Y = torch.unsqueeze(Y,1)
    X_test = Var(x_test)
    X_test = torch.unsqueeze(X_test,1)

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.l1 = torch.nn.Linear(n_feature, n_hidden)   
            self.l2 =  torch.nn.Linear(n_hidden, n_hidden)   
            self.l3 =  torch.nn.Linear(n_hidden, n_hidden)  
            self.predict = torch.nn.Linear(n_hidden, n_output)   

        def forward(self, x):
            x = F.relu(self.l1(x))     
            x = F.relu(self.l2(x))      
            x = F.relu(self.l3(x))      
            x = self.predict(x)        
            return x

    NN_net = Net(n_feature=1, n_hidden=16, n_output=1)
    

    optimizer = torch.optim.SGD(NN_net.parameters(), lr=0.2)

    for epoch in range(2000):
        pred = NN_net(X)    
        loss = torch.nn.MSELoss(pred, Y)    
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        

    #Save the trained model
    torch.save(NN_net.state_dict(), 'Models/new_data_range/Regression_NN_cont_homo.pth')
    


BBB_Regression(x,y)
NN_Regression(x,y,x_test)