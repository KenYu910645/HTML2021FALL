import torch.nn as nn

class HTML_NN(nn.Module):
    def __init__(self):
        super(HTML_NN, self).__init__()
        '''
        DON'T ADD ACTIVATION LAYER IN THE LAST LAYER
        '''
        self.mlp = nn.Sequential( # Drop out is not good 
            # 0.36484
            # nn.Linear(43, 43),
            # nn.ReLU(inplace=True),
            # nn.Linear(43, 43),
            # nn.ReLU(inplace=True),
            # nn.Linear(43, 20),
            # nn.ReLU(inplace=True),
            # nn.Linear(20, 10),
            # nn.ReLU(inplace=True),
            # nn.Linear(10, 6),
            
            
            nn.Linear(43, 43),
            nn.ReLU(inplace=True),
            nn.Linear(43, 43),
            nn.ReLU(inplace=True),
            nn.Linear(43, 43),
            nn.ReLU(inplace=True),
            nn.Linear(43, 43),
            nn.ReLU(inplace=True),
            nn.Linear(43, 6),
            
            # nn.Linear(57, 57),
            # nn.ReLU(inplace=True),
            # nn.Linear(57, 57),
            # nn.ReLU(inplace=True),
            # nn.Linear(57, 57),
            # nn.ReLU(inplace=True),
            # nn.Linear(57, 57),
            # nn.ReLU(inplace=True),
            # nn.Linear(57, 6),

            # nn.Linear(43, 6),
        )
    
    def forward(self, x):
        return self.mlp(x)