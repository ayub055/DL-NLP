# Models
class BILSTM(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0)

        self.lstm = nn.GRU(input_size = 300, hidden_size = 128, num_layers = 2, batch_first = True, bidirectional = True, dropout=0.5)

        self.lin1 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(64, 1)

        self.lin3 = nn.Linear(256, 1)

    def forward(self, xb, tsne = False):

        xe = self.embeddings(xb)
        out, y = self.lstm(xe)
        
        x = self.lin3(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat((x, y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin1(x)

        if tsne == True:
            return x 

        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

class DAN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(600, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 64)
        self.linear5 = nn.Linear(64, 1)

    def forward(self, Xb, tsne = False):
        x = self.linear1(Xb)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x) 
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        
        if tsne == True:
            return x
            
        x = F.relu(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x
