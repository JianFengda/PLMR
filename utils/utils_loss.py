from utils.utils_algo import *
from utils.estimators import *

class partial_loss(nn.Module):
    def forward(self, output1, target, true, dis_label):
        target = target.cuda()
        output = F.softmax(output1, dim=1)
        l = dis_label * target* torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        revisedY = target.clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * output
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

        new_target = revisedY

        return loss,new_target
critic_params = {
    'dim': 64,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'Relu',
}

def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)
BASELINES = {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(dim=64, hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    'gaussian': lambda: log_prob_gaussian,
}

class MILoss(nn.Module):
    def forward(self,features, mask=None,batch_size=-1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.float().detach().to(device)
        critic_fn = SeparableCritic(**critic_params).to(device)
        mi = estimate_mutual_information(features[:batch_size], features[batch_size:], critic_fn,mask)
        loss = -mi
        return loss
class CenterMILoss(nn.Module):
    def forward(self, center,features,mask_c=None,batch_size=-1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_c = mask_c.float().detach().to(device)
        critic_fn = SeparableCritic(**critic_params).to(device)
        c_mi = estimate_mutual_information(features[:batch_size], center, critic_fn,mask_c)
        loss_c = -c_mi
        return loss_c


