import torch
from torchvision import datasets, transforms
import numpy as np

# get the individual datasets
train_mnist = datasets.MNIST('data', train=True, download=True,
                             transform=transforms.ToTensor())
test_mnist = datasets.MNIST('data', train=False, download=True,
                            transform=transforms.ToTensor())
train_svhn = datasets.SVHN('data', split='train', download=True,
                           transform=transforms.ToTensor())
test_svhn = datasets.SVHN('data', split='test', download=True,
                          transform=transforms.ToTensor())
# svhn labels need extra work
train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

dm = 10                         # data multiplier: random permutations to match

# Get pairs of mnist-svhn data by choosing the min of size for each label
# available:
# for train
mnist_l, mnist_li = train_mnist.targets.sort()
svhn_l, svhn_li = train_svhn.labels.sort()
m_i, s_i = torch.LongTensor(), torch.LongTensor()
for i in np.arange(0, 10):
    c = min(mnist_l.eq(i).sum(), svhn_l.eq(i).sum(), 10000)
    m_c, s_c = mnist_l.lt(i).sum(), svhn_l.lt(i).sum()
    mli, sli = mnist_li[m_c:m_c + c], svhn_li[s_c:s_c + c]
    m_i, s_i = torch.cat([m_i, mli], 0), torch.cat([s_i, sli], 0)
    for i in np.arange(1, dm):
        m_i = torch.cat([m_i, mli[torch.randperm(c)]], 0)
        s_i = torch.cat([s_i, sli[torch.randperm(c)]], 0)

print('len train idx:', len(m_i), len(s_i))
torch.save(m_i, 'data/train-ms-mnist-idx.pt')
torch.save(s_i, 'data/train-ms-svhn-idx.pt')

dm = 10                         # data multiplier: random permutations to match

# for test
mnist_l, mnist_li = test_mnist.targets.sort()
svhn_l, svhn_li = test_svhn.labels.sort()
m_i, s_i = torch.LongTensor(), torch.LongTensor()
for i in np.arange(0, 10):
    c = min(mnist_l.eq(i).sum(), svhn_l.eq(i).sum(), 10000)
    m_c, s_c = mnist_l.lt(i).sum(), svhn_l.lt(i).sum()
    mli, sli = mnist_li[m_c:m_c + c], svhn_li[s_c:s_c + c]
    m_i, s_i = torch.cat([m_i, mli], 0), torch.cat([s_i, sli], 0)
    for i in np.arange(1, dm):
        m_i = torch.cat([m_i, mli[torch.randperm(c)]], 0)
        s_i = torch.cat([s_i, sli[torch.randperm(c)]], 0)

print('len test idx:', len(m_i), len(s_i))
torch.save(m_i, 'data/test-ms-mnist-idx.pt')
torch.save(s_i, 'data/test-ms-svhn-idx.pt')
