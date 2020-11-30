import random
from torch.utils.data.sampler import Sampler


class MultiDomainSampler(Sampler):
    def __init__(self, domain_img_list, mode, batch_size, domain_names, random_shuffle):
        super(MultiDomainSampler, self).__init__(domain_img_list)
        self.domain_img_list = domain_img_list[mode]
        self.batch_size = batch_size
        self.domain_names = domain_names
        self.random_shuffle = random_shuffle

        if self.random_shuffle:
            for domain in self.domain_names:
                random.shuffle(self.domain_img_list[domain])

    def __iter__(self):
        for i in range(len(self)):
            batch_idx = []
            for domain in self.domain_names:
                domain_imgs = self.domain_img_list[domain][i * self.batch_size: (i + 1) * self.batch_size]
                batch_idx += domain_imgs
            if i == len(self) - 1 and self.random_shuffle:
                for domain in self.domain_names:
                    random.shuffle(self.domain_img_list[domain])
            yield batch_idx

    def __len__(self):
        l = 1e10
        for domain in self.domain_names:
            d_l = len(self.domain_img_list[domain]) // self.batch_size
            l = min(d_l, l)
        return l