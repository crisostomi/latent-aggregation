import math
import torch


class CKA:
    def __init__(self, mode, device):
        self.mode = mode
        self.device = device

    def __call__(self, X, Y, sigma=None):
        X, Y = X.to(self.device), Y.to(self.device)
        if self.mode == "linear":
            return self.linear_CKA(X, Y)
        elif self.mode == "rbf":
            return self.kernel_CKA(X, Y, sigma)
        else:
            return NotImplementedError(f"No such mode {self.mode}")

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity_mat = torch.eye(n, device=self.device)
        H = identity_mat - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
